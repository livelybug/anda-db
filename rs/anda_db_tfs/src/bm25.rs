//! # Anda-DB BM25 Full-Text Search Library

use dashmap::DashMap;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, HashMap},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::*;
use crate::query::*;
use crate::tokenizer::*;

/// BM25 search index with customizable tokenization
pub struct BM25Index<T: Tokenizer + Clone> {
    /// Index name
    name: String,

    /// Tokenizer used to process text
    tokenizer: T,

    /// BM25 algorithm parameters
    config: BM25Config,

    /// Maps segment IDs to their token counts
    seg_tokens: DashMap<u64, usize>,

    /// Inverted index mapping tokens to (version, Vec<(segment_id, term_frequency)>)
    postings: DashMap<String, PostingValue>,

    /// Index metadata.
    metadata: RwLock<BM25Metadata>,

    /// Average number of tokens per segment
    avg_seg_tokens: RwLock<f32>,

    /// Number of search operations performed.
    search_count: AtomicU64,
}

/// configuration parameters for the BM25 ranking algorithm
///
/// - `k1`: Controls term frequency saturation. Higher values give more weight to term frequency.
/// - `b`: Controls segment length normalization. 0.0 means no normalization, 1.0 means full normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    pub k1: f32,
    pub b: f32,
}

impl Default for BM25Config {
    /// Returns default BM25 parameters (k1=1.2, b=0.75) which work well for most use cases
    fn default() -> Self {
        BM25Config { k1: 1.2, b: 0.75 }
    }
}

/// Type alias for posting values: (update version, Vec<(segment_id, token_frequency)>)
pub type PostingValue = (u64, Vec<(u64, usize)>);

/// Mapping function for a posting: (token, posting value) -> Option<R>
pub type PostingMapFn<R> = fn(&str, &PostingValue) -> Option<R>;

/// No-op function for posting mapping.
pub const POSTING_NOOP_FN: PostingMapFn<()> = |_: &str, _: &PostingValue| None;

/// Function to serialize a posting into binary in CBOR format.
pub const POSTING_SERIALIZE_FN: PostingMapFn<Vec<u8>> = |token: &str, val: &PostingValue| {
    let mut buf = Vec::new();
    ciborium::into_writer(&(token, val), &mut buf).expect("Failed to serialize posting value");
    Some(buf)
};

/// Function to retrieve the version of a posting.
pub const NODE_VERSION_FN: PostingMapFn<u64> = |_: &str, val: &PostingValue| Some(val.0);

/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Metadata {
    /// Index name.
    pub name: String,
    /// BM25 algorithm parameters
    pub config: BM25Config,
    /// Index statistics.
    pub stats: BM25Stats,
}

/// Index statistics.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BM25Stats {
    /// Last insertion timestamp (unix ms).
    pub last_inserted: u64,

    /// Last deletion timestamp (unix ms).
    pub last_deleted: u64,

    /// Last saved timestamp (unix ms).
    pub last_saved: u64,

    /// Updated version for the index. It will be incremented when the index is updated.
    pub version: u64,

    /// Number of elements in the index.
    pub num_elements: u64,

    /// Average number of tokens per segment
    pub avg_seg_tokens: f32,

    /// Number of search operations performed.
    pub search_count: u64,

    /// Number of insert operations performed.
    pub insert_count: u64,

    /// Number of delete operations performed.
    pub delete_count: u64,
}

/// Serializable BM25 index structure (owned version).
#[derive(Clone, Serialize, Deserialize)]
struct BM25IndexOwned {
    seg_tokens: DashMap<u64, usize>,
    postings: DashMap<String, PostingValue>,
    metadata: BM25Metadata,
}

#[derive(Clone, Serialize)]
struct BM25IndexRef<'a> {
    seg_tokens: &'a DashMap<u64, usize>,
    postings: &'a DashMap<String, PostingValue>,
    metadata: &'a BM25Metadata,
}

impl<T> BM25Index<T>
where
    T: Tokenizer + Clone,
{
    /// Creates a new empty BM25 index with the given tokenizer and optional config.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the index
    /// * `tokenizer` - Tokenizer to use for processing text
    /// * `config` - Optional BM25 configuration parameters
    ///
    /// # Returns
    ///
    /// * `BM25Index` - A new instance of the BM25 index
    pub fn new(name: String, tokenizer: T, config: Option<BM25Config>) -> Self {
        let config = config.unwrap_or_default();
        BM25Index {
            name: name.clone(),
            tokenizer,
            config: config.clone(),
            seg_tokens: DashMap::new(),
            postings: DashMap::new(),
            metadata: RwLock::new(BM25Metadata {
                name,
                config,
                stats: BM25Stats::default(),
            }),
            avg_seg_tokens: RwLock::new(0.0),
            search_count: AtomicU64::new(0),
        }
    }

    /// Loads an index from a reader
    ///
    /// # Arguments
    ///
    /// * `r` - Any type implementing the [`futures::io::AsyncRead`] trait
    /// * `tokenizer` - Tokenizer to use with the loaded index
    ///
    /// # Returns
    ///
    /// * `Result<(), BM25Error>` - Success or error.
    pub async fn load<R: AsyncRead + Unpin>(mut r: R, tokenizer: T) -> Result<Self, BM25Error> {
        let data = {
            let mut buf = Vec::new();
            AsyncReadExt::read_to_end(&mut r, &mut buf)
                .await
                .map_err(|err| BM25Error::Generic {
                    name: "unknown".to_string(),
                    source: err.into(),
                })?;
            buf
        };

        let index: BM25IndexOwned =
            ciborium::from_reader(&data[..]).map_err(|err| BM25Error::Serialization {
                name: "unknown".to_string(),
                source: err.into(),
            })?;
        let search_count = AtomicU64::new(index.metadata.stats.search_count);
        let avg_seg_tokens = RwLock::new(index.metadata.stats.avg_seg_tokens);

        Ok(BM25Index {
            name: index.metadata.name.clone(),
            tokenizer,
            config: index.metadata.config.clone(),
            seg_tokens: index.seg_tokens,
            postings: index.postings,
            metadata: RwLock::new(index.metadata),
            avg_seg_tokens,
            search_count,
        })
    }

    /// Returns the number of segments in the index
    pub fn len(&self) -> usize {
        self.seg_tokens.len()
    }

    /// Returns whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.seg_tokens.is_empty()
    }

    /// Returns the index name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the index metadata
    pub fn metadata(&self) -> BM25Metadata {
        let mut metadata = self.metadata.read().clone();
        metadata.stats.search_count = self.search_count.load(Ordering::Relaxed);
        metadata.stats.num_elements = self.seg_tokens.len() as u64;
        metadata.stats.avg_seg_tokens = *self.avg_seg_tokens.read();
        metadata
    }

    /// Gets current statistics about the index
    ///
    /// # Returns
    ///
    /// * `IndexStats` - Current statistics
    pub fn stats(&self) -> BM25Stats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.search_count = self.search_count.load(Ordering::Relaxed);
        stats.num_elements = self.seg_tokens.len() as u64;
        stats.avg_seg_tokens = *self.avg_seg_tokens.read();
        stats
    }

    /// Gets all tokens in the index.
    pub fn tokens(&self) -> BTreeSet<String> {
        self.postings.iter().map(|v| v.key().clone()).collect()
    }

    /// Gets a posting by token and applies a function to it.
    pub fn get_posting_with<R, F>(&self, token: &str, f: F) -> Result<Option<R>, BM25Error>
    where
        F: FnOnce(&str, &PostingValue) -> Option<R>,
    {
        self.postings
            .get(token)
            .map(|v| f(token, &v))
            .ok_or_else(|| BM25Error::NotFound {
                name: self.name.clone(),
                token: token.to_string(),
            })
    }

    /// Sets the posting if it is not already present or if the version is newer.
    /// This method is only used to bootstrap the index from persistent storage.
    pub fn set_posting(&self, token: String, value: PostingValue) -> bool {
        match self.postings.entry(token) {
            dashmap::Entry::Occupied(mut v) => {
                let n = v.get_mut();
                if n.0 < value.0 {
                    *n = value;
                    return true;
                }
                false
            }
            dashmap::Entry::Vacant(v) => {
                v.insert(value);
                true
            }
        }
    }

    /// Inserts a segment to the index
    ///
    /// # Arguments
    ///
    /// * `id` - Unique segment identifier
    /// * `text` - Segment text content
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the segment was successfully added
    /// * `Err(BM25Error::AlreadyExists)` if a segment with the same ID already exists
    /// * `Err(BM25Error::TokenizeFailed)` if tokenization produced no tokens
    pub fn insert(&self, id: u64, text: &str, now_ms: u64) -> Result<(), BM25Error> {
        self.insert_with(id, text, now_ms, POSTING_NOOP_FN)
            .map(|_| ())
    }

    /// Inserts a segment to the index with hook function
    ///
    /// # Arguments
    ///
    /// * `id` - Unique segment identifier
    /// * `text` - Segment text content
    /// * `now_ms` - Current timestamp in milliseconds
    /// * `hook` - Function to apply to each token and its posting value
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the segment was successfully added
    /// * `Err(BM25Error::AlreadyExists)` if a segment with the same ID already exists
    /// * `Err(BM25Error::TokenizeFailed)` if tokenization produced no tokens
    pub fn insert_with<R, F>(
        &self,
        id: u64,
        text: &str,
        now_ms: u64,
        hook: F,
    ) -> Result<Vec<(String, R)>, BM25Error>
    where
        F: Fn(&str, &PostingValue) -> Option<R>,
    {
        if self.seg_tokens.contains_key(&id) {
            return Err(BM25Error::AlreadyExists {
                name: self.name.clone(),
                id,
            });
        }

        // Tokenize the segment
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text, None)
        };

        // Count token frequencies
        if token_freqs.is_empty() {
            return Err(BM25Error::TokenizeFailed {
                name: self.name.clone(),
                id,
                text: text.to_string(),
            });
        }

        let docs = self.seg_tokens.len();
        let tokens: usize = token_freqs.values().sum();
        let total_tokens: usize = self.seg_tokens.iter().map(|r| *r.value()).sum();
        let mut updated_posting_hook_result: Vec<(String, R)> =
            Vec::with_capacity(token_freqs.len());
        match self.seg_tokens.entry(id) {
            dashmap::Entry::Occupied(_) => {
                return Err(BM25Error::AlreadyExists {
                    name: self.name.clone(),
                    id,
                });
            }
            dashmap::Entry::Vacant(v) => {
                v.insert(tokens);

                // Update inverted index
                for (token, freq) in token_freqs {
                    let mut entry = self.postings.entry(token.clone()).or_default();
                    entry.0 += 1; // increment update version
                    entry.1.push((id, freq));
                    if let Some(rt) = hook(&token, &entry) {
                        updated_posting_hook_result.push((token, rt));
                    }
                }

                // Calculate new average segment length
                let avg_seg_tokens = (total_tokens + tokens) as f32 / (docs + 1) as f32;
                *self.avg_seg_tokens.write() = avg_seg_tokens;

                self.update_metadata(|m| {
                    m.stats.version += 1;
                    m.stats.last_inserted = now_ms;
                    m.stats.insert_count += 1;
                });
            }
        }

        Ok(updated_posting_hook_result)
    }

    /// Removes a segment from the index
    ///
    /// # Arguments
    ///
    /// * `id` - Segment identifier to remove
    /// * `text` - Original segment text (needed to identify tokens to remove)
    ///
    /// # Returns
    ///
    /// * `true` if the segment was found and removed
    /// * `false` if the segment was not found
    pub fn remove(&self, id: u64, text: &str, now_ms: u64) -> bool {
        let (deleted, _) = self.remove_with(id, text, now_ms, POSTING_NOOP_FN);
        deleted
    }

    /// Removes a segment from the index with hook function.
    ///
    /// # Arguments
    ///
    /// * `id` - Segment identifier to remove
    /// * `text` - Original segment text (needed to identify tokens to remove)
    ///
    /// # Returns
    ///
    /// * `true` if the segment was found and removed
    /// * `false` if the segment was not found
    pub fn remove_with<R, F>(
        &self,
        id: u64,
        text: &str,
        now_ms: u64,
        hook: F,
    ) -> (bool, Vec<(String, R)>)
    where
        F: Fn(&str, &PostingValue) -> Option<R>,
    {
        if self.seg_tokens.remove(&id).is_none() {
            // Segment not found
            return (false, Vec::new());
        }

        // Tokenize the segment
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text, None)
        };

        let mut updated_posting_hook_result: Vec<(String, R)> =
            Vec::with_capacity(token_freqs.len());
        // Remove from inverted index
        let mut tokens_to_remove = Vec::new();
        for (token, _) in token_freqs {
            if let Some(mut posting) = self.postings.get_mut(&token) {
                // Remove segment from postings list
                if let Some(pos) = posting.1.iter().position(|&(idx, _)| idx == id) {
                    posting.0 += 1; // increment update version
                    posting.1.swap_remove(pos);
                    if posting.1.is_empty() {
                        tokens_to_remove.push(token.clone());
                    }
                    if let Some(rt) = hook(&token, &posting) {
                        updated_posting_hook_result.push((token, rt));
                    }
                }
            }
        }

        for token in tokens_to_remove {
            self.postings.remove(&token);
        }

        // Recalculate average segment length
        let total_tokens: usize = self.seg_tokens.iter().map(|r| *r.value()).sum();
        let avg_seg_tokens = if self.seg_tokens.is_empty() {
            0.0
        } else {
            total_tokens as f32 / self.seg_tokens.len() as f32
        };
        *self.avg_seg_tokens.write() = avg_seg_tokens;
        self.update_metadata(|m| {
            m.stats.version += 1;
            m.stats.last_deleted = now_ms;
            m.stats.delete_count += 1;
        });

        (true, updated_posting_hook_result)
    }

    /// Searches the index for segments matching the query
    ///
    /// # Arguments
    ///
    /// * `query` - Search query text
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (segment_id, score) pairs, sorted by descending score
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let scored_docs = self.score_term(query.trim());

        self.search_count.fetch_add(1, Ordering::Relaxed);
        let mut sorted_scores: Vec<(u64, f32)> = scored_docs.into_iter().collect();
        // Convert to vector and sort by score (descending)
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_scores.truncate(top_k);
        sorted_scores
    }

    /// Searches the index for segments matching the query expression
    ///
    /// # Arguments
    ///
    /// * `query` - Search query text, which can include boolean operators (OR, AND, NOT), example:
    ///   `(hello AND world) OR (rust AND NOT java)`
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (segment_id, score) pairs, sorted by descending score
    pub fn search_advanced(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let query_expr = QueryType::parse(query);
        let scored_docs = self.execute_query(&query_expr);

        self.search_count.fetch_add(1, Ordering::Relaxed);
        // Convert to vector and sort by score (descending)
        let mut results: Vec<(u64, f32)> = scored_docs.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }

    /// Execute a query expression, returning a mapping of segment IDs to scores
    fn execute_query(&self, query: &QueryType) -> HashMap<u64, f32> {
        match query {
            QueryType::Term(term) => self.score_term(term),
            QueryType::And(subqueries) => self.score_and(subqueries),
            QueryType::Or(subqueries) => self.score_or(subqueries),
            QueryType::Not(subquery) => self.score_not(subquery),
        }
    }

    /// Scores a single term
    fn score_term(&self, term: &str) -> HashMap<u64, f32> {
        if self.postings.is_empty() {
            return HashMap::new();
        }

        let mut tokenizer = self.tokenizer.clone();
        let query_terms = collect_tokens(&mut tokenizer, term, None);
        if query_terms.is_empty() {
            return HashMap::new();
        }

        let mut scores: HashMap<u64, f32> = HashMap::with_capacity(self.seg_tokens.len().min(1000));
        let seg_tokens = self.seg_tokens.len() as f32;
        let avg_seg_tokens = self.avg_seg_tokens.read().max(1.0);
        let term_scores: Vec<HashMap<u64, f32>> = query_terms
            .par_iter()
            .filter_map(|(term, _)| {
                self.postings.get(term).map(|postings| {
                    let df = postings.1.len() as f32;
                    let idf = ((seg_tokens - df + 0.5) / (df + 0.5) + 1.0).ln();

                    // compute BM25 score for each segment
                    let mut term_scores = HashMap::new();
                    for (doc_id, tf) in postings.1.iter() {
                        let tokens = self
                            .seg_tokens
                            .get(doc_id)
                            .map(|v| *v as f32)
                            .unwrap_or(0.0);
                        let tf_component = (*tf as f32 * (self.config.k1 + 1.0))
                            / (*tf as f32
                                + self.config.k1
                                    * (1.0 - self.config.b
                                        + self.config.b * tokens / avg_seg_tokens));

                        let score = idf * tf_component;
                        term_scores.insert(*doc_id, score);
                    }
                    term_scores
                })
            })
            .collect();

        // merge term scores into a single score for each segment
        for term_score in term_scores {
            for (doc_id, score) in term_score {
                *scores.entry(doc_id).or_default() += score;
            }
        }

        scores
    }

    /// Scores an OR query
    fn score_or(&self, subqueries: &[Box<QueryType>]) -> HashMap<u64, f32> {
        let mut result = HashMap::new();
        if subqueries.is_empty() {
            return result;
        }

        // Execute all subqueries and merge results
        for subquery in subqueries {
            let sub_result = self.execute_query(subquery);

            for (doc_id, score) in sub_result {
                *result.entry(doc_id).or_insert(0.0) += score;
            }
        }

        result
    }

    /// Scores an AND query
    fn score_and(&self, subqueries: &[Box<QueryType>]) -> HashMap<u64, f32> {
        if subqueries.is_empty() {
            return HashMap::new();
        }

        // Execute the first subquery
        let mut result = self.execute_query(&subqueries[0]);
        if result.is_empty() {
            return HashMap::new();
        }

        // Execute the remaining subqueries and intersect the results
        for subquery in &subqueries[1..] {
            let sub_result = self.execute_query(subquery);
            if matches!(subquery.as_ref(), QueryType::Not(_)) {
                // handle NOT query, remove it from the result
                for doc_id in sub_result.keys() {
                    result.remove(doc_id);
                }
                continue;
            }

            // Retain only segments that are in both results
            result.retain(|k, _| sub_result.contains_key(k));
            if result.is_empty() {
                return HashMap::new();
            }

            // Merge scores
            for (doc_id, score) in sub_result {
                result.entry(doc_id).and_modify(|s| *s += score);
            }
        }

        result
    }

    /// Scores a NOT query
    fn score_not(&self, subquery: &QueryType) -> HashMap<u64, f32> {
        self.execute_query(subquery)
    }

    /// Stores the index without postings to a writer.
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`futures::io::AsyncWrite`] trait
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), BM25Error>` - Success or error.
    pub async fn store<W: AsyncWrite + Unpin>(
        &self,
        mut w: W,
        now_ms: u64,
    ) -> Result<(), BM25Error> {
        // clone data to avoid holding the lock for a long time
        let serialized_data = {
            let mut buf = Vec::with_capacity(8192);
            self.update_metadata(|m| {
                m.stats.last_saved = now_ms.max(m.stats.last_saved);
            });
            ciborium::into_writer(
                &BM25IndexRef {
                    seg_tokens: &self.seg_tokens,
                    postings: &DashMap::new(),
                    metadata: &self.metadata(),
                },
                &mut buf,
            )
            .map_err(|err| BM25Error::Serialization {
                name: self.name.clone(),
                source: err.into(),
            })?;
            buf
        };

        AsyncWriteExt::write_all(&mut w, &serialized_data)
            .await
            .map_err(|err| BM25Error::Generic {
                name: self.name.clone(),
                source: err.into(),
            })?;

        Ok(())
    }

    /// Stores the index with postings to a writer. It maybe very large.
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`futures::io::AsyncWrite`] trait
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), BM25Error>` - Success or error.
    pub async fn store_all<W: AsyncWrite + Unpin>(
        &self,
        mut w: W,
        now_ms: u64,
    ) -> Result<(), BM25Error> {
        // clone data to avoid holding the lock for a long time
        let serialized_data = {
            let mut buf = Vec::new();
            self.update_metadata(|m| {
                m.stats.last_saved = now_ms.max(m.stats.last_saved);
            });
            ciborium::into_writer(
                &BM25IndexRef {
                    seg_tokens: &self.seg_tokens,
                    postings: &self.postings,
                    metadata: &self.metadata(),
                },
                &mut buf,
            )
            .map_err(|err| BM25Error::Serialization {
                name: self.name.clone(),
                source: err.into(),
            })?;
            buf
        };

        AsyncWriteExt::write_all(&mut w, &serialized_data)
            .await
            .map_err(|err| BM25Error::Generic {
                name: self.name.clone(),
                source: err.into(),
            })?;

        Ok(())
    }

    /// Updates the index metadata
    ///
    /// # Arguments
    ///
    /// * `f` - Function that modifies the metadata
    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut BM25Metadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

impl<T> BM25Index<T>
where
    T: Tokenizer + Clone + Send + Sync,
{
    /// Inserts multiple segments to the index in parallel
    ///
    /// # Arguments
    ///
    /// * `docs` - Vector of (segment_id, segment_text) pairs
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// A vector of `Ok(())` if the segments were successfully added, or `Err(BM25Error)` if any segment failed to add.
    /// The vector is in the same order as the input `docs` vector.
    pub fn insert_batch(
        &self,
        docs: Vec<(u64, String)>,
        now_ms: u64,
    ) -> Vec<Result<(), BM25Error>> {
        if docs.is_empty() {
            return Vec::new();
        }

        let existing_ids: Vec<u64> = docs
            .iter()
            .filter(|(id, _)| self.seg_tokens.contains_key(id))
            .map(|(id, _)| *id)
            .collect();

        // parallel tokenize
        #[allow(clippy::type_complexity)]
        let mut processed_docs: Vec<(u64, HashMap<String, usize>, Result<(), BM25Error>)> = docs
            .par_iter()
            .map(|(id, text)| {
                if existing_ids.contains(id) {
                    return (
                        *id,
                        HashMap::new(),
                        Err(BM25Error::AlreadyExists {
                            name: self.name.clone(),
                            id: *id,
                        }),
                    );
                }

                let mut tokenizer = self.tokenizer.clone();
                let token_freqs = collect_tokens(&mut tokenizer, text, None);

                if token_freqs.is_empty() {
                    (
                        *id,
                        HashMap::new(),
                        Err(BM25Error::TokenizeFailed {
                            name: self.name.clone(),
                            id: *id,
                            text: text.clone(),
                        }),
                    )
                } else {
                    (*id, token_freqs, Ok(()))
                }
            })
            .collect();

        let has_valid_docs = processed_docs.iter().any(|(_, _, result)| result.is_ok());
        if !has_valid_docs {
            return processed_docs
                .into_iter()
                .map(|(_, _, result)| result)
                .collect::<Vec<_>>();
        }

        // Check for existing segment IDs again
        for (id, token_freqs, result) in processed_docs.iter_mut() {
            if result.is_ok() && self.seg_tokens.contains_key(id) {
                token_freqs.clear();
                *result = Err(BM25Error::AlreadyExists {
                    name: self.name.clone(),
                    id: *id,
                });
            }
        }

        // split into valid docs and results
        #[allow(clippy::type_complexity)]
        let (docs, results): (
            Vec<(u64, HashMap<String, usize>, bool)>,
            Vec<Result<(), BM25Error>>,
        ) = processed_docs
            .into_iter()
            .map(|(id, token_freqs, result)| ((id, token_freqs, result.is_ok()), result))
            .collect();

        let mut insert_count = 0u64;
        for (id, token_freqs, ok) in docs {
            if ok {
                insert_count += 1;
                let tokens_len = token_freqs.values().sum();
                self.seg_tokens.insert(id, tokens_len);
                for (token, freq) in token_freqs {
                    let mut entry = self.postings.entry(token).or_default();
                    entry.0 += 1; // increment update version
                    entry.1.push((id, freq));
                }
            }
        }

        let total_tokens: usize = self.seg_tokens.iter().map(|r| *r.value()).sum();
        let avg_seg_tokens = total_tokens as f32 / self.seg_tokens.len() as f32;
        *self.avg_seg_tokens.write() = avg_seg_tokens;

        self.update_metadata(|m| {
            m.stats.version += 1;
            m.stats.last_inserted = now_ms;
            m.stats.insert_count += insert_count;
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 创建一个简单的测试索引
    fn create_test_index() -> BM25Index<TokenizerChain> {
        let index = BM25Index::new("anda_db_tfs_bm25".to_string(), default_tokenizer(), None);

        // 添加一些测试文档
        index
            .insert(1, "The quick brown fox jumps over the lazy dog", 0)
            .unwrap();
        index
            .insert(2, "A fast brown fox runs past the lazy dog", 0)
            .unwrap();
        index.insert(3, "The lazy dog sleeps all day", 0).unwrap();
        index
            .insert(4, "Quick brown foxes are rare in the wild", 0)
            .unwrap();

        index
    }

    #[test]
    fn test_insert() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);

        // 测试添加新文档
        index
            .insert(5, "A new segment about cats and dogs", 0)
            .unwrap();
        assert_eq!(index.len(), 5);

        // 测试添加已存在的文档ID
        let result = index.insert(3, "This should fail", 0);
        assert!(matches!(
            result,
            Err(BM25Error::AlreadyExists { id: 3, .. })
        ));

        // 测试添加空文档
        let result = index.insert(6, "", 0);
        assert!(matches!(
            result,
            Err(BM25Error::TokenizeFailed { id: 6, .. })
        ));
    }

    #[test]
    fn test_remove() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);

        // 测试移除存在的文档
        let removed = index.remove(2, "A fast brown fox runs past the lazy dog", 0);
        assert!(removed);
        assert_eq!(index.len(), 3);

        // 测试移除不存在的文档
        let removed = index.remove(99, "This segment doesn't exist", 0);
        assert!(!removed);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_search() {
        let index = create_test_index();

        // 测试基本搜索功能
        let results = index.search("fox", 10);
        assert_eq!(results.len(), 3); // 应该找到3个包含"fox"的文档

        // 检查结果排序 - 文档1和2应该排在前面，因为它们都包含"fox"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试多词搜索
        let results = index.search("quick fox dog", 10);
        assert!(results[0].0 == 1); // 文档1应该排在最前面，因为它同时包含"quick", "fox", "dog"

        // 测试top_k限制
        let results = index.search("dog", 2);
        assert_eq!(results.len(), 2); // 应该只返回2个结果，尽管有3个文档包含"dog"

        // 测试空查询
        let results = index.search("", 10);
        assert_eq!(results.len(), 0);

        // 测试无匹配查询
        let results = index.search("elephant giraffe", 10);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_empty_index() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new("anda_db_tfs_bm25".to_string(), tokenizer, None);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        // 测试空索引的搜索
        let results = index.search("test", 10);
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_serialization() {
        let index = create_test_index();

        // 创建临时文件
        let mut data: Vec<u8> = Vec::new();

        // 保存索引
        index.store_all(&mut data, 0).await.unwrap();

        // 加载索引
        let tokenizer = default_tokenizer();
        let loaded_index = BM25Index::load(&data[..], tokenizer).await.unwrap();

        // 验证加载的索引
        assert_eq!(loaded_index.len(), index.len());

        // 验证搜索结果
        let mut original_results = index.search("fox", 10);
        let mut loaded_results = loaded_index.search("fox", 10);

        assert_eq!(original_results.len(), loaded_results.len());
        original_results.sort_by(|a, b| a.0.cmp(&b.0));
        loaded_results.sort_by(|a, b| a.0.cmp(&b.0));
        // 比较文档ID和分数（允许浮点数有小误差）
        for i in 0..original_results.len() {
            assert_eq!(original_results[i].0, loaded_results[i].0);
            assert!((original_results[i].1 - loaded_results[i].1).abs() < 0.001);
        }
    }

    #[test]
    fn test_insert_batch_parallel() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new("anda_db_tfs_bm25".to_string(), tokenizer, None);

        // 准备多个文档
        let docs = vec![
            (1, "The quick brown fox jumps over the lazy dog".to_string()),
            (2, "A fast brown fox runs past the lazy dog".to_string()),
            (3, "The lazy dog sleeps all day".to_string()),
            (4, "Quick brown foxes are rare in the wild".to_string()),
            (5, "Cats and dogs are common pets".to_string()),
        ];

        // 并行添加文档
        let results = index.insert_batch(docs, 0);

        // 验证结果
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
        }
        assert_eq!(index.len(), 5);

        // 测试搜索
        let search_results = index.search("fox", 10);
        assert_eq!(search_results.len(), 3);

        // 测试添加已存在的文档
        let duplicate_docs = vec![
            (3, "This should fail".to_string()),
            (6, "This should succeed".to_string()),
        ];

        let results = index.insert_batch(duplicate_docs, 0);
        assert_eq!(results.len(), 2);
        assert!(matches!(
            results[0],
            Err(BM25Error::AlreadyExists { id: 3, .. })
        ));
        assert!(results[1].is_ok());

        // 验证索引状态
        assert_eq!(index.len(), 6);
    }

    #[test]
    fn test_bm25_params() {
        let tokenizer = default_tokenizer();

        // 使用默认参数
        let default_index = create_test_index();

        // 使用自定义参数
        let custom_index = BM25Index::new(
            "anda_db_tfs_bm25".to_string(),
            tokenizer,
            Some(BM25Config { k1: 1.5, b: 0.75 }),
        );

        // 添加相同的文档
        custom_index
            .insert(1, "The quick brown fox jumps over the lazy dog", 0)
            .unwrap();
        custom_index
            .insert(2, "A fast brown fox runs past the lazy dog", 0)
            .unwrap();
        custom_index
            .insert(3, "The lazy dog sleeps all day", 0)
            .unwrap();
        custom_index
            .insert(4, "Quick brown foxes are rare in the wild", 0)
            .unwrap();

        // 搜索相同的查询
        let default_results = default_index.search("fox", 10);
        let custom_results = custom_index.search("fox", 10);

        // 验证结果数量相同但分数不同
        assert_eq!(default_results.len(), custom_results.len());

        // 至少有一个文档的分数应该不同
        let mut scores_different = false;
        for i in 0..default_results.len() {
            if (default_results[i].1 - custom_results[i].1).abs() > 0.001 {
                scores_different = true;
                break;
            }
        }
        assert!(scores_different);
    }

    #[test]
    fn test_search_advanced() {
        let index = create_test_index();

        // 测试简单的 Term 查询
        let results = index.search_advanced("fox", 10);
        assert_eq!(results.len(), 3); // 应该找到3个包含"fox"的文档

        // 测试 AND 查询
        let results = index.search_advanced("fox AND lazy", 10);
        assert_eq!(results.len(), 2); // 文档1和2同时包含"fox"和"lazy"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));

        // 测试 OR 查询
        let results = index.search_advanced("quick OR fast", 10);
        assert_eq!(results.len(), 3); // 文档1包含"quick"，文档2包含"fast"，文档4包含"quick"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试 NOT 查询
        let results = index.search_advanced("dog AND NOT lazy", 10);
        assert_eq!(results.len(), 0); // 所有包含"dog"的文档也包含"lazy"

        // 测试复杂的嵌套查询
        let results = index.search_advanced("(quick OR fast) AND fox", 10);
        assert_eq!(results.len(), 3); // 文档1、2和4

        // 测试更复杂的嵌套查询
        let results = index.search_advanced("(brown AND fox) AND NOT (rare OR sleeps)", 10);
        assert_eq!(results.len(), 2); // 文档1和2，排除了包含"rare"的文档4和包含"sleeps"的文档3
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));

        // 测试空查询
        let results = index.search_advanced("", 10);
        assert_eq!(results.len(), 0);

        // 测试无匹配查询
        let results = index.search_advanced("elephant AND giraffe", 10);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_advanced_with_parentheses() {
        let index = create_test_index();

        // 测试带括号的复杂查询
        let results = index.search_advanced("(fox AND quick) OR (dog AND sleeps)", 10);
        assert_eq!(results.len(), 3); // 文档1, 3, 4
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 3));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试多层嵌套括号
        let results =
            index.search_advanced("((brown AND fox) OR (lazy AND sleeps)) AND NOT rare", 10);
        assert_eq!(results.len(), 3); // 文档1、2和3，排除了包含"rare"的文档4
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 3));

        // 测试带括号的 NOT 查询
        let results = index.search_advanced("dog AND NOT (quick OR fast)", 10);
        assert_eq!(results.len(), 1); // 只有文档3，因为它包含"dog"但不包含"quick"或"fast"
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_search_advanced_score_ordering() {
        let index = create_test_index();

        // 测试分数排序 - 包含更多匹配词的文档应该排在前面
        let results = index.search_advanced("quick OR fox OR dog", 10);
        assert!(results.len() >= 3);

        // 文档1应该排在最前面，因为它同时包含所有三个词
        assert_eq!(results[0].0, 1);

        // 测试 top_k 限制
        let results = index.search_advanced("dog", 2);
        assert_eq!(results.len(), 2); // 应该只返回2个结果，尽管有3个文档包含"dog"
    }

    #[test]
    fn test_search_vs_search_advanced() {
        let index = create_test_index();

        // 对于简单查询，search 和 search_advanced 应该返回相似的结果
        let simple_results = index.search("fox", 10);
        let advanced_results = index.search_advanced("fox", 10);

        assert_eq!(simple_results.len(), advanced_results.len());

        // 检查文档ID是否匹配（不检查分数，因为实现可能略有不同）
        let simple_ids: Vec<u64> = simple_results.iter().map(|(id, _)| *id).collect();
        let advanced_ids: Vec<u64> = advanced_results.iter().map(|(id, _)| *id).collect();

        assert_eq!(simple_ids.len(), advanced_ids.len());
        for id in simple_ids {
            assert!(advanced_ids.contains(&id));
        }

        // 测试多词查询 - search 将它们视为 OR，search_advanced 也应该如此
        let simple_results = index.search("quick fox", 10);
        let advanced_results = index.search_advanced("quick OR fox", 10);

        // 检查文档ID是否匹配
        let simple_ids: Vec<u64> = simple_results.iter().map(|(id, _)| *id).collect();
        let advanced_ids: Vec<u64> = advanced_results.iter().map(|(id, _)| *id).collect();

        assert_eq!(simple_ids.len(), advanced_ids.len());
        for id in simple_ids {
            assert!(advanced_ids.contains(&id));
        }
    }
}
