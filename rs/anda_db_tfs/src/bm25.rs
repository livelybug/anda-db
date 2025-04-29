//! # Anda-DB BM25 Full-Text Search Library

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::{Read, Write},
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

use crate::error::*;
use crate::query::*;
use crate::tokenizer::*;

const SEGMENTS_BUCKET_SIZE: u64 = 100_000;

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

    /// Buckets store information about where posting entries are stored and their current state
    /// The mapping is: bucket_id -> (bucket_size, is_dirty, vec<tokens>)
    /// - bucket_size: Current size of the bucket in bytes
    /// - is_dirty: Indicates if the bucket has new data that needs to be persisted
    /// - tokens: List of tokens stored in this bucket
    buckets: DashMap<u32, (u32, bool, Vec<String>)>,

    /// Inverted index mapping tokens to (bucket id, Vec<(segment_id, term_frequency)>)
    postings: DashMap<String, PostingValue>,

    /// Index metadata.
    metadata: RwLock<BM25Metadata>,

    /// Maximum bucket ID currently in use
    max_bucket_id: AtomicU32,

    /// Maximum segment ID currently in use
    max_segment_id: AtomicU64,

    /// Set of dirty segment buckets that need to be persisted
    dirty_segment_buckets: RwLock<BTreeSet<u32>>,

    /// Maximum size of a bucket before creating a new one
    /// When a bucket's stored data exceeds this size,
    /// a new bucket should be created for new data
    bucket_overload_size: u32,

    /// Average number of tokens per segment
    avg_seg_tokens: RwLock<f32>,

    /// Number of search operations performed.
    search_count: AtomicU64,

    /// Last saved version of the index
    last_saved_version: AtomicU64,
}

/// Parameters for the BM25 ranking algorithm
///
/// - `k1`: Controls term frequency saturation. Higher values give more weight to term frequency.
/// - `b`: Controls segment length normalization. 0.0 means no normalization, 1.0 means full normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for BM25Params {
    /// Returns default BM25 parameters (k1=1.2, b=0.75) which work well for most use cases
    fn default() -> Self {
        BM25Params { k1: 1.2, b: 0.75 }
    }
}

/// Configuration parameters for the BM25 index
///
/// - `bm25`: BM25 algorithm parameters
/// - `bucket_overload_size`: Maximum size of a bucket before creating a new one (in bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25Config {
    pub bm25: BM25Params,
    /// Maximum size of a bucket before creating a new one (in bytes)
    pub bucket_overload_size: u32,
}

impl Default for BM25Config {
    /// Returns default BM25 parameters (k1=1.2, b=0.75) which work well for most use cases
    fn default() -> Self {
        BM25Config {
            bm25: BM25Params::default(),
            bucket_overload_size: 1024 * 512,
        }
    }
}

/// Type alias for posting values: (bucket id, Vec<(segment_id, token_frequency)>)
/// - bucket_id: The bucket where this posting is stored
/// - Vec<(segment_id, token_frequency)>: List of segments and their term frequencies
pub type PostingValue = (u32, Vec<(u64, usize)>);

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

    /// Number of search operations performed.
    pub search_count: u64,

    /// Number of insert operations performed.
    pub insert_count: u64,

    /// Number of delete operations performed.
    pub delete_count: u64,

    /// Maximum bucket ID currently in use
    pub max_bucket_id: u32,

    /// Maximum segment ID currently in use
    pub max_segment_id: u64,

    /// Average number of tokens per segment
    pub avg_seg_tokens: f32,
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
        let bucket_overload_size = config.bucket_overload_size;
        let mut stats = BM25Stats::default();
        stats.version = 1;
        BM25Index {
            name: name.clone(),
            tokenizer,
            config: config.clone(),
            seg_tokens: DashMap::new(),
            postings: DashMap::new(),
            buckets: DashMap::from_iter(vec![(0, (0, true, Vec::new()))]),
            metadata: RwLock::new(BM25Metadata {
                name,
                config,
                stats,
            }),
            bucket_overload_size,
            max_bucket_id: AtomicU32::new(0),
            max_segment_id: AtomicU64::new(0),
            dirty_segment_buckets: RwLock::new(BTreeSet::new()),
            avg_seg_tokens: RwLock::new(0.0),
            search_count: AtomicU64::new(0),
            last_saved_version: AtomicU64::new(0),
        }
    }

    /// Loads an index from metadata reader and closure for loading segments and postings.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - Tokenizer to use for processing text
    /// * `metadata` - Metadata reader
    /// * `f1` - Closure for loading segments
    /// * `f2` - Closure for loading postings
    ///
    /// # Returns
    ///
    /// * `Result<Self, HnswError>` - Loaded index or error.
    pub async fn load_all<R: Read, F1, F2>(
        tokenizer: T,
        metadata: R,
        segments_fn: F1,
        postings_fn: F2,
    ) -> Result<Self, BM25Error>
    where
        F1: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
        F2: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
    {
        let mut index = Self::load_metadata(tokenizer, metadata)?;
        index.load_segments(segments_fn).await?;
        index.load_postings(postings_fn).await?;
        Ok(index)
    }

    /// Loads an index from a reader
    /// This only loads metadata, you need to call [`Self::load_buckets`] to load the actual posting data.
    ///
    /// # Arguments
    ///
    /// * `tokenizer` - Tokenizer to use with the loaded index
    /// * `r` - Any type implementing the [`Read`] trait
    ///
    /// # Returns
    ///
    /// * `Result<(), BM25Error>` - Success or error.
    pub fn load_metadata<R: Read>(tokenizer: T, r: R) -> Result<Self, BM25Error> {
        let index: BM25IndexOwned =
            ciborium::from_reader(r).map_err(|err| BM25Error::Serialization {
                name: "unknown".to_string(),
                source: err.into(),
            })?;
        let bucket_overload_size = index.metadata.config.bucket_overload_size;
        let max_bucket_id = AtomicU32::new(index.metadata.stats.max_bucket_id);
        let max_segment_id = AtomicU64::new(index.metadata.stats.max_segment_id);
        let search_count = AtomicU64::new(index.metadata.stats.search_count);
        let avg_seg_tokens = RwLock::new(index.metadata.stats.avg_seg_tokens);
        let last_saved_version = AtomicU64::new(index.metadata.stats.version);

        Ok(BM25Index {
            name: index.metadata.name.clone(),
            tokenizer,
            config: index.metadata.config.clone(),
            seg_tokens: index.seg_tokens,
            postings: index.postings,
            buckets: DashMap::from_iter(vec![(0, (0, true, Vec::new()))]),
            metadata: RwLock::new(index.metadata),
            dirty_segment_buckets: RwLock::new(BTreeSet::new()),
            bucket_overload_size,
            max_bucket_id,
            max_segment_id,
            avg_seg_tokens,
            search_count,
            last_saved_version,
        })
    }

    /// Loads segments from buckets using the provided async function
    /// This function should be called during database startup to load all segment data
    /// and form a complete segment index
    ///
    /// # Arguments
    ///
    /// * `f` - Async function that reads posting data from a specified bucket.
    ///   `F: AsyncFn(u64) -> Result<Vec<u8>, BTreeError>`
    ///   The function should take a bucket ID as input and return a vector of bytes
    ///   containing the serialized segment data.
    ///
    /// # Returns
    ///
    /// * `Result<(), BTreeError>` - Success or error
    pub async fn load_segments<F>(&mut self, mut f: F) -> Result<(), BM25Error>
    where
        F: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
    {
        for i in 0..=self.max_segment_id.load(Ordering::Relaxed) / SEGMENTS_BUCKET_SIZE {
            let data = f(i as u32).await.map_err(|err| BM25Error::Generic {
                name: self.name.clone(),
                source: err,
            })?;
            if let Some(data) = data {
                let segments: HashMap<u64, usize> =
                    ciborium::from_reader(&data[..]).map_err(|err| BM25Error::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    })?;

                self.seg_tokens.extend(segments);
            }
        }

        Ok(())
    }

    /// Loads posting data from buckets using the provided async function
    /// This function should be called during database startup to load all bucket posting data
    /// and form a complete posting index
    ///
    /// # Arguments
    ///
    /// * `f` - Async function that reads posting data from a specified bucket.
    ///   `F: AsyncFn(u32) -> Result<Option<Vec<u8>>, BTreeError>`
    ///   The function should take a bucket ID as input and return a vector of bytes
    ///   containing the serialized posting data.
    ///
    /// # Returns
    ///
    /// * `Result<(), BTreeError>` - Success or error
    pub async fn load_postings<F>(&mut self, mut f: F) -> Result<(), BM25Error>
    where
        F: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
    {
        for i in 0..=self.max_bucket_id.load(Ordering::Relaxed) {
            let data = f(i).await.map_err(|err| BM25Error::Generic {
                name: self.name.clone(),
                source: err,
            })?;
            if let Some(data) = data {
                let postings: HashMap<String, PostingValue> = ciborium::from_reader(&data[..])
                    .map_err(|err| BM25Error::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    })?;
                let bks = postings.keys().cloned().collect::<Vec<_>>();
                // Update bucket information
                // Larger buckets have the most recent state and can override smaller buckets
                self.buckets.insert(i, (data.len() as u32, false, bks));
                self.postings.extend(postings);
            }
        }

        Ok(())
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
        metadata.stats.max_bucket_id = self.max_bucket_id.load(Ordering::Relaxed);
        metadata.stats.max_segment_id = self.max_segment_id.load(Ordering::Relaxed);
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
        stats.max_bucket_id = self.max_bucket_id.load(Ordering::Relaxed);
        stats.max_segment_id = self.max_segment_id.load(Ordering::Relaxed);
        stats.avg_seg_tokens = *self.avg_seg_tokens.read();
        stats
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
    /// * `Err(BM25Error)` if failed
    pub fn insert(&self, id: u64, text: &str, now_ms: u64) -> Result<(), BM25Error> {
        if self.seg_tokens.contains_key(&id) {
            return Err(BM25Error::AlreadyExists {
                name: self.name.clone(),
                id,
            });
        }

        // Tokenize the segment
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens(&mut tokenizer, text, None)
        };

        // Count token frequencies
        if token_freqs.is_empty() {
            return Err(BM25Error::TokenizeFailed {
                name: self.name.clone(),
                id,
                text: text.to_string(),
            });
        }

        let _ = self.max_segment_id.fetch_max(id, Ordering::Relaxed).max(id);

        // Phase 1: Update the postings collection
        let bucket = self.max_bucket_id.load(Ordering::Acquire);
        let docs = self.seg_tokens.len();
        let tokens: usize = token_freqs.values().sum();
        let total_tokens: usize = self.seg_tokens.iter().map(|r| *r.value()).sum();
        // buckets_to_update: BTreeMap<bucketid, BTreeMap<token, size_increase>>
        let mut buckets_to_update: BTreeMap<u32, BTreeMap<String, u32>> = BTreeMap::new();
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
                    match self.postings.entry(token.clone()) {
                        dashmap::Entry::Occupied(mut entry) => {
                            let val = (id, freq);
                            let size_increase = CountingWriter::count_cbor(&val) + 2;
                            let e = entry.get_mut();
                            e.1.push(val);
                            let b = buckets_to_update.entry(e.0).or_default();
                            b.insert(token, size_increase as u32);
                        }
                        dashmap::Entry::Vacant(entry) => {
                            // Create new posting
                            let val = (bucket, vec![(id, freq)]);
                            let size_increase = CountingWriter::count_cbor(&val) + 2;
                            entry.insert(val);
                            let b = buckets_to_update.entry(bucket).or_default();
                            b.insert(token, size_increase as u32);
                        }
                    };
                }

                // Calculate new average segment length
                let avg_seg_tokens = (total_tokens + tokens) as f32 / (docs + 1) as f32;
                *self.avg_seg_tokens.write() = avg_seg_tokens;
                self.dirty_segment_buckets
                    .write()
                    .insert((id / SEGMENTS_BUCKET_SIZE) as u32);

                self.update_metadata(|m| {
                    m.stats.version += 1;
                    m.stats.last_inserted = now_ms;
                    m.stats.insert_count += 1;
                });
            }
        }

        // Phase 2: Update bucket states
        // tokens_to_migrate: (old_bucket_id, token, size)
        let mut tokens_to_migrate: Vec<(u32, String, u32)> = Vec::new();
        for (id, val) in buckets_to_update {
            let mut bucket = self.buckets.entry(id).or_default();
            // Mark as dirty, needs to be persisted
            bucket.1 = true;
            for (token, size) in val {
                if bucket.2.is_empty() || bucket.0 + size < self.bucket_overload_size {
                    bucket.0 += size;
                    // Add field value to bucket if not already present
                    if !bucket.2.contains(&token) {
                        bucket.2.push(token);
                    }
                } else {
                    tokens_to_migrate.push((id, token, size));
                }
            }
        }

        // Phase 3: Create new buckets if needed
        if !tokens_to_migrate.is_empty() {
            let mut next_bucket_id = self.max_bucket_id.fetch_add(1, Ordering::Release) + 1;

            {
                self.buckets
                    .entry(next_bucket_id)
                    .or_insert_with(|| (0, true, Vec::new()));
                // release the lock on the entry
            }

            for (old_bucket_id, token, size) in tokens_to_migrate {
                if let Some(mut posting) = self.postings.get_mut(&token) {
                    posting.0 = next_bucket_id;
                }

                if let Some(mut ob) = self.buckets.get_mut(&old_bucket_id) {
                    if let Some(pos) = ob.2.iter().position(|k| &token == k) {
                        ob.0 = ob.0.saturating_sub(size);
                        // ob.1 = true; // do not need to set dirty
                        ob.2.swap_remove(pos);
                    }
                }

                let mut new_bucket = false;
                if let Some(mut nb) = self.buckets.get_mut(&next_bucket_id) {
                    if nb.2.is_empty() || nb.0 + size < self.bucket_overload_size {
                        // Bucket has enough space, update directly
                        nb.0 += size;
                        if !nb.2.contains(&token) {
                            nb.2.push(token.clone());
                        }
                    } else {
                        // Bucket doesn't have enough space, need to migrate to the next bucket
                        new_bucket = true;
                    }
                }

                if new_bucket {
                    next_bucket_id = self.max_bucket_id.fetch_add(1, Ordering::Release) + 1;
                    // update the posting's bucket_id again
                    if let Some(mut posting) = self.postings.get_mut(&token) {
                        posting.0 = next_bucket_id;
                    }

                    match self.buckets.entry(next_bucket_id) {
                        dashmap::Entry::Vacant(entry) => {
                            // Create a new bucket with the initial size
                            entry.insert((size, true, vec![token]));
                        }
                        dashmap::Entry::Occupied(mut entry) => {
                            let bucket_entry = entry.get_mut();
                            bucket_entry.0 += size;
                            bucket_entry.1 = true; // Mark as dirty
                            if !bucket_entry.2.contains(&token) {
                                bucket_entry.2.push(token);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Removes a segment from the index
    ///
    /// # Arguments
    ///
    /// * `id` - Segment identifier to remove
    /// * `text` - Original segment text (needed to identify tokens to remove)
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `true` if the segment was found and removed
    /// * `false` if the segment was not found
    pub fn remove(&self, id: u64, text: &str, now_ms: u64) -> bool {
        if self.seg_tokens.remove(&id).is_none() {
            // Segment not found
            return false;
        }

        // Tokenize the segment
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens(&mut tokenizer, text, None)
        };

        // buckets_to_update: BTreeMap<bucketid, BTreeMap<token, size_decrease>>
        let mut buckets_to_update: BTreeMap<u32, BTreeMap<String, u32>> = BTreeMap::new();
        // Remove from inverted index
        let mut tokens_to_remove = Vec::new();
        for (token, _) in token_freqs {
            if let Some(mut posting) = self.postings.get_mut(&token) {
                // Remove segment from postings list
                if let Some(pos) = posting.1.iter().position(|&(idx, _)| idx == id) {
                    let val = posting.1.swap_remove(pos);
                    let mut size_decrease = CountingWriter::count_cbor(&val) + 2;
                    if posting.1.is_empty() {
                        size_decrease += CountingWriter::count_cbor(&token) + 2;
                        tokens_to_remove.push(token.clone());
                    }
                    let b = buckets_to_update.entry(posting.0).or_default();
                    b.insert(token, size_decrease as u32);
                }
            }
        }

        for (id, val) in buckets_to_update {
            if let Some(mut b) = self.buckets.get_mut(&id) {
                // Mark as dirty, needs to be persisted
                // bucket.1 = true; // do not need to set dirty
                for (token, size_decrease) in val {
                    b.0 = b.0.saturating_sub(size_decrease);
                    if tokens_to_remove.contains(&token) {
                        if let Some(pos) = b.2.iter().position(|k| &token == k) {
                            b.2.swap_remove(pos);
                        }
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
        self.dirty_segment_buckets
            .write()
            .insert((id / SEGMENTS_BUCKET_SIZE) as u32);

        self.update_metadata(|m| {
            m.stats.version += 1;
            m.stats.last_deleted = now_ms;
            m.stats.delete_count += 1;
        });

        true
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
    pub fn search(&self, query: &str, top_k: usize, params: Option<BM25Params>) -> Vec<(u64, f32)> {
        let params = params.as_ref().unwrap_or(&self.config.bm25);
        let scored_docs = self.score_term(query.trim(), params);

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
    pub fn search_advanced(
        &self,
        query: &str,
        top_k: usize,
        params: Option<BM25Params>,
    ) -> Vec<(u64, f32)> {
        let query_expr = QueryType::parse(query);
        let params = params.as_ref().unwrap_or(&self.config.bm25);
        let scored_docs = self.execute_query(&query_expr, params);

        self.search_count.fetch_add(1, Ordering::Relaxed);
        // Convert to vector and sort by score (descending)
        let mut results: Vec<(u64, f32)> = scored_docs.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }

    /// Execute a query expression, returning a mapping of segment IDs to scores
    fn execute_query(&self, query: &QueryType, params: &BM25Params) -> HashMap<u64, f32> {
        match query {
            QueryType::Term(term) => self.score_term(term, params),
            QueryType::And(subqueries) => self.score_and(subqueries, params),
            QueryType::Or(subqueries) => self.score_or(subqueries, params),
            QueryType::Not(subquery) => self.score_not(subquery, params),
        }
    }

    /// Scores a single term
    fn score_term(&self, term: &str, params: &BM25Params) -> HashMap<u64, f32> {
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
            .iter()
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
                        let tf_component = (*tf as f32 * (params.k1 + 1.0))
                            / (*tf as f32
                                + params.k1
                                    * (1.0 - params.b + params.b * tokens / avg_seg_tokens));

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
    fn score_or(&self, subqueries: &[Box<QueryType>], params: &BM25Params) -> HashMap<u64, f32> {
        let mut result = HashMap::new();
        if subqueries.is_empty() {
            return result;
        }

        // Execute all subqueries and merge results
        for subquery in subqueries {
            let sub_result = self.execute_query(subquery, params);

            for (doc_id, score) in sub_result {
                *result.entry(doc_id).or_insert(0.0) += score;
            }
        }

        result
    }

    /// Scores an AND query
    fn score_and(&self, subqueries: &[Box<QueryType>], params: &BM25Params) -> HashMap<u64, f32> {
        if subqueries.is_empty() {
            return HashMap::new();
        }

        // Execute the first subquery
        let mut result = self.execute_query(&subqueries[0], params);
        if result.is_empty() {
            return HashMap::new();
        }

        // Execute the remaining subqueries and intersect the results
        for subquery in &subqueries[1..] {
            let sub_result = self.execute_query(subquery, params);
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
    fn score_not(&self, subquery: &QueryType, params: &BM25Params) -> HashMap<u64, f32> {
        self.execute_query(subquery, params)
    }

    /// Stores the index metadata, IDs and nodes to persistent storage.
    pub async fn flush<W: Write, F1, F2>(
        &self,
        metadata: W,
        now_ms: u64,
        segments_fn: F1,
        postings_fn: F2,
    ) -> Result<bool, BM25Error>
    where
        F1: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
        F2: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
    {
        if !self.store_metadata(metadata, now_ms)? {
            return Ok(false);
        }

        self.store_dirty_segments(segments_fn).await?;
        self.store_dirty_postings(postings_fn).await?;
        Ok(true)
    }

    /// Stores the index metadata to a writer in CBOR format.
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`Write`] trait
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<bool, BM25Error>` - true if the metadata was saved, false if the version was not updated
    pub fn store_metadata<W: Write>(&self, w: W, now_ms: u64) -> Result<bool, BM25Error> {
        let mut meta = self.metadata();
        let prev_saved_version = self
            .last_saved_version
            .fetch_max(meta.stats.version, Ordering::Release);
        if prev_saved_version >= meta.stats.version {
            // No need to save if the version is not updated
            return Ok(false);
        }

        meta.stats.last_saved = now_ms.max(meta.stats.last_saved);

        ciborium::into_writer(
            &BM25IndexRef {
                seg_tokens: &self.seg_tokens,
                postings: &DashMap::new(),
                metadata: &self.metadata(),
            },
            w,
        )
        .map_err(|err| BM25Error::Serialization {
            name: self.name.clone(),
            source: err.into(),
        })?;

        self.update_metadata(|m| {
            m.stats.last_saved = meta.stats.last_saved.max(m.stats.last_saved);
        });

        Ok(true)
    }

    /// Stores dirty segments to persistent storage using the provided async function
    pub async fn store_dirty_segments<F>(&self, mut f: F) -> Result<(), BM25Error>
    where
        F: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
    {
        let mut dirty_segment_buckets = BTreeSet::new();
        {
            // move the dirty nodes into a temporary variable
            // and release the lock
            dirty_segment_buckets.append(&mut self.dirty_segment_buckets.write());
        }

        let mut buf = Vec::with_capacity(1024 * 256);
        while let Some(bucket) = dirty_segment_buckets.pop_first() {
            let mut segments = HashMap::new();
            for id in
                (bucket as u64 * SEGMENTS_BUCKET_SIZE)..((bucket as u64 + 1) * SEGMENTS_BUCKET_SIZE)
            {
                if let Some(val) = self.seg_tokens.get(&id) {
                    segments.insert(id, *val);
                }
            }

            if segments.is_empty() {
                continue;
            }

            buf.clear();
            ciborium::into_writer(&segments, &mut buf).expect("Failed to serialize node");
            match f(bucket, &buf).await {
                Ok(true) => {
                    // continue
                }
                Ok(false) => {
                    // stop and refund the unprocessed dirty nodes
                    self.dirty_segment_buckets
                        .write()
                        .append(&mut dirty_segment_buckets);
                    return Ok(());
                }
                Err(err) => {
                    // refund the unprocessed dirty nodes
                    self.dirty_segment_buckets
                        .write()
                        .append(&mut dirty_segment_buckets);
                    return Err(BM25Error::Generic {
                        name: self.name.clone(),
                        source: err,
                    });
                }
            }
        }

        Ok(())
    }

    /// Stores dirty postings to persistent storage using the provided async function
    ///
    /// This method iterates through all posting buckets and persists those that have been modified
    /// since the last save operation.
    ///
    /// # Arguments
    ///
    /// * `f` - Async function that writes bucket data to persistent storage
    ///   The function takes a bucket ID and serialized data, and returns whether to continue
    ///
    /// # Returns
    ///
    /// * `Result<(), BTreeError>` - Success or error
    pub async fn store_dirty_postings<F>(&self, mut f: F) -> Result<(), BM25Error>
    where
        F: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
    {
        let mut buf = Vec::with_capacity(8192);
        for mut bucket in self.buckets.iter_mut() {
            if bucket.1 {
                // If the bucket is dirty, it needs to be persisted
                let mut postings = HashMap::with_capacity(bucket.2.len());
                for k in bucket.2.iter() {
                    if let Some(posting) = self.postings.get(k) {
                        postings.insert(
                            k,
                            ciborium::cbor!(posting).map_err(|err| BM25Error::Serialization {
                                name: self.name.clone(),
                                source: err.into(),
                            })?,
                        );
                    }
                }

                buf.clear();
                ciborium::into_writer(&postings, &mut buf).map_err(|err| {
                    BM25Error::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    }
                })?;

                if let Ok(conti) = f(*bucket.key(), &buf).await {
                    // Only mark as clean if persistence was successful, otherwise wait for next round
                    bucket.1 = false;
                    if !conti {
                        return Ok(());
                    }
                }
            }
        }

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

/// Utility for counting the size of serialized CBOR data
pub struct CountingWriter {
    count: usize,
}

impl Default for CountingWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CountingWriter {
    pub fn new() -> Self {
        CountingWriter { count: 0 }
    }

    pub fn size(&self) -> usize {
        self.count
    }

    // TODO: refactor this function to use a more efficient way to count the size
    pub fn count_cbor(val: &impl Serialize) -> usize {
        let mut writer = CountingWriter::new();
        let _ = ciborium::into_writer(val, &mut writer);
        writer.count
    }
}

impl std::io::Write for CountingWriter {
    /// Implements the write method for the Write trait
    /// This simply counts the bytes without actually writing them
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = buf.len();
        self.count += len;
        Ok(len)
    }

    /// Implements the flush method for the Write trait
    /// This is a no-op since we're not actually writing data
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
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
        let results = index.search("fox", 10, None);
        assert_eq!(results.len(), 3); // 应该找到3个包含"fox"的文档

        // 检查结果排序 - 文档1和2应该排在前面，因为它们都包含"fox"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试多词搜索
        let results = index.search("quick fox dog", 10, None);
        assert!(results[0].0 == 1); // 文档1应该排在最前面，因为它同时包含"quick", "fox", "dog"

        // 测试top_k限制
        let results = index.search("dog", 2, None);
        assert_eq!(results.len(), 2); // 应该只返回2个结果，尽管有3个文档包含"dog"

        // 测试空查询
        let results = index.search("", 10, None);
        assert_eq!(results.len(), 0);

        // 测试无匹配查询
        let results = index.search("elephant giraffe", 10, None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_empty_index() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new("anda_db_tfs_bm25".to_string(), tokenizer, None);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        // 测试空索引的搜索
        let results = index.search("test", 10, None);
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_serialization() {
        let index = create_test_index();

        // 创建临时文件
        let mut metadata: Vec<u8> = Vec::new();
        let mut segments: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut postings: HashMap<u32, Vec<u8>> = HashMap::new();

        // 保存索引
        index
            .flush(
                &mut metadata,
                0,
                async |id: u32, data: &[u8]| {
                    segments.insert(id, data.to_vec());
                    Ok(true)
                },
                async |id: u32, data: &[u8]| {
                    postings.insert(id, data.to_vec());
                    Ok(true)
                },
            )
            .await
            .unwrap();

        // 加载索引
        let tokenizer = default_tokenizer();
        let loaded_index = BM25Index::load_all(
            tokenizer,
            &metadata[..],
            async |id| Ok(segments.get(&id).cloned()),
            async |id| Ok(postings.get(&id).cloned()),
        )
        .await
        .unwrap();

        // 验证加载的索引
        assert_eq!(loaded_index.len(), index.len());

        // 验证搜索结果
        let mut original_results = index.search("fox", 10, None);
        let mut loaded_results = loaded_index.search("fox", 10, None);

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
    fn test_bm25_params() {
        // 使用默认参数
        let default_index = create_test_index();

        // 搜索相同的查询
        let default_results = default_index.search("fox", 10, None);
        let custom_results = default_index.search("fox", 10, Some(BM25Params { k1: 1.5, b: 0.75 }));

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
        let results = index.search_advanced("fox", 10, None);
        assert_eq!(results.len(), 3); // 应该找到3个包含"fox"的文档

        // 测试 AND 查询
        let results = index.search_advanced("fox AND lazy", 10, None);
        assert_eq!(results.len(), 2); // 文档1和2同时包含"fox"和"lazy"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));

        // 测试 OR 查询
        let results = index.search_advanced("quick OR fast", 10, None);
        assert_eq!(results.len(), 3); // 文档1包含"quick"，文档2包含"fast"，文档4包含"quick"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试 NOT 查询
        let results = index.search_advanced("dog AND NOT lazy", 10, None);
        assert_eq!(results.len(), 0); // 所有包含"dog"的文档也包含"lazy"

        // 测试复杂的嵌套查询
        let results = index.search_advanced("(quick OR fast) AND fox", 10, None);
        assert_eq!(results.len(), 3); // 文档1、2和4

        // 测试更复杂的嵌套查询
        let results = index.search_advanced("(brown AND fox) AND NOT (rare OR sleeps)", 10, None);
        assert_eq!(results.len(), 2); // 文档1和2，排除了包含"rare"的文档4和包含"sleeps"的文档3
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));

        // 测试空查询
        let results = index.search_advanced("", 10, None);
        assert_eq!(results.len(), 0);

        // 测试无匹配查询
        let results = index.search_advanced("elephant AND giraffe", 10, None);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_advanced_with_parentheses() {
        let index = create_test_index();

        // 测试带括号的复杂查询
        let results = index.search_advanced("(fox AND quick) OR (dog AND sleeps)", 10, None);
        assert_eq!(results.len(), 3); // 文档1, 3, 4
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 3));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试多层嵌套括号
        let results = index.search_advanced(
            "((brown AND fox) OR (lazy AND sleeps)) AND NOT rare",
            10,
            None,
        );
        assert_eq!(results.len(), 3); // 文档1、2和3，排除了包含"rare"的文档4
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 3));

        // 测试带括号的 NOT 查询
        let results = index.search_advanced("dog AND NOT (quick OR fast)", 10, None);
        assert_eq!(results.len(), 1); // 只有文档3，因为它包含"dog"但不包含"quick"或"fast"
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_search_advanced_score_ordering() {
        let index = create_test_index();

        // 测试分数排序 - 包含更多匹配词的文档应该排在前面
        let results = index.search_advanced("quick OR fox OR dog", 10, None);
        assert!(results.len() >= 3);

        // 文档1应该排在最前面，因为它同时包含所有三个词
        assert_eq!(results[0].0, 1);

        // 测试 top_k 限制
        let results = index.search_advanced("dog", 2, None);
        assert_eq!(results.len(), 2); // 应该只返回2个结果，尽管有3个文档包含"dog"
    }

    #[test]
    fn test_search_vs_search_advanced() {
        let index = create_test_index();

        // 对于简单查询，search 和 search_advanced 应该返回相似的结果
        let simple_results = index.search("fox", 10, None);
        let advanced_results = index.search_advanced("fox", 10, None);

        assert_eq!(simple_results.len(), advanced_results.len());

        // 检查文档ID是否匹配（不检查分数，因为实现可能略有不同）
        let simple_ids: Vec<u64> = simple_results.iter().map(|(id, _)| *id).collect();
        let advanced_ids: Vec<u64> = advanced_results.iter().map(|(id, _)| *id).collect();

        assert_eq!(simple_ids.len(), advanced_ids.len());
        for id in simple_ids {
            assert!(advanced_ids.contains(&id));
        }

        // 测试多词查询 - search 将它们视为 OR，search_advanced 也应该如此
        let simple_results = index.search("quick fox", 10, None);
        let advanced_results = index.search_advanced("quick OR fox", 10, None);

        // 检查文档ID是否匹配
        let simple_ids: Vec<u64> = simple_results.iter().map(|(id, _)| *id).collect();
        let advanced_ids: Vec<u64> = advanced_results.iter().map(|(id, _)| *id).collect();

        assert_eq!(simple_ids.len(), advanced_ids.len());
        for id in simple_ids {
            assert!(advanced_ids.contains(&id));
        }
    }
}
