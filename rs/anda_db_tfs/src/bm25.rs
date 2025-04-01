//! # BM25 Full-Text Search Library
//!
//! This library implements a full-text search engine based on the BM25 ranking algorithm.
//! BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance
//! of documents to a given search query. It's an extension of the TF-IDF model.
//!
//! ## Features
//!
//! - Document indexing with BM25 scoring
//! - Document removal
//! - Query search with top-k results
//! - Serialization and deserialization of indices in CBOR format
//! - Customizable tokenization

use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;
use thiserror::Error;

use crate::tokenizer::*;

/// Errors that can occur during BM25 index operations
#[derive(Error, Debug)]
pub enum BM25Error {
    /// IO errors during read/write operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// CBOR serialization/deserialization errors
    #[error("CBOR serialization error: {0}")]
    Cbor(String),

    /// Error when trying to add a document with an ID that already exists
    #[error("Document {0} already exists")]
    AlreadyExists(u64),

    /// Error when tokenization produces no tokens for a document
    #[error("Document {id} tokenization failed: {text:?}")]
    TokenizeFailed { id: u64, text: String },
}

/// Parameters for the BM25 ranking algorithm
///
/// - `k1`: Controls term frequency saturation. Higher values give more weight to term frequency.
/// - `b`: Controls document length normalization. 0.0 means no normalization, 1.0 means full normalization.
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

/// Internal data structure for the BM25 index
#[derive(Debug, Serialize, Deserialize)]
struct BM25IndexData {
    /// BM25 algorithm parameters
    params: BM25Params,

    /// Average number of tokens per document
    avg_doc_tokens: f32,

    /// Maps document IDs to their token counts
    doc_tokens: HashMap<u64, usize>,

    /// Maps tokens to the number of documents containing them (document frequency)
    token_doc_freq: HashMap<String, usize>,

    /// Maps tokens to their total frequency across all documents
    token_total_freq: HashMap<String, usize>,

    /// Inverted index mapping tokens to (document_id, term_frequency) pairs
    postings: HashMap<String, Vec<(u64, usize)>>,
}

/// BM25 search index with customizable tokenization
#[derive(Clone)]
pub struct BM25Index<T: Tokenizer + Clone> {
    /// Tokenizer used to process text
    tokenizer: T,

    /// Thread-safe shared index data
    data: Arc<RwLock<BM25IndexData>>,
}

impl<T> BM25Index<T>
where
    T: Tokenizer + Clone,
{
    /// Creates a new empty BM25 index with the given tokenizer
    pub fn new(tokenizer: T) -> Self {
        BM25Index {
            tokenizer,
            data: Arc::new(RwLock::new(BM25IndexData {
                token_doc_freq: HashMap::new(),
                token_total_freq: HashMap::new(),
                doc_tokens: HashMap::new(),
                avg_doc_tokens: 0.0,
                postings: HashMap::new(),
                params: BM25Params::default(),
            })),
        }
    }

    /// Sets custom BM25 parameters and returns the modified index
    pub fn with_params(self, params: BM25Params) -> Self {
        self.data.write().params = params;
        self
    }

    /// Adds a document to the index
    ///
    /// # Arguments
    ///
    /// * `id` - Unique document identifier
    /// * `text` - Document text content
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the document was successfully added
    /// * `Err(BM25Error::AlreadyExists)` if a document with the same ID already exists
    /// * `Err(BM25Error::TokenizeFailed)` if tokenization produced no tokens
    pub fn add_document(&self, id: u64, text: &str) -> Result<(), BM25Error> {
        {
            let read_guard = self.data.read();
            if read_guard.doc_tokens.contains_key(&id) {
                return Err(BM25Error::AlreadyExists(id));
            }
        }

        // Tokenize the document
        let tokens = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text)
        };

        // Count token frequencies
        let tokens_len = tokens.len();
        if tokens_len == 0 {
            return Err(BM25Error::TokenizeFailed {
                id,
                text: text.to_string(),
            });
        }

        let mut token_freqs: HashMap<String, usize> = HashMap::with_capacity(tokens_len / 2);
        for token in tokens {
            *token_freqs.entry(token).or_default() += 1;
        }

        let mut data = self.data.write();
        if data.doc_tokens.contains_key(&id) {
            return Err(BM25Error::AlreadyExists(id));
        }
        // Update document length and count
        data.doc_tokens.insert(id, tokens_len);

        // Calculate new average document length
        let total_tokens: usize = data.doc_tokens.values().sum();
        data.avg_doc_tokens = total_tokens as f32 / data.doc_tokens.len() as f32;

        // Update inverted index
        for (token, freq) in token_freqs {
            // Update postings list
            data.postings
                .entry(token.clone())
                .or_default()
                .push((id, freq));

            // Update document frequency (number of documents containing this token)
            *data.token_doc_freq.entry(token.clone()).or_default() += 1;

            // Update total token frequency
            *data.token_total_freq.entry(token).or_default() += freq;
        }

        Ok(())
    }

    /// Removes a document from the index
    ///
    /// # Arguments
    ///
    /// * `id` - Document identifier to remove
    /// * `text` - Original document text (needed to identify tokens to remove)
    ///
    /// # Returns
    ///
    /// * `true` if the document was found and removed
    /// * `false` if the document was not found
    pub fn remove_document(&self, id: u64, text: &str) -> bool {
        {
            if !self.data.read().doc_tokens.contains_key(&id) {
                return false;
            }
        }

        // Tokenize the document
        let tokens = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text)
        };
        // Count token frequencies
        let mut token_freqs: HashMap<String, usize> = HashMap::with_capacity(tokens.len() / 2);
        for token in tokens {
            *token_freqs.entry(token).or_default() += 1;
        }

        let mut data = self.data.write();
        // Check if document exists
        if !data.doc_tokens.contains_key(&id) {
            return false;
        }

        // Remove from inverted index
        let mut tokens_to_remove = Vec::new();
        for (token, freq) in token_freqs {
            if let Some(postings) = data.postings.get_mut(&token) {
                // Remove document from postings list
                if let Some(pos) = postings.iter().position(|&(idx, _)| idx == id) {
                    postings.swap_remove(pos);
                }

                // Update document frequency
                if let Some(df) = data.token_doc_freq.get_mut(&token) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        data.token_doc_freq.remove(&token);
                        tokens_to_remove.push(token.clone());
                    }
                }

                // Update total token frequency
                if let Some(tf) = data.token_total_freq.get_mut(&token) {
                    *tf = tf.saturating_sub(freq);
                    if *tf == 0 {
                        data.token_total_freq.remove(&token);
                    }
                }
            }
        }

        for token in tokens_to_remove {
            data.postings.remove(&token);
        }

        // Remove from document storage
        data.doc_tokens.remove(&id);

        // Recalculate average document length
        let total_tokens: usize = data.doc_tokens.values().sum();
        data.avg_doc_tokens = if data.doc_tokens.is_empty() {
            0.0
        } else {
            total_tokens as f32 / data.doc_tokens.len() as f32
        };

        true
    }

    /// Searches the index for documents matching the query
    ///
    /// # Arguments
    ///
    /// * `query` - Search query text
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (document_id, score) pairs, sorted by descending score
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let data = self.data.read();
        if data.doc_tokens.is_empty() {
            return Vec::new();
        }

        let mut tokenizer = self.tokenizer.clone();
        let query_terms = collect_tokens(&mut tokenizer, query, None);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u64, f32> = HashMap::with_capacity(data.doc_tokens.len().min(1000));

        let term_scores: Vec<HashMap<u64, f32>> = query_terms
            .par_iter()
            .filter_map(|term| {
                data.postings.get(term).map(|postings| {
                    let df = data.token_doc_freq.get(term).cloned().unwrap_or(0) as f32;
                    let idf = ((data.doc_tokens.len() as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let avg_doc_tokens = data.avg_doc_tokens.max(1.0);

                    // 为每个查询词计算文档得分
                    let mut term_scores = HashMap::new();
                    for &(doc_id, tf) in postings {
                        let tokens = data.doc_tokens.get(&doc_id).cloned().unwrap_or(0) as f32;
                        let tf_component = (tf as f32 * (data.params.k1 + 1.0))
                            / (tf as f32
                                + data.params.k1
                                    * (1.0 - data.params.b
                                        + data.params.b * tokens / avg_doc_tokens));

                        let score = idf * tf_component;
                        term_scores.insert(doc_id, score);
                    }
                    term_scores
                })
            })
            .collect();

        // 合并各个查询词的得分
        for term_score in term_scores {
            for (doc_id, score) in term_score {
                *scores.entry(doc_id).or_default() += score;
            }
        }

        // Convert to vector and sort by score (descending)
        let mut sorted_scores: Vec<(u64, f32)> = scores.into_iter().collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_scores.truncate(top_k);
        sorted_scores
    }

    /// Returns the number of documents in the index
    pub fn len(&self) -> usize {
        self.data.read().doc_tokens.len()
    }

    /// Returns whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.data.read().doc_tokens.is_empty()
    }

    /// Serializes the index to a writer
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the `Write` trait
    ///
    /// # Returns
    ///
    /// * `Ok(())` if serialization was successful
    /// * `Err(BM25Error::Cbor)` if serialization failed
    pub fn save<W: Write>(&self, w: W) -> Result<(), BM25Error> {
        let data = self.data.read();
        ciborium::into_writer(&*data, w).map_err(|e| BM25Error::Cbor(e.to_string()))?;
        Ok(())
    }

    /// Deserializes an index from a reader
    ///
    /// # Arguments
    ///
    /// * `r` - Any type implementing the `Read` trait
    /// * `tokenizer` - Tokenizer to use with the loaded index
    ///
    /// # Returns
    ///
    /// * `Ok(BM25Index)` if deserialization was successful
    /// * `Err(BM25Error::Cbor)` if deserialization failed
    pub fn load<R: Read>(r: R, tokenizer: T) -> Result<Self, BM25Error> {
        let index_data: BM25IndexData =
            ciborium::from_reader(r).map_err(|e| BM25Error::Cbor(e.to_string()))?;

        Ok(BM25Index {
            data: Arc::new(RwLock::new(index_data)),
            tokenizer,
        })
    }
}

impl<T> BM25Index<T>
where
    T: Tokenizer + Clone + Send + Sync,
{
    /// Adds multiple documents to the index in parallel
    ///
    /// # Arguments
    ///
    /// * `docs` - Vector of (document_id, document_text) pairs
    ///
    /// # Returns
    ///
    /// A vector of `Ok(())` if the documents were successfully added, or `Err(BM25Error)` if any document failed to add.
    /// The vector is in the same order as the input `docs` vector.
    pub fn add_documents(&self, docs: Vec<(u64, String)>) -> Vec<Result<(), BM25Error>> {
        if docs.is_empty() {
            return Vec::new();
        }

        let existing_ids: Vec<u64> = {
            let read_guard = self.data.read();
            docs.par_iter()
                .filter(|(id, _)| read_guard.doc_tokens.contains_key(id))
                .map(|(id, _)| *id)
                .collect()
        };

        // parallel tokenize
        let mut processed_docs: Vec<(u64, Vec<String>, Result<(), BM25Error>)> = docs
            .par_iter()
            .map(|(id, text)| {
                if existing_ids.contains(id) {
                    return (*id, Vec::new(), Err(BM25Error::AlreadyExists(*id)));
                }

                let mut tokenizer = self.tokenizer.clone();
                let tokens = collect_tokens(&mut tokenizer, text, None);

                if tokens.is_empty() {
                    (
                        *id,
                        Vec::new(),
                        Err(BM25Error::TokenizeFailed {
                            id: *id,
                            text: text.clone(),
                        }),
                    )
                } else {
                    (*id, tokens, Ok(()))
                }
            })
            .collect();

        let has_valid_docs = processed_docs.iter().any(|(_, _, result)| result.is_ok());
        if has_valid_docs {
            let mut data = self.data.write();

            // Check for existing document IDs again
            for (id, tokens, result) in processed_docs.iter_mut() {
                if result.is_ok() && data.doc_tokens.contains_key(id) {
                    tokens.clear();
                    *result = Err(BM25Error::AlreadyExists(*id));
                }
            }

            let valid_docs: Vec<(u64, Vec<String>)> = processed_docs
                .iter()
                .filter_map(|(id, tokens, result)| {
                    if result.is_ok() {
                        Some((*id, tokens.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            // 计算词频
            let doc_tokens_updates: Vec<(u64, usize, HashMap<String, usize>)> = valid_docs
                .par_iter()
                .map(|(id, tokens)| {
                    let tokens_len = tokens.len();
                    let mut token_freqs: HashMap<String, usize> =
                        HashMap::with_capacity(tokens_len / 2);
                    for token in tokens {
                        *token_freqs.entry(token.clone()).or_default() += 1;
                    }
                    (*id, tokens_len, token_freqs)
                })
                .collect();

            for (id, tokens_len, _) in &doc_tokens_updates {
                data.doc_tokens.insert(*id, *tokens_len);
            }

            let total_tokens: usize = data.doc_tokens.values().sum();
            data.avg_doc_tokens = total_tokens as f32 / data.doc_tokens.len() as f32;

            for (id, _, token_freqs) in doc_tokens_updates {
                for (token, freq) in token_freqs {
                    data.postings
                        .entry(token.clone())
                        .or_default()
                        .push((id, freq));

                    *data.token_doc_freq.entry(token.clone()).or_default() += 1;
                    *data.token_total_freq.entry(token).or_default() += freq;
                }
            }
        }

        processed_docs
            .into_iter()
            .map(|(_, _, result)| result)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 创建一个简单的测试索引
    fn create_test_index() -> BM25Index<TokenizerChain> {
        let index = BM25Index::new(default_tokenizer());

        // 添加一些测试文档
        index
            .add_document(1, "The quick brown fox jumps over the lazy dog")
            .unwrap();
        index
            .add_document(2, "A fast brown fox runs past the lazy dog")
            .unwrap();
        index
            .add_document(3, "The lazy dog sleeps all day")
            .unwrap();
        index
            .add_document(4, "Quick brown foxes are rare in the wild")
            .unwrap();

        index
    }

    #[test]
    fn test_add_document() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);

        // 测试添加新文档
        index
            .add_document(5, "A new document about cats and dogs")
            .unwrap();
        assert_eq!(index.len(), 5);

        // 测试添加已存在的文档ID
        let result = index.add_document(3, "This should fail");
        assert!(matches!(result, Err(BM25Error::AlreadyExists(3))));

        // 测试添加空文档
        let result = index.add_document(6, "");
        assert!(matches!(
            result,
            Err(BM25Error::TokenizeFailed { id: 6, .. })
        ));
    }

    #[test]
    fn test_remove_document() {
        let index = create_test_index();
        assert_eq!(index.len(), 4);

        // 测试移除存在的文档
        let removed = index.remove_document(2, "A fast brown fox runs past the lazy dog");
        assert!(removed);
        assert_eq!(index.len(), 3);

        // 测试移除不存在的文档
        let removed = index.remove_document(99, "This document doesn't exist");
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
        let index = BM25Index::new(tokenizer);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        // 测试空索引的搜索
        let results = index.search("test", 10);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_serialization() {
        let index = create_test_index();

        // 创建临时文件
        let mut data: Vec<u8> = Vec::new();

        // 保存索引
        index.save(&mut data).unwrap();

        // 加载索引
        let tokenizer = default_tokenizer();
        let loaded_index = BM25Index::load(&data[..], tokenizer).unwrap();

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
    fn test_add_documents_parallel() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new(tokenizer);

        // 准备多个文档
        let docs = vec![
            (1, "The quick brown fox jumps over the lazy dog".to_string()),
            (2, "A fast brown fox runs past the lazy dog".to_string()),
            (3, "The lazy dog sleeps all day".to_string()),
            (4, "Quick brown foxes are rare in the wild".to_string()),
            (5, "Cats and dogs are common pets".to_string()),
        ];

        // 并行添加文档
        let results = index.add_documents(docs);

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

        let results = index.add_documents(duplicate_docs);
        assert_eq!(results.len(), 2);
        assert!(matches!(results[0], Err(BM25Error::AlreadyExists(3))));
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
        let custom_params = BM25Params { k1: 1.5, b: 0.75 };
        let custom_index = BM25Index::new(tokenizer).with_params(custom_params);

        // 添加相同的文档
        custom_index
            .add_document(1, "The quick brown fox jumps over the lazy dog")
            .unwrap();
        custom_index
            .add_document(2, "A fast brown fox runs past the lazy dog")
            .unwrap();
        custom_index
            .add_document(3, "The lazy dog sleeps all day")
            .unwrap();
        custom_index
            .add_document(4, "Quick brown foxes are rare in the wild")
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
}
