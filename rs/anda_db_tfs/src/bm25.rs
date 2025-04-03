//! # Anda-DB BM25 Full-Text Search Library
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

use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    sync::RwLock,
};

use crate::tokenizer::*;

/// Errors that can occur during BM25 index operations
#[derive(Error, Debug, Clone)]
pub enum BM25Error {
    /// Database-related errors.
    #[error("DB error: {0}")]
    Db(String),

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
#[derive(Debug)]
struct BM25IndexData {
    /// BM25 algorithm parameters
    params: BM25Params,

    /// Average number of tokens per document
    avg_doc_tokens: RwLock<f32>,

    /// Maps document IDs to their token counts
    doc_tokens: DashMap<u64, usize>,

    /// Inverted index mapping tokens to (document_id, term_frequency) pairs
    postings: DashMap<String, Vec<(u64, usize)>>,
}

/// Serializable BM25 index structure (owned version).
#[derive(Clone, Serialize, Deserialize)]
struct BM25IndexDataSerdeOwn {
    params: BM25Params,
    avg_doc_tokens: f32,
    doc_tokens: DashMap<u64, usize>,
    postings: DashMap<String, Vec<(u64, usize)>>,
}

#[derive(Clone, Serialize)]
struct BM25IndexDataSerdeRef<'a> {
    params: &'a BM25Params,
    avg_doc_tokens: f32,
    doc_tokens: &'a DashMap<u64, usize>,
    postings: &'a DashMap<String, Vec<(u64, usize)>>,
}

impl From<BM25IndexDataSerdeOwn> for BM25IndexData {
    fn from(data: BM25IndexDataSerdeOwn) -> Self {
        BM25IndexData {
            params: data.params,
            avg_doc_tokens: RwLock::new(data.avg_doc_tokens),
            doc_tokens: data.doc_tokens,
            postings: data.postings,
        }
    }
}

/// BM25 search index with customizable tokenization
#[derive(Clone)]
pub struct BM25Index<T: Tokenizer + Clone> {
    /// Tokenizer used to process text
    tokenizer: T,

    /// Thread-safe shared index data
    data: Arc<BM25IndexData>,
}

impl<T> BM25Index<T>
where
    T: Tokenizer + Clone,
{
    /// Creates a new empty BM25 index with the given tokenizer and optional parameters.
    pub fn new(tokenizer: T, params: Option<BM25Params>) -> Self {
        BM25Index {
            tokenizer,
            data: Arc::new(BM25IndexData {
                doc_tokens: DashMap::new(),
                avg_doc_tokens: RwLock::new(0.0),
                postings: DashMap::new(),
                params: params.unwrap_or_default(),
            }),
        }
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
    pub async fn add_document(&self, id: u64, text: &str) -> Result<(), BM25Error> {
        if self.data.doc_tokens.contains_key(&id) {
            return Err(BM25Error::AlreadyExists(id));
        }

        // Tokenize the document
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text, None)
        };

        // Count token frequencies
        if token_freqs.is_empty() {
            return Err(BM25Error::TokenizeFailed {
                id,
                text: text.to_string(),
            });
        }

        // Update document length and count
        self.data.doc_tokens.insert(id, token_freqs.values().sum());
        // Update inverted index
        for (token, freq) in token_freqs {
            self.data
                .postings
                .entry(token.clone())
                .or_default()
                .push((id, freq));
        }

        // Calculate new average document length
        let total_tokens: usize = self.data.doc_tokens.iter().map(|r| *r.value()).sum();
        let avg_doc_tokens = total_tokens as f32 / self.data.doc_tokens.len() as f32;
        *self.data.avg_doc_tokens.write().await = avg_doc_tokens;

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
    pub async fn remove_document(&self, id: u64, text: &str) -> bool {
        if !self.data.doc_tokens.contains_key(&id) {
            return false;
        }

        // Tokenize the document
        let token_freqs = {
            let mut tokenizer = self.tokenizer.clone();
            collect_tokens_parallel(&mut tokenizer, text, None)
        };

        // Remove from inverted index
        let mut tokens_to_remove = Vec::new();
        for token in token_freqs.keys() {
            if let Some(mut postings) = self.data.postings.get_mut(token) {
                // Remove document from postings list
                if let Some(pos) = postings.iter().position(|&(idx, _)| idx == id) {
                    postings.swap_remove(pos);
                }

                if postings.is_empty() {
                    tokens_to_remove.push(token);
                }
            }
        }

        // Remove from document storage
        self.data.doc_tokens.remove(&id);

        for token in tokens_to_remove {
            self.data.postings.remove(token);
        }

        // Recalculate average document length
        let total_tokens: usize = self.data.doc_tokens.iter().map(|r| *r.value()).sum();
        let avg_doc_tokens = if self.data.doc_tokens.is_empty() {
            0.0
        } else {
            total_tokens as f32 / self.data.doc_tokens.len() as f32
        };
        *self.data.avg_doc_tokens.write().await = avg_doc_tokens;

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
    pub async fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        let mut tokenizer = self.tokenizer.clone();
        let query_terms = collect_tokens(&mut tokenizer, query, None);
        if query_terms.is_empty() {
            return Vec::new();
        }

        if self.data.doc_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<u64, f32> =
            HashMap::with_capacity(self.data.doc_tokens.len().min(1000));
        let doc_tokens = self.data.doc_tokens.len() as f32;
        let avg_doc_tokens = self.data.avg_doc_tokens.read().await.max(1.0);
        let term_scores: Vec<HashMap<u64, f32>> = query_terms
            .par_iter()
            .filter_map(|(term, _)| {
                self.data.postings.get(term).map(|postings| {
                    let df = postings.len() as f32;
                    let idf = ((doc_tokens - df + 0.5) / (df + 0.5) + 1.0).ln();

                    // compute BM25 score for each document
                    let mut term_scores = HashMap::new();
                    for (doc_id, tf) in postings.iter() {
                        let tokens = self
                            .data
                            .doc_tokens
                            .get(doc_id)
                            .map(|v| *v as f32)
                            .unwrap_or(0.0);
                        let tf_component = (*tf as f32 * (self.data.params.k1 + 1.0))
                            / (*tf as f32
                                + self.data.params.k1
                                    * (1.0 - self.data.params.b
                                        + self.data.params.b * tokens / avg_doc_tokens));

                        let score = idf * tf_component;
                        term_scores.insert(*doc_id, score);
                    }
                    term_scores
                })
            })
            .collect();

        // merge term scores into a single score for each document
        for term_score in term_scores {
            for (doc_id, score) in term_score {
                *scores.entry(doc_id).or_default() += score;
            }
        }

        let mut sorted_scores: Vec<(u64, f32)> = scores.into_iter().collect();

        // Convert to vector and sort by score (descending)
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_scores.truncate(top_k);
        sorted_scores
    }

    /// Returns the number of documents in the index
    pub fn len(&self) -> usize {
        self.data.doc_tokens.len()
    }

    /// Returns whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.data.doc_tokens.is_empty()
    }

    /// Serializes the index to a writer
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the `tokio::io::AsyncWrite` trait
    ///
    /// # Returns
    ///
    /// * `Ok(())` if serialization was successful
    /// * `Err(BM25Error::Cbor)` if serialization failed
    pub async fn save<W: AsyncWrite + Unpin>(&self, mut w: W) -> Result<(), BM25Error> {
        // clone data to avoid holding the lock for a long time
        let serialized_data = {
            let mut buf = Vec::new();
            ciborium::into_writer(
                &BM25IndexDataSerdeRef {
                    params: &self.data.params,
                    avg_doc_tokens: *self.data.avg_doc_tokens.read().await,
                    doc_tokens: &self.data.doc_tokens,
                    postings: &self.data.postings,
                },
                &mut buf,
            )
            .map_err(|e| BM25Error::Cbor(e.to_string()))?;
            buf
        };

        AsyncWriteExt::write_all(&mut w, &serialized_data)
            .await
            .map_err(|e| BM25Error::Db(e.to_string()))?;
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
    pub async fn load<R: AsyncRead + Unpin>(mut r: R, tokenizer: T) -> Result<Self, BM25Error> {
        let data = {
            let mut buf = Vec::new();
            AsyncReadExt::read_to_end(&mut r, &mut buf)
                .await
                .map_err(|e| BM25Error::Db(e.to_string()))?;
            buf
        };

        let index_data: BM25IndexDataSerdeOwn =
            ciborium::from_reader(&data[..]).map_err(|e| BM25Error::Cbor(e.to_string()))?;

        Ok(BM25Index {
            data: Arc::new(index_data.into()),
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
    pub async fn add_documents(&self, docs: Vec<(u64, String)>) -> Vec<Result<(), BM25Error>> {
        if docs.is_empty() {
            return Vec::new();
        }

        let existing_ids: Vec<u64> = docs
            .iter()
            .filter(|(id, _)| self.data.doc_tokens.contains_key(id))
            .map(|(id, _)| *id)
            .collect();

        // parallel tokenize
        #[allow(clippy::type_complexity)]
        let mut processed_docs: Vec<(u64, HashMap<String, usize>, Result<(), BM25Error>)> = docs
            .par_iter()
            .map(|(id, text)| {
                if existing_ids.contains(id) {
                    return (*id, HashMap::new(), Err(BM25Error::AlreadyExists(*id)));
                }

                let mut tokenizer = self.tokenizer.clone();
                let token_freqs = collect_tokens(&mut tokenizer, text, None);

                if token_freqs.is_empty() {
                    (
                        *id,
                        HashMap::new(),
                        Err(BM25Error::TokenizeFailed {
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

        // Check for existing document IDs again
        for (id, token_freqs, result) in processed_docs.iter_mut() {
            if result.is_ok() && self.data.doc_tokens.contains_key(id) {
                token_freqs.clear();
                *result = Err(BM25Error::AlreadyExists(*id));
            }
        }

        let results: Vec<Result<(), BM25Error>> = processed_docs
            .iter()
            .map(|(_, _, result)| result.clone())
            .collect();

        for (id, token_freqs, result) in processed_docs {
            if result.is_ok() {
                let tokens_len = token_freqs.values().sum();
                self.data.doc_tokens.insert(id, tokens_len);
                for (token, freq) in token_freqs {
                    self.data
                        .postings
                        .entry(token.clone())
                        .or_default()
                        .push((id, freq));
                }
            }
        }

        let total_tokens: usize = self.data.doc_tokens.iter().map(|r| *r.value()).sum();
        let avg_doc_tokens = total_tokens as f32 / self.data.doc_tokens.len() as f32;
        *self.data.avg_doc_tokens.write().await = avg_doc_tokens;

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 创建一个简单的测试索引
    async fn create_test_index() -> BM25Index<TokenizerChain> {
        let index = BM25Index::new(default_tokenizer(), None);

        // 添加一些测试文档
        index
            .add_document(1, "The quick brown fox jumps over the lazy dog")
            .await
            .unwrap();
        index
            .add_document(2, "A fast brown fox runs past the lazy dog")
            .await
            .unwrap();
        index
            .add_document(3, "The lazy dog sleeps all day")
            .await
            .unwrap();
        index
            .add_document(4, "Quick brown foxes are rare in the wild")
            .await
            .unwrap();

        index
    }

    #[tokio::test]
    async fn test_add_document() {
        let index = create_test_index().await;
        assert_eq!(index.len(), 4);

        // 测试添加新文档
        index
            .add_document(5, "A new document about cats and dogs")
            .await
            .unwrap();
        assert_eq!(index.len(), 5);

        // 测试添加已存在的文档ID
        let result = index.add_document(3, "This should fail").await;
        assert!(matches!(result, Err(BM25Error::AlreadyExists(3))));

        // 测试添加空文档
        let result = index.add_document(6, "").await;
        assert!(matches!(
            result,
            Err(BM25Error::TokenizeFailed { id: 6, .. })
        ));
    }

    #[tokio::test]
    async fn test_remove_document() {
        let index = create_test_index().await;
        assert_eq!(index.len(), 4);

        // 测试移除存在的文档
        let removed = index
            .remove_document(2, "A fast brown fox runs past the lazy dog")
            .await;
        assert!(removed);
        assert_eq!(index.len(), 3);

        // 测试移除不存在的文档
        let removed = index
            .remove_document(99, "This document doesn't exist")
            .await;
        assert!(!removed);
        assert_eq!(index.len(), 3);
    }

    #[tokio::test]
    async fn test_search() {
        let index = create_test_index().await;

        // 测试基本搜索功能
        let results = index.search("fox", 10).await;
        assert_eq!(results.len(), 3); // 应该找到3个包含"fox"的文档

        // 检查结果排序 - 文档1和2应该排在前面，因为它们都包含"fox"
        assert!(results.iter().any(|(id, _)| *id == 1));
        assert!(results.iter().any(|(id, _)| *id == 2));
        assert!(results.iter().any(|(id, _)| *id == 4));

        // 测试多词搜索
        let results = index.search("quick fox dog", 10).await;
        assert!(results[0].0 == 1); // 文档1应该排在最前面，因为它同时包含"quick", "fox", "dog"

        // 测试top_k限制
        let results = index.search("dog", 2).await;
        assert_eq!(results.len(), 2); // 应该只返回2个结果，尽管有3个文档包含"dog"

        // 测试空查询
        let results = index.search("", 10).await;
        assert_eq!(results.len(), 0);

        // 测试无匹配查询
        let results = index.search("elephant giraffe", 10).await;
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_empty_index() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new(tokenizer, None);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        // 测试空索引的搜索
        let results = index.search("test", 10).await;
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_serialization() {
        let index = create_test_index().await;

        // 创建临时文件
        let mut data: Vec<u8> = Vec::new();

        // 保存索引
        index.save(&mut data).await.unwrap();

        // 加载索引
        let tokenizer = default_tokenizer();
        let loaded_index = BM25Index::load(&data[..], tokenizer).await.unwrap();

        // 验证加载的索引
        assert_eq!(loaded_index.len(), index.len());

        // 验证搜索结果
        let mut original_results = index.search("fox", 10).await;
        let mut loaded_results = loaded_index.search("fox", 10).await;

        assert_eq!(original_results.len(), loaded_results.len());
        original_results.sort_by(|a, b| a.0.cmp(&b.0));
        loaded_results.sort_by(|a, b| a.0.cmp(&b.0));
        // 比较文档ID和分数（允许浮点数有小误差）
        for i in 0..original_results.len() {
            assert_eq!(original_results[i].0, loaded_results[i].0);
            assert!((original_results[i].1 - loaded_results[i].1).abs() < 0.001);
        }
    }

    #[tokio::test]
    async fn test_add_documents_parallel() {
        let tokenizer = default_tokenizer();
        let index = BM25Index::new(tokenizer, None);

        // 准备多个文档
        let docs = vec![
            (1, "The quick brown fox jumps over the lazy dog".to_string()),
            (2, "A fast brown fox runs past the lazy dog".to_string()),
            (3, "The lazy dog sleeps all day".to_string()),
            (4, "Quick brown foxes are rare in the wild".to_string()),
            (5, "Cats and dogs are common pets".to_string()),
        ];

        // 并行添加文档
        let results = index.add_documents(docs).await;

        // 验证结果
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
        }
        assert_eq!(index.len(), 5);

        // 测试搜索
        let search_results = index.search("fox", 10).await;
        assert_eq!(search_results.len(), 3);

        // 测试添加已存在的文档
        let duplicate_docs = vec![
            (3, "This should fail".to_string()),
            (6, "This should succeed".to_string()),
        ];

        let results = index.add_documents(duplicate_docs).await;
        assert_eq!(results.len(), 2);
        assert!(matches!(results[0], Err(BM25Error::AlreadyExists(3))));
        assert!(results[1].is_ok());

        // 验证索引状态
        assert_eq!(index.len(), 6);
    }

    #[tokio::test]
    async fn test_bm25_params() {
        let tokenizer = default_tokenizer();

        // 使用默认参数
        let default_index = create_test_index().await;

        // 使用自定义参数
        let custom_index = BM25Index::new(tokenizer, Some(BM25Params { k1: 1.5, b: 0.75 }));

        // 添加相同的文档
        custom_index
            .add_document(1, "The quick brown fox jumps over the lazy dog")
            .await
            .unwrap();
        custom_index
            .add_document(2, "A fast brown fox runs past the lazy dog")
            .await
            .unwrap();
        custom_index
            .add_document(3, "The lazy dog sleeps all day")
            .await
            .unwrap();
        custom_index
            .add_document(4, "Quick brown foxes are rare in the wild")
            .await
            .unwrap();

        // 搜索相同的查询
        let default_results = default_index.search("fox", 10).await;
        let custom_results = custom_index.search("fox", 10).await;

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
