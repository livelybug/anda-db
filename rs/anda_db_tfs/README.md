# Anda-DB BM25 Full-Text Search Library

[![Crates.io](https://img.shields.io/crates/v/anda_db_tfs)](https://crates.io/crates/anda_db_tfs)
[![Documentation](https://docs.rs/anda_db_tfs/badge.svg)](https://docs.rs/anda_db_tfs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/ldclabs/anda-db/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda-db/actions)

`anda_db_tfs` is a full-text search library implementing the BM25 ranking algorithm in Rust. BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of segments to a given search query. It's an extension of the TF-IDF model.

## Features

- **High Performance**: Optimized for speed with parallel processing using Rayon.
- **Customizable Tokenization**: Support for various tokenizers including Chinese text via jieba.
- **BM25 Ranking**: Industry-standard relevance scoring algorithm.
- **Serialization**: Save and load indices in CBOR format with optional compression.
- **Incremental Persistent**: Support incremental index updates persistent (insertions and deletions)
- **Thread-safe concurrent access**: Safely use the index from multiple threads

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anda_db_tfs = "0.3"
```

For full features including tantivy tokenizers and jieba support:

```toml
[dependencies]
anda_db_tfs = { version = "0.3", features = ["full"] }
```

## Quick Start

```rust
use anda_db_tfs::{BM25Index, SimpleTokenizer};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

// Create a new index with a simple tokenizer
let index = BM25Index::new("my_bm25_index".to_string(), SimpleTokenizer::default(), None);

// Add segments to the index
index.insert(1, "The quick brown fox jumps over the lazy dog", now_ms).unwrap();
index.insert(2, "A fast brown fox runs past the lazy dog", now_ms).unwrap();
index.insert(3, "The lazy dog sleeps all day", now_ms).unwrap();

// Search for segments containing "fox"
let results = index.search("fox", 10);
for (seg_id, score) in results {
    println!("Segment {}: score {}", seg_id, score);
}

// Remove a segment
index.remove(3, "The lazy dog sleeps all day", now_ms);

// Store the index to a file
let file = tokio::fs::File::create("index.cbor").await.unwrap().compat_write();
index.store_all(file, now_ms).await.unwrap();

// Load the index from a file
let file = tokio::fs::File::open("index.cbor").await.unwrap().compat();
let loaded_index = BM25Index::load(file, SimpleTokenizer::default()).await.unwrap();
```

## Chinese Text Support

With the `tantivy-jieba` feature enabled, you can use the jieba tokenizer for Chinese text:

```rust
use anda_db_tfs::{BM25Index, jieba_tokenizer};

// Create an index with jieba tokenizer
let index = BM25Index::new("my_bm25_index".to_string(), jieba_tokenizer(), None);

// Add segments with Chinese text
index.insert(1, "Rust 是一种系统编程语言", now_ms).unwrap();
index.insert(2, "Rust 快速且内存高效，安全、并发、实用", now_ms).unwrap();

// Search for segments
let results = index.search("安全", 10);
```

## Advanced Usage

### Custom Tokenizer and BM25 Parameters

```rust
use anda_db_tfs::{BM25Index, BM25Config};
use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer};

// Create an index with custom BM25 parameters
let params = BM25Config { k1: 1.5, b: 0.75 };
let index_name = "my_custom_index".to_string();
let tokenizer = TokenizerChain::builder(SimpleTokenizer::default())
  .filter(RemoveLongFilter::limit(32))
  .filter(LowerCaser)
  .filter(Stemmer::default())
  .build();
let index = BM25Index::new(index_name, tokenizer, Some(params));
```

### Batch Segment Processing

```rust
use anda_db_tfs::{BM25Index, default_tokenizer};

let index = BM25Index::new("my_index".to_string(), default_tokenizer(), None);

// Prepare multiple segments
let docs = vec![
    (1, "Segment one content".to_string()),
    (2, "Segment two content".to_string()),
    (3, "Segment three content".to_string()),
];

// Add segments in batch
let results = index.insert(docs, now_ms);
```

## API Documentation

👉 https://docs.rs/anda_db_tfs

### BM25Index

The main struct for creating and managing a search index.

```rust
// Creates a new index
pub fn new(name: String, tokenizer: T, config: Option<BM25Config>) -> Self;

// Loads the index from a reader
pub async fn load<R: AsyncWrite>(r: R, tokenizer: T) -> Result<Self, BM25Error>;

// Gets the number of segments in the index
pub fn len(&self) -> usize;

// Checks if the index is empty
pub fn is_empty(&self) -> bool;

/// Returns the index name
pub fn name(&self) -> &str;

// Returns the index metadata
pub fn metadata(&self) -> BM25Metadata;

// Gets current statistics about the index
pub fn stats(&self) -> BM25Stats;

// Gets all tokens in the index.
pub fn tokens(&self) -> BTreeSet<String>;

// Gets a posting by token and applies a function to it.
pub fn get_posting_with<R, F>(&self, token: &str, f: F) -> Result<Option<R>, BM25Error>
where
    F: FnOnce(&str, &PostingValue) -> Option<R>;

// Sets the posting if it is not already present or if the version is newer.
// This method is only used to bootstrap the index from persistent storage.
pub fn set_posting(&self, token: String, value: PostingValue) -> bool;

// Inserts a segment to the index
pub fn insert(&self, id: u64, text: &str, now_ms: u64) -> Result<(), BM25Error>;

// Inserts a segment to the index with hook function
pub fn insert_with<R, F>(
    &self,
    id: u64,
    text: &str,
    now_ms: u64,
    hook: F,
) -> Result<Vec<(String, R)>, BM25Error>
where
    F: Fn(&str, &PostingValue) -> Option<R>;

// Inserts multiple segments to the index in parallel
pub fn insert_batch(&self, docs: Vec<(u64, String)>, now_ms: u64) -> Vec<Result<(), BM25Error>>;

// Removes a segment from the index
pub fn remove(&self, id: u64, text: &str, now_ms: u64) -> bool;

// Removes a segment from the index with hook function.
pub fn remove_with<R, F>(
    &self,
    id: u64,
    text: &str,
    now_ms: u64,
    hook: F,
) -> (bool, Vec<(String, R)>)
where
    F: Fn(&str, &PostingValue) -> Option<R>;

// Searches the index
pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)>;

/// Searches the index for segments matching the query expression,
/// which can include boolean operators (AND, OR, NOT).
pub fn search_advanced(&self, query: &str, top_k: usize) -> Vec<(u64, f32)>;

// Stores the index without postings to a writer.
pub async fn store<W: AsyncRead>(&self, w: W, now_ms: u64) -> Result<(), BM25Error>;

// Stores the index with postings to a writer. It maybe very large.
pub async fn store_all<W: AsyncRead>(&self, w: W, now_ms: u64) -> Result<(), BM25Error>;
```

### BM25Config

Parameters for the BM25 ranking algorithm.

```rust
pub struct BM25Config {
    // Controls term frequency saturation
    pub k1: f32,
    // Controls segment length normalization
    pub b: f32,
}
```

Default values: `k1 = 1.2, b = 0.75`

## Error Handling

The library uses a custom error type `BM25Error` for various error conditions:

- `BM25Error::Db`: Database-related errors.
- `BM25Error::Cbor`: Serialization/deserialization errors.
- `BM25Error::AlreadyExists`: When trying to add a segment with an ID that already exists.
- `BM25Error::TokenizeFailed`: When tokenization produces no tokens for a segment.

## Performance Considerations

- For large segments, the library automatically uses parallel processing for tokenization.
- The search function uses parallel processing for query terms.
- For best performance with large indices, consider using SSD storage for serialized indices.
- Memory usage scales with the number of segments and unique terms.

## License
Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](../../LICENSE) for the full license text.