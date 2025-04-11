# Anda-DB HNSW Vector Search Library

[![Crates.io](https://img.shields.io/crates/v/anda_db_hnsw)](https://crates.io/crates/anda_db_hnsw)
[![Documentation](https://docs.rs/anda_db_hnsw/badge.svg)](https://docs.rs/anda_db_hnsw)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/ldclabs/anda-db/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda-db/actions)

A high-performance implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search in high-dimensional spaces.

## Features

- Fast approximate nearest neighbor search
- Multiple distance metrics (Euclidean, Cosine, Inner Product, Manhattan)
- Configurable index parameters
- Thread-safe implementation with concurrent read/write operations
- Serialization and deserialization in CBOR format
- Support for bf16 (brain floating point) vector storage for memory efficiency
- Support incremental index updates persistent (insertions and deletions)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anda_db_hnsw = "0.3"
```

## Quick Start

```rust
use anda_db_hnsw::{HnswConfig, HnswIndex};

// Create a new index for 384-dimensional vectors
const DIM: usize = 384;
let config = HnswConfig {
    dimension: DIM,
    ..Default::default()
};
let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

// Insert vectors
let vector1: Vec<f32> = vec![0.1, 0.2, /* ... */]; // 384 dimensions
index.insert_f32(1, vector1, now_ms)?;

// Search for nearest neighbors
let query: Vec<f32> = vec![0.15, 0.25, /* ... */]; // 384 dimensions
let results = index.search_f32(&query, 10)?;

// Results contain (id, distance) pairs
for (id, distance) in results {
    println!("ID: {}, Distance: {}", id, distance);
}

// Save index to file
let file = tokio::fs::File::create("index.cbor")?;
index.save(file, now_ms).await?;

// Load index from file
let file = tokio::fs::File::open("index.cbor")?;
let loaded_index = HnswIndex::load(file).await?;
```

## API Documentation

👉 https://docs.rs/anda_db_hnsw

### HnswIndex

The main struct for creating and managing a search index.

```rust
// Creates a new HNSW index.
pub fn new(name: String, config: Option<HnswConfig>) -> Self;

// Loads an index from a reader.
pub async fn load<R: AsyncRead + Unpin>(mut r: R) -> Result<Self, HnswError>;

// Returns the number of vectors in the index.
pub fn len(&self) -> usize;

// Checks if the index is empty
pub fn is_empty(&self) -> bool;

/// Returns the index name
pub fn name(&self) -> &str;

// Returns the dimensionality of vectors in the index
pub fn dimension(&self) -> usize;

// Returns the index metadata
pub fn metadata(&self) -> HnswIndexMetadata;

// Gets current statistics about the index
pub fn stats(&self) -> HnswIndexStats;

// Gets all node IDs in the index.
pub fn node_ids(&self) -> BTreeSet<u64>;

// Gets a node by ID and applies a function to it.
pub fn get_node_with<R, F>(&self, id: u64, f: F) -> Result<Option<R>, HnswError>;

// Sets the node if it is not already present or if the version is newer.
// This method is only used to bootstrap the index from persistent storage.
pub fn set_node(&self, node: HnswNode) -> bool;

// Inserts a vector into the index
pub fn insert(&self, id: u64, vector: Vec<bf16>, now_ms: u64) -> Result<(), HnswError>;

// Inserts a vector into the index with hook function.
pub fn insert_with<R, F>(
    &self,
    id: u64,
    vector: Vec<bf16>,
    now_ms: u64,
    hook: F,
) -> Result<Vec<(u64, R)>, HnswError>
where
    F: Fn(&HnswNode) -> Option<R>;

// Inserts a vector with f32 values into the index
pub fn insert_f32(&self, id: u64, vector: Vec<f32>, now_ms: u64) -> Result<(), HnswError>;

// Inserts a vector into the index with hook function.
pub fn insert_f32_with<R, F>(
    &self,
    id: u64,
    vector: Vec<f32>,
    now_ms: u64,
    hook: F,
) -> Result<Vec<(u64, R)>, HnswError>
where
    F: Fn(&HnswNode) -> Option<R>;

// Removes a vector from the index
pub fn remove(&self, id: u64, now_ms: u64) -> bool;

// Removes a vector from the index with hook function.
pub fn remove_with<R, F>(&self, id: u64, now_ms: u64, hook: F) -> (bool, Vec<(u64, R)>)
where
    F: Fn(&HnswNode) -> Option<R>;

// Searches for the k nearest neighbors to the query vector
pub fn search(&self, query: &[bf16], top_k: usize) -> Result<Vec<(u64, f32)>, HnswError>;

// Searches for nearest neighbors using f32 query vector
pub fn search_f32(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>, HnswError>;

// Stores the index without nodes to a writer.
pub async fn store<W: AsyncWrite + Unpin>(
    &self,
    mut w: W,
    now_ms: u64,
) -> Result<(), HnswError>;

// Stores the index with nodes to a writer.
pub async fn store_all<W: AsyncWrite + Unpin>(
    &self,
    mut w: W,
    now_ms: u64,
) -> Result<(), HnswError>;
```

### Custom Configuration

```rust
use anda_db_hnsw::{HnswConfig, HnswIndex, DistanceMetric, SelectNeighborsStrategy};

let config = HnswConfig {
    dimension: 512,                                  // Vector dimension
    max_layers: 16,                                  // Maximum number of layers
    max_connections: 32,                             // Maximum connections per node
    ef_construction: 200,                            // Expansion factor during construction
    ef_search: 50,                                   // Candidates to consider during search
    distance_metric: DistanceMetric::Cosine,         // Distance metric
    max_nodes: Some(1_000_000),                   // Optional capacity limit
    scale_factor: Some(1.2),                         // Layer distribution scaling
    select_neighbors_strategy: SelectNeighborsStrategy::Heuristic, // Neighbor selection strategy
};

let index = HnswIndex::new("my_index", Some(config));
```

### Distance Metrics

The library supports multiple distance metrics:

```rust
// Euclidean distance (L2 norm)
let config = HnswConfig {
    distance_metric: DistanceMetric::Euclidean,
    ..Default::default()
};

// Cosine distance (1 - cosine similarity)
let config = HnswConfig {
    distance_metric: DistanceMetric::Cosine,
    ..Default::default()
};

// Inner product (negative dot product)
let config = HnswConfig {
    distance_metric: DistanceMetric::InnerProduct,
    ..Default::default()
};

// Manhattan distance (L1 norm)
let config = HnswConfig {
    distance_metric: DistanceMetric::Manhattan,
    ..Default::default()
};
```

### Removing Vectors

```rust
// Remove a vector by ID
let removed = index.remove(vector_id)?;
```

### Index Statistics

```rust
// Get index statistics
let stats = index.stats();
println!("Total vectors: {}", stats.num_nodes);
println!("Max layer: {}", stats.max_layer);
println!("Avg connections: {:.2}", stats.avg_connections);
println!("Search operations: {}", stats.search_count);
println!("Insert operations: {}", stats.insert_count);
println!("Delete operations: {}", stats.delete_count);
```

## Example

Here's a complete example demonstrating vector insertion, search, and deletion:

```rust
use anda_db_hnsw::{HnswConfig, HnswIndex};
use rand::Rng;
use tokio::time;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

pub fn unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    ts.as_millis() as u64
}

// cargo build --example hnsw_demo --release
// ./target/release/examples/hnsw_demo
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    structured_logger::Builder::new().init();

    const DIM: usize = 384;

    // 创建索引 (384维向量，如BERT嵌入)
    let config = HnswConfig {
        dimension: DIM,
        max_nodes: Some(1_000_000),
        ..Default::default()
    };
    let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

    // 模拟数据流
    let mut rng = rand::rng();

    let mut inert_start = time::Instant::now();
    for i in 0..50_000 {
        let vector: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
        let _ = index.insert_f32(i as u64, vector, unix_ms())?;
        // println!("{} inserted vector {}", i, i);

        // 模拟搜索查询
        if i % 100 == 0 {
            println!("{} inserted 100 vectors in {:?}", i, inert_start.elapsed());
            inert_start = time::Instant::now();

            let query: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
            let query_start = time::Instant::now();
            let results = index.search_f32(&query, 10)?;
            println!(
                "{} Search returned {} results in {:?}",
                i,
                results.len(),
                query_start.elapsed()
            );
        }

        // 模拟删除
        if i % 1000 == 0 && i > 0 {
            let to_remove = rng.random_range(0..i);
            let remove_start = time::Instant::now();
            index.remove(to_remove, unix_ms());
            println!(
                "{} Removed vector {} in {:?}",
                i,
                to_remove,
                remove_start.elapsed()
            );
        }
    }

    // 打印统计信息
    let stats = index.stats();
    println!("Index statistics:");
    println!("- Total vectors: {}", stats.num_nodes);
    println!("- Max layer: {}", stats.max_layer);
    println!("- Avg connections: {:.2}", stats.avg_connections);
    println!("- Search operations: {}", stats.search_count);
    println!("- Insert operations: {}", stats.insert_count);
    println!("- Delete operations: {}", stats.delete_count);

    // 最终保存
    {
        let file = tokio::fs::File::create("hnsw_demo.cbor").await?.compat_write();
        let store_start = time::Instant::now();
        index.store_all(file, unix_ms()).await?;
        println!("Stored index with nodes in {:?}", store_start.elapsed());
    }

    let file = tokio::fs::File::open("hnsw_demo.cbor").await?.compat();
    let load_start = time::Instant::now();
    let index = HnswIndex::load(file).await?;
    println!("Load index in {:?}", load_start.elapsed());
    let query: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
    let query_start = time::Instant::now();
    let results = index.search_f32(&query, 10)?;
    println!(
        "Search returned {} results in {:?}",
        results.len(),
        query_start.elapsed()
    );

    Ok(())
}
```

## Error Handling

The library uses a custom error type `HnswError` for various error conditions:

- `HnswError::Generic`: Database-related errors
- `HnswError::Serialization`: Serialization/deserialization errors
- `HnswError::DimensionMismatch`: When vector dimensions don't match the index
- `HnswError::EmptyIndex`: When trying to search an empty index
- `HnswError::IndexFull`: When index has reached its maximum capacity
- `HnswError::NotFound`: When a vector with the specified ID is not found
- `HnswError::AlreadyExists`: When trying to add a vector with an ID that already exists
- `HnswError::DistanceMetric`: Errors related to distance calculations

## Performance Considerations

- The HNSW algorithm provides logarithmic search complexity with respect to the number of vectors
- The `ef_search` parameter controls the trade-off between search speed and accuracy
- The `ef_construction` parameter affects build time and quality of the graph structure
- The `SelectNeighborsStrategy::Heuristic` option provides better search quality at the cost of longer index construction time
- Using bf16 format for vector storage significantly reduces memory usage with minimal precision loss

## License

Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](../../LICENSE) for the full license text.
