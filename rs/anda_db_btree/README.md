# Anda-DB B-tree Index Library

[![Crates.io](https://img.shields.io/crates/v/anda_db_btree)](https://crates.io/crates/anda_db_btree)
[![Documentation](https://docs.rs/anda_db_btree/badge.svg)](https://docs.rs/anda_db_btree)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/ldclabs/anda-db/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda-db/actions)

A high-performance B-tree based index implementation for Anda-DB, optimized for concurrent access and efficient range queries.

## Features

- **Support for various data types**: Index fields of u64, i64, String, binary data and more
- **Efficient range queries**: Optimized for fast range-based lookups
- **Prefix search**: Specialized support for string prefix searches
- **Efficient serialization**: Fast CBOR-based serialization and deserialization
- **Incremental Persistent**: Support incremental index updates persistent (insertions and deletions)
- **Thread-safe concurrent access**: Safely use the index from multiple threads

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
anda_db_btree = "0.1.0"
```

### Basic Example

```rust
use anda_db_btree::{BtreeIndex, BtreeConfig, RangeQuery};

// Create a new B-tree index
let config = BtreeConfig {
    bucket_overload_size: 1024 * 512, // 512KB per bucket
    allow_duplicates: true,
};
let index = BtreeIndex::<String, u64>::new("my_index".to_string(), Some(config));

// Insert some data
let now_ms = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_millis() as u64;

index.insert(1, "apple".to_string(), now_ms).unwrap();
index.insert(2, "banana".to_string(), now_ms).unwrap();
index.insert(3, "cherry".to_string(), now_ms).unwrap();

// Batch insert
let items = vec![
    (4, "date".to_string()),
    (5, "elderberry".to_string()),
];
index.batch_insert(items, now_ms).unwrap();

// Search for exact matches
let result = index.search_with("apple".to_string(), |ids| Some(ids.clone()));
assert!(result.is_some());
println!("Documents with 'apple': {:?}", result.unwrap());

// Range queries
let query = RangeQuery::Between("banana".to_string(), "date".to_string());
let results = index.search_range_with(query, |k, ids| {
    println!("Key: {}, IDs: {:?}", k, ids);
    (true, Some(k.clone()))
});
println!("Keys in range: {:?}", results);

// Prefix search (for String keys)
let results = index.search_prefix_with("app", |k, ids| {
    (true, Some((k.to_string(), ids.clone())))
});
println!("Keys with prefix 'app': {:?}", results);

// Remove data
index.remove(1, "apple".to_string(), now_ms);
```

### Persistence

The B-tree index supports serialization and deserialization for persistence:

```rust
use futures::io::Cursor;

// Serialize index metadata
let mut buf = Vec::new();
index.store_metadata(&mut buf, now_ms).await.unwrap();

// Store dirty buckets
index.store_dirty_buckets(async |bucket_id, data| {
    // Store bucket data to disk or other storage
    // For example, write to a file named by bucket_id
    Ok(true) // Return true to continue with next bucket
}).await.unwrap();

// Later, load the index
let loaded_index = BtreeIndex::<String, u64>::load_metadata(Cursor::new(&buf)).await.unwrap();

// Load buckets
loaded_index.load_buckets(async |bucket_id| {
    // Load bucket data from storage
    // Return the raw bytes
    Ok(Vec::new()) // Replace with actual data loading
}).await.unwrap();
```

## Configuration

The `BtreeConfig` struct allows customizing the index behavior:

```rust
let config = BtreeConfig {
    // Maximum size of a bucket before creating a new one (in bytes)
    bucket_overload_size: 1024 * 512, // 512KB

    // Whether to allow duplicate keys
    // If false, attempting to insert a duplicate key will result in an error
    allow_duplicates: true,
};
```

## Performance Considerations

- **Bucket Size**: Adjust `bucket_overload_size` based on your data characteristics.
- **Concurrency**: The index is designed for concurrent access, using lock-free data structures where possible.
- **Memory Usage**: The index keeps all data in memory for fast access. For very large datasets, consider using multiple smaller indices.

## Error Handling

The library provides a comprehensive error type `BtreeError` that covers various failure scenarios:

- `BtreeError::Generic`: General index-related errors
- `BtreeError::Serialization`: CBOR serialization/deserialization errors
- `BtreeError::NotFound`: When a requested value is not found in the index
- `BtreeError::AlreadyExists`: When trying to insert a duplicate key with `allow_duplicates` set to false

## License
Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](../../LICENSE) for the full license text.