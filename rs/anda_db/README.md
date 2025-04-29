# Anda DB

Anda DB is a Rust library designed as a specialized database for AI Agents, focusing on knowledge memory. It supports multimodal data storage, full-text search, and vector similarity search, integrating seamlessly as a local database within AI Agent applications.

## Key Features

-   **Embedded Library:** Functions as a Rust library, not a standalone remote database service, enabling direct integration into AI Agent builds.
-   **Object Store Backend:** Leverages an [Object Store](https://docs.rs/object_store) interface, supporting various backends like AWS S3, Google Cloud Storage, Azure Blob Storage, local filesystem, and even the [ICP blockchain](https://internetcomputer.org/).
-   **Encrypted Storage:** Offers optional encrypted storage, writing all data as ciphertext to the Object Store (currently supported for the ICP backend) to ensure data privacy.
-   **Multimodal Data:** Natively handles storage and retrieval of diverse data types including text, images, audio, video, and arbitrary binary data within a flexible document structure.
-   **Flexible Schema & ORM:** Document-oriented design with a flexible schema supporting various field types like `bfloat16` vectors, binary data, JSON, etc. Includes built-in ORM support via procedural macros.
-   **Advanced Indexing:**
    -   **BTree Index:** Enables precise matching, range queries (including timestamps), and multi-conditional logical queries on `U64`, `I64`, `String`, `Bytes`, `Array<T>`, `Option<T>` fields, powered by [`anda_db_btree`](https://docs.rs/anda_db_btree).
    -   **BM25 Index:** Provides efficient full-text search capabilities with multi-conditional logic and powerful tokenizer, powered by [`anda_db_tfs`](https://docs.rs/anda_db_tfs).
    -   **HNSW Index:** Offers high-performance approximate nearest neighbor (ANN) search for vector similarity, powered by [`anda_db_btree`](https://docs.rs/anda_db_hnsw).
-   **Hybrid Search:** Automatically combines and weights text (BM25) and vector (HNSW) search results using Reciprocal Rank Fusion (RRF) for comprehensive retrieval.
-   **Incremental Updates & Persistence:** Supports efficient incremental index updates and document deletions without requiring costly full index rebuilds. Capably saves and loads the entire database state, ensuring data durability.
-   **Efficient Serialization:** Uses CBOR (Concise Binary Object Representation) and Zstd for compact and efficient data serialization.
-   **Collection Management:** Organizes documents into distinct collections, each with its own schema and indexes.

## Installation

Add Anda DB to your `Cargo.toml`:

```toml
[dependencies]
anda_db = { version = "0.2" } # Replace with the desired version
# Add other necessary dependencies like tokio, object_store implementation, etc.
```

## Basic Usage

Here's a simplified example demonstrating how to connect to a database, define a schema, create a collection, add documents, and perform a query.

Source code: https://github.com/ldclabs/anda-db/blob/main/rs/anda_db/examples/db_demo.rs

```rs
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::{AndaDB, DBConfig},
    error::DBError,
    index::HnswConfig,
    query::{Filter, Query, RangeQuery, Search},
    schema::{Document, Fe, Ft, Fv, Json, Resource, Schema, Segment},
    storage::StorageConfig,
};
use anda_db_tfs::jieba_tokenizer;
use ic_auth_types::Xid;
use object_store::local::LocalFileSystem;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};
use structured_logger::unix_ms;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Knowledge {
    pub id: u64,
    // thread ID, thread is a conversation that multi agents can join.
    pub thread: Xid,
    // seconds since epoch
    pub created_at: u64,
    // knowledge authors
    pub authors: Vec<String>,
    // knowledge metadata
    pub metadata: BTreeMap<String, Json>,
    // knowledge segments for text search and vector search
    pub segments: Vec<Segment>,
    // Data source
    pub source: Option<Resource>,
    // confidence score
    pub score: Option<i64>,
    // verification hash
    pub hash: Option<[u8; 32]>,
}

// cargo run --example db_demo --features=full
#[tokio::main]
async fn main() -> Result<(), DBError> {
    // init structured logger
    structured_logger::init();

    // create an in-memory object store
    // It's a simple in-memory storage for testing purposes.
    // In a real application, you would use a persistent storage backend.
    let object_store = LocalFileSystem::new_with_prefix("./debug")?;

    let db_config = DBConfig {
        name: "anda_db_demo".to_string(),
        description: "Anda DB demo".to_string(),
        storage: StorageConfig {
            compress_level: 0, // no compression
            ..Default::default()
        },
    };

    // connect to the database (create if it doesn't exist)
    let db = AndaDB::connect(Arc::new(object_store), db_config).await?;
    log::info!(
        action = "connect",
        database = db.name();
        "connected to database"
    );

    // knowledge schema
    let mut schema = Schema::builder();
    schema
        .add_field(Fe::new("thread".to_string(), Ft::Bytes)?.with_description(
            "thread id, thread is a conversation that multi agents can join.".to_string(),
        ))?
        .add_field(
            Fe::new("created_at".to_string(), Ft::U64)?
                .with_description("knowledge created at in seconds since epoch".to_string()),
        )?
        .add_field(
            Fe::new("authors".to_string(), Ft::Array(vec![Ft::Text]))?
                .with_description("knowledge authors".to_string()),
        )?
        .add_field(
            Fe::new("metadata".to_string(), Ft::Map(BTreeMap::new()))?
                .with_description("knowledge metadata".to_string()),
        )?
        .with_segments("segments", true)?
        .with_resource("source", false)?
        .add_field(
            Fe::new("score".to_string(), Ft::Option(Box::new(Ft::I64)))?
                .with_description("knowledge confidence score".to_string()),
        )?
        .add_field(
            Fe::new("hash".to_string(), Ft::Option(Box::new(Ft::Bytes)))?
                .with_description("verification hash".to_string()),
        )?;

    let schema = schema.build()?;

    let collection_config = CollectionConfig {
        name: "knowledges".to_string(),
        description: "My knowledges".to_string(),
    };

    let collection = db
        .open_or_create_collection(schema, collection_config, async |collection| {
            // set tokenizer
            collection.set_tokenizer(jieba_tokenizer());

            // create BTree indexes if not exists
            collection
                .create_btree_index_nx("btree_thread", "thread")
                .await?;
            collection
                .create_btree_index_nx("btree_created_at", "created_at")
                .await?;
            collection
                .create_btree_index_nx("btree_authors", "authors")
                .await?;
            collection
                .create_btree_index_nx("btree_score", "score")
                .await?;

            // create BM25 & HNSW indexes if not exists
            collection
                .create_search_index_nx(
                    "search_segments",
                    "segments",
                    HnswConfig {
                        dimension: 10,
                        ..Default::default()
                    },
                )
                .await?;
            Ok::<(), DBError>(())
        })
        .await?;
    log::info!(
        action = "open_or_create_collection",
        collection = collection.name();
        "opened or created collection"
    );

    add_knowledges_and_query(&collection).await?;

    db.close().await?;

    Ok(())
}

async fn add_knowledges_and_query(collection: &Arc<Collection>) -> Result<(), DBError> {
    let mut thread = Xid::new();

    let knowledges = vec![
        Knowledge {
            id: 0,
            thread: thread.clone(),
            created_at: unix_ms() / 1000,
            authors: vec!["Anda".to_string(), "Bill".to_string()],
            metadata: BTreeMap::new(),
            segments: vec![
                Segment::new(
                    "Rust 是一门系统级编程语言，专注于安全性、并发性和性能。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                Segment::new(
                    "Rust 的所有权系统是其最独特的特性之一，它在编译时确保内存安全。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]),
            ],
            source: None,
            score: None,
            hash: None,
        },
        Knowledge {
            id: 0,
            thread: thread.clone(),
            created_at: unix_ms() / 1000,
            authors: vec!["Charlie".to_string()],
            metadata: BTreeMap::new(),
            segments: vec![
                Segment::new(
                    "向量数据库是一种特殊类型的数据库，专门用于存储和检索向量嵌入。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                Segment::new(
                    "与传统数据库相比，向量数据库能够高效地进行相似性搜索。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
            ],
            source: None,
            score: None,
            hash: None,
        },
    ];

    let metadata = collection.metadata();
    println!("-----> Collection metadata: {:?}", metadata);

    println!("-----> Add knowledges");
    if metadata.stats.num_documents == 0 {
        for mut knowledge in knowledges {
            collection.obtain_segment_ids(&mut knowledge.segments);
            let doc = Document::try_from(collection.schema(), &knowledge)?;
            let id = collection.add(doc).await?;
            println!("Knowledge id: {id}");
        }
        collection.flush(unix_ms()).await?;
    }

    println!("-----> Search: id = 1");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            filter: Some(Filter::Field((
                "id".to_string(),
                RangeQuery::Eq(Fv::U64(1)),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    // set thread id to the first knowledge for next search
    thread = result[0].thread.clone();
    for doc in &result {
        println!("Find knowledge: {:?}\n", doc);
    }

    println!("-----> Search: thread = xxx");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            filter: Some(Filter::Field((
                "thread".to_string(),
                RangeQuery::Eq(Fv::Bytes(thread.as_slice().into())),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 2);
    for doc in &result {
        println!("Find knowledge: {:?}\n", doc);
    }

    println!("-----> Search: text = Rust");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                text: Some("rust".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    for doc in &result {
        println!("Find knowledge: {:?}\n", doc);
    }

    println!("-----> Search: vector search");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                vector: Some(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 2);
    for doc in &result {
        println!("Find knowledge: {:?}\n", doc);
    }

    println!("-----> Search: compound query");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                text: Some("数据库".to_string()),
                vector: Some(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                ..Default::default()
            }),
            filter: Some(Filter::Field((
                "id".to_string(),
                RangeQuery::Gt(Fv::U64(1)),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    for doc in &result {
        println!("Find knowledge: {:?}\n", doc);
    }

    Ok(())
}
```

## License

Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](../../LICENSE) for the full license text.
