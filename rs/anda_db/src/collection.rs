use anda_db_hnsw::HnswIndex;
use anda_db_tfs::{BM25Index, TokenizerChain};
use dashmap::{DashMap, mapref::one::Ref};
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::atomic::AtomicU64;

use crate::{
    error::DbError,
    schema::*,
    storage::{Storage, StorageBuilder},
};

pub struct Collection {
    /// Collection name
    name: String,
    /// Collection metadata
    schema: Schema,
    /// Storage backend
    storage: Storage,
    doc_segments: DashMap<Xid, BTreeSet<u64>>,
    inverted_doc_segments: DashMap<u64, Ref<'static, Xid, BTreeSet<u64>>>,
    bm25_indexes: Vec<BM25Index<TokenizerChain>>,
    hnsw_indexes: Vec<HnswIndex>,
    max_segment_id: AtomicU64,
    /// Auto-commit interval in seconds (if enabled)
    auto_commit_interval: Option<u64>,
}

/// Collection configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct CollectionConfig {
    /// Auto-commit interval in seconds (if enabled)
    pub auto_commit_interval: Option<u64>,
}


/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// Index name
    pub name: String,

    /// Collection description
    pub description: String,

    /// Collection configuration.
    pub config: CollectionConfig,

    pub bm25_indexes: BTreeSet<String>,

    pub hnsw_indexes: BTreeSet<String>,

    pub btree_indexes: BTreeSet<String>,

    /// Collection statistics.
    pub stats: CollectionStats,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    pub max_segment_id: u64,

    /// Last insertion timestamp (unix ms).
    pub last_inserted: u64,

    /// Last deletion timestamp (unix ms).
    pub last_deleted: u64,

    /// Last saved timestamp (unix ms).
    pub last_saved: u64,

    /// Updated version for the collection. It will be incremented when the collection is updated.
    pub version: u64,

    /// Number of documents in the index.
    pub num_documents: u64,

    /// Number of search operations performed.
    pub search_count: u64,

    /// Number of get operations performed.
    pub get_count: u64,

    /// Number of insert operations performed.
    pub insert_count: u64,

    /// Number of delete operations performed.
    pub delete_count: u64,
}

/// Builder for creating collections
pub struct CollectionBuilder {
    name: String,
    description: String,
    storage: StorageBuilder,
    schema: SchemaBuilder,
    /// Auto-commit interval in seconds (if enabled)
    auto_commit_interval: Option<u64>,
}

impl CollectionBuilder {
    /// Create a new collection builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            storage: StorageBuilder::default(),
            schema: SchemaBuilder::default(),
            auto_commit_interval: Some(30),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set storage backend
    pub fn with_storage(mut self, storage: StorageBuilder) -> Self {
        self.storage = storage;
        self
    }

    pub fn with_schema(mut self, schema: SchemaBuilder) -> Self {
        self.schema = schema;
        self
    }

    /// Set auto-commit interval
    pub fn with_auto_commit(mut self, interval_secs: Option<u64>) -> Self {
        self.auto_commit_interval = interval_secs;
        self
    }

    /// Build the collection
    pub fn build(self) -> Result<Collection, DbError> {
        validate_field_name(self.name.as_str())?;

        let storage = self
            .storage
            .with_base_path(Path::from(self.name.as_str()))
            .build()?;
        let schema = self.schema.build()?;
        Ok(Collection {
            name: self.name,
            schema,
            storage,
            doc_segments: DashMap::new(),
            inverted_doc_segments: DashMap::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(0),
            auto_commit_interval: self.auto_commit_interval,
        })
    }
}

impl Collection {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
