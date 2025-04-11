use anda_db_hnsw::HnswIndex;
use anda_db_tfs::{BM25Index, TokenizerChain};
use dashmap::DashMap;
use object_store::path::Path;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::{Arc, atomic::AtomicU64};

use crate::{database::AndaDB, error::DbError, schema::*, storage::Storage, unix_ms};

pub struct Collection {
    /// Collection name
    name: String,
    /// Collection metadata
    schema: Schema,
    /// Storage backend
    storage: Storage,
    doc_segments: DashMap<Xid, BTreeSet<u64>>,
    inverted_doc_segments: DashMap<u64, Arc<Xid>>,
    bm25_indexes: Vec<BM25Index<TokenizerChain>>,
    hnsw_indexes: Vec<HnswIndex>,
    max_segment_id: AtomicU64,
    metadata: RwLock<CollectionMetadata>,
}

/// Collection configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionConfig {
    /// Index name
    pub name: String,

    /// Collection description
    pub description: String,
}

/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// Collection configuration.
    pub config: CollectionConfig,

    pub schema: Schema,

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

impl Collection {
    const METADATA_PATH: &'static str = "collection_metadata.cbor";
    const DOC_SEGMENTS_PATH: &'static str = "doc_segments.cbor";

    pub(crate) async fn create(
        db: &AndaDB,
        schema: Schema,
        config: CollectionConfig,
    ) -> Result<Self, DbError> {
        validate_field_name(config.name.as_str())?;

        let base_path = Path::from(db.name()).child(config.name.as_str());
        let db_metadata = db.metadata();
        if db_metadata.collections.contains(&config.name) {
            return Err(DbError::AlreadyExists {
                name: config.name,
                path: base_path.to_string(),
                source: "".into(),
            });
        }

        let storage = Storage::connect(
            base_path.to_string(),
            db.object_store(),
            db_metadata.config.storage.clone(),
        )
        .await?;
        let metadata = CollectionMetadata {
            config: config.clone(),
            schema: schema.clone(),
            bm25_indexes: BTreeSet::new(),
            hnsw_indexes: BTreeSet::new(),
            btree_indexes: BTreeSet::new(),
            stats: CollectionStats::default(),
        };

        match storage.create(Self::METADATA_PATH, &metadata).await {
            Ok(_) => {
                // created successfully, and store storage metadata
                storage.store(unix_ms()).await?;
            }
            Err(err) => return Err(err),
        }

        Ok(Self {
            name: config.name.clone(),
            schema,
            storage,
            doc_segments: DashMap::new(),
            inverted_doc_segments: DashMap::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(0),
            metadata: RwLock::new(metadata),
        })
    }

    pub(crate) async fn open(db: &AndaDB, name: String) -> Result<Self, DbError> {
        validate_field_name(name.as_str())?;
        let base_path = Path::from(db.name()).child(name.as_str());
        let db_metadata = db.metadata();
        let storage = Storage::connect(
            base_path.to_string(),
            db.object_store(),
            db_metadata.config.storage.clone(),
        )
        .await?;

        let (metadata, _) = storage
            .fetch::<CollectionMetadata>(Self::METADATA_PATH)
            .await?;

        let mut collection = Self {
            name,
            schema: metadata.schema.clone(),
            storage,
            doc_segments: DashMap::new(),
            inverted_doc_segments: DashMap::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(metadata.stats.max_segment_id),
            metadata: RwLock::new(metadata),
        };

        collection.init_doc_segments().await?;
        collection.init_indexes().await?;
        Ok(collection)
    }

    async fn init_doc_segments(&mut self) -> Result<(), DbError> {
        let (doc_segments, _) = self
            .storage
            .fetch::<DashMap<Xid, BTreeSet<u64>>>(Self::DOC_SEGMENTS_PATH)
            .await?;
        for doc in doc_segments.iter() {
            let xid = Arc::new(doc.key().clone());
            for id in doc.value() {
                self.inverted_doc_segments.insert(*id, xid.clone());
            }
        }
        self.doc_segments = doc_segments;
        Ok(())
    }

    async fn init_indexes(&mut self) -> Result<(), DbError> {
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
