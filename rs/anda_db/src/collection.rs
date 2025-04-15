use anda_db_hnsw::HnswIndex;
use anda_db_tfs::{BM25Index, TokenizerChain};
use dashmap::DashMap;
use futures::try_join;
use object_store::path::Path;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering as AtomicOrdering},
};

use crate::{
    database::AndaDB, error::DBError, index::*, query::Query, schema::*, storage::Storage, unix_ms,
};

pub struct Collection {
    /// Collection name
    name: String,
    /// Collection metadata
    schema: Arc<Schema>,
    /// Storage backend
    storage: Storage,
    doc_segments: DashMap<Xid, BTreeSet<u64>>,
    inverted_doc_segments: DashMap<u64, Arc<Xid>>,
    btree_indexes: Vec<BTree>,
    bm25_indexes: Vec<BM25Index<TokenizerChain>>,
    hnsw_indexes: Vec<HnswIndex>,
    metadata: RwLock<CollectionMetadata>,
    max_segment_id: AtomicU64,
    search_count: AtomicU64,
    get_count: AtomicU64,
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

    pub btree_indexes: BTreeMap<String, FieldEntry>,

    pub bm25_indexes: BTreeMap<String, FieldEntry>,

    pub hnsw_indexes: BTreeMap<String, FieldEntry>,

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
    const METADATA_PATH: &'static str = "collection_meta.cbor";
    const DOC_SEGMENTS_PATH: &'static str = "doc_segments.cbor";

    fn doc_path(id: &Xid) -> String {
        format!("data/{}.cbor", id)
    }

    pub(crate) async fn create(
        db: &AndaDB,
        schema: Schema,
        config: CollectionConfig,
    ) -> Result<Self, DBError> {
        validate_field_name(config.name.as_str())?;

        let base_path = Path::from(db.name()).child(config.name.as_str());
        let db_metadata = db.metadata();
        if db_metadata.collections.contains(&config.name) {
            return Err(DBError::AlreadyExists {
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
            btree_indexes: BTreeMap::new(),
            bm25_indexes: BTreeMap::new(),
            hnsw_indexes: BTreeMap::new(),
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
            schema: Arc::new(schema),
            storage,
            doc_segments: DashMap::new(),
            inverted_doc_segments: DashMap::new(),
            btree_indexes: Vec::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            metadata: RwLock::new(metadata),
        })
    }

    pub(crate) async fn open(db: &AndaDB, name: String) -> Result<Self, DBError> {
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
            schema: Arc::new(metadata.schema.clone()),
            storage,
            doc_segments: DashMap::new(),
            inverted_doc_segments: DashMap::new(),
            btree_indexes: Vec::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(metadata.stats.max_segment_id),
            search_count: AtomicU64::new(metadata.stats.search_count),
            get_count: AtomicU64::new(metadata.stats.get_count),
            metadata: RwLock::new(metadata),
        };

        collection.init_doc_segments().await?;
        collection.init_indexes().await?;
        Ok(collection)
    }

    async fn init_doc_segments(&mut self) -> Result<(), DBError> {
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

    async fn init_indexes(&mut self) -> Result<(), DBError> {
        let meta = self.metadata.read();
        for (name, field) in meta.btree_indexes.iter() {
            let btree = BTree::bootstrap(name.clone(), field.clone(), self.storage.clone()).await?;
            if field.unique() {
                self.btree_indexes.insert(0, btree);
            } else {
                self.btree_indexes.push(btree);
            }
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Returns the collection metadata
    /// This includes up-to-date statistics about the collection
    pub fn metadata(&self) -> CollectionMetadata {
        let mut metadata = self.metadata.read().clone();
        metadata.stats.num_documents = self.doc_segments.len() as u64;
        metadata.stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        metadata.stats.get_count = self.get_count.load(AtomicOrdering::Relaxed);
        metadata
    }

    /// Gets current statistics about the collection
    pub fn stats(&self) -> CollectionStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.num_documents = self.doc_segments.len() as u64;
        stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        stats.get_count = self.get_count.load(AtomicOrdering::Relaxed);
        stats
    }

    pub fn new_document(&self) -> Document {
        Document::with_id(self.schema.clone(), Xid::new())
    }

    async fn store(&self, now_ms: u64) -> Result<(), DBError> {
        let metadata = self.metadata();
        try_join!(
            self.storage.put(Self::METADATA_PATH, &metadata, None),
            self.storage.store(now_ms)
        )?;

        Ok(())
    }

    pub async fn create_btree_index(&mut self, name: &str, field: &str) -> Result<(), DBError> {
        validate_field_name(name)?;

        let now_ms = unix_ms();
        {
            let mut meta = self.metadata.write();
            if meta.btree_indexes.contains_key(name) {
                return Err(DBError::AlreadyExists {
                    name: name.to_string(),
                    path: self.name.clone(),
                    source: "BTree index already exists".into(),
                });
            }

            let field = self
                .schema
                .get_field(field)
                .ok_or_else(|| DBError::NotFound {
                    name: field.to_string(),
                    path: self.name.clone(),
                    source: "field not found".into(),
                })?;

            let btree = BTree::new(
                name.to_string(),
                field.clone(),
                self.storage.clone(),
                now_ms,
            )
            .await?;

            meta.btree_indexes.insert(name.to_string(), field.clone());
            if field.unique() {
                self.btree_indexes.insert(0, btree);
            } else {
                self.btree_indexes.push(btree);
            }
        }

        self.store(now_ms).await?;
        Ok(())
    }

    pub async fn add(&self, doc: Document) -> Result<Xid, DBError> {
        let now_ms = unix_ms();
        self.schema.validate(doc.fields())?;
        let id = doc.id().clone();
        if id.is_empty() {
            return Err(DBError::Schema {
                name: self.name.clone(),
                source: "document ID is empty".into(),
            });
        }

        let mut inserted_btree: HashMap<&BTree, &FieldValue> = HashMap::new();
        for btree in &self.btree_indexes {
            if let Some(fv) = doc.get_field(btree.field_name()) {
                inserted_btree.insert(btree, fv);
                if let Err(err) = btree.insert(&id, fv, now_ms) {
                    // rollback insertions
                    for (k, v) in inserted_btree {
                        k.remove(&id, v, now_ms);
                    }
                    return Err(err);
                }
            }
        }

        if let Ok(segments) = doc.get_field_as::<Vec<Segment>>("segments") {
            for segment in segments {
                // TODO
            }
        }

        let path = Self::doc_path(&id);
        if let Err(err) = self.storage.create(&path, &doc).await {
            // rollback insertions
            for (k, v) in inserted_btree {
                k.remove(&id, v, now_ms);
            }

            return Err(DBError::Storage {
                name: self.name.clone(),
                source: err.into(),
            });
        }

        self.update_metadata(|meta| {
            meta.stats.last_inserted = now_ms;
            meta.stats.version += 1;
            meta.stats.insert_count += 1;
        });

        Ok(id)
    }

    pub async fn remove(&self, id: &Xid) -> Result<(), DBError> {
        unimplemented!()
    }

    pub async fn search<'a>(&'a self, query: Query<'a>) -> Result<Vec<Xid>, DBError> {
        let limit = query.limit.unwrap_or(10).min(1000);

        Ok(vec![])
    }

    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut CollectionMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}
