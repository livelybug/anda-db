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
    bm25_hnsw_indexes: Vec<(BM25, Hnsw)>,
    metadata: RwLock<CollectionMetadata>,
    max_segment_id: AtomicU64,
    search_count: AtomicU64,
    get_count: AtomicU64,
    tokenizer: TokenizerChain,
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

    pub bm25_hnsw_indexes: BTreeMap<String, FieldEntry>,

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
    const METADATA_PATH: &'static str = "meta.cbor";
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
            bm25_hnsw_indexes: BTreeMap::new(),
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
            bm25_hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            metadata: RwLock::new(metadata),
            tokenizer: jieba_tokenizer(),
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
            bm25_hnsw_indexes: Vec::new(),
            max_segment_id: AtomicU64::new(metadata.stats.max_segment_id),
            search_count: AtomicU64::new(metadata.stats.search_count),
            get_count: AtomicU64::new(metadata.stats.get_count),
            metadata: RwLock::new(metadata),
            tokenizer: jieba_tokenizer(),
        };

        collection.load_doc_segments().await?;
        collection.load_indexes().await?;
        Ok(collection)
    }

    async fn load_doc_segments(&mut self) -> Result<(), DBError> {
        // TODO: sharding
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

    async fn load_indexes(&mut self) -> Result<(), DBError> {
        let meta = self.metadata.read();
        let (btree_indexes, bm25_indexes, hnsw_indexes) = try_join!(
            async {
                let mut btree_indexes = Vec::new();
                for (name, field) in meta.btree_indexes.iter() {
                    let index =
                        BTree::bootstrap(name.clone(), field.clone(), self.storage.clone()).await?;
                    if field.unique() {
                        btree_indexes.insert(0, index);
                    } else {
                        btree_indexes.push(index);
                    }
                }
                Ok::<Vec<BTree>, DBError>(btree_indexes)
            },
            async {
                let mut bm25_indexes = Vec::new();
                for (name, field) in meta.bm25_hnsw_indexes.iter() {
                    let index = BM25::bootstrap(
                        name.clone(),
                        field.clone(),
                        self.tokenizer.clone(),
                        self.storage.clone(),
                    )
                    .await?;

                    bm25_indexes.push(index);
                }
                Ok::<Vec<BM25>, DBError>(bm25_indexes)
            },
            async {
                let mut hnsw_indexes = Vec::new();
                for (name, field) in meta.bm25_hnsw_indexes.iter() {
                    let index =
                        Hnsw::bootstrap(name.clone(), field.clone(), self.storage.clone()).await?;

                    hnsw_indexes.push(index);
                }
                Ok::<Vec<Hnsw>, DBError>(hnsw_indexes)
            }
        )?;

        self.btree_indexes = btree_indexes;
        self.bm25_hnsw_indexes = bm25_indexes.into_iter().zip(hnsw_indexes).collect();
        Ok(())
    }

    pub fn set_tokenizer(&mut self, tokenizer: TokenizerChain) {
        self.tokenizer = tokenizer;
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

    pub fn obtain_segment_ids(&self, segments: &mut Vec<Segment>) {
        let count = segments.len();
        if count == 0 {
            return;
        }
        let start = self
            .max_segment_id
            .fetch_add(count as u64, AtomicOrdering::Relaxed)
            + 1;
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.id = start + i as u64;
        }
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

            let index = BTree::new(
                name.to_string(),
                field.clone(),
                self.storage.clone(),
                now_ms,
            )
            .await?;

            meta.btree_indexes.insert(name.to_string(), field.clone());
            if field.unique() {
                self.btree_indexes.insert(0, index);
            } else {
                self.btree_indexes.push(index);
            }
        }

        self.store(now_ms).await?;
        Ok(())
    }

    pub async fn create_search_index(
        &mut self,
        name: &str,
        field: &str,
        config: HnswConfig,
    ) -> Result<(), DBError> {
        validate_field_name(name)?;

        let now_ms = unix_ms();
        {
            let mut meta = self.metadata.write();
            if meta.bm25_hnsw_indexes.contains_key(name) {
                return Err(DBError::AlreadyExists {
                    name: name.to_string(),
                    path: self.name.clone(),
                    source: "Search (BM25 & HNSW) index already exists".into(),
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
            if field.r#type() != &FieldType::Array(vec![Segment::field_type()]) {
                return Err(DBError::Schema {
                    name: self.name.clone(),
                    source: "The type of field for search (BM25 & HNSW) index should be FieldType::Array(Vec<Segment>)".into(),
                });
            }

            let (bm25, hnsw) = try_join!(
                BM25::new(
                    name.to_string(),
                    field.clone(),
                    self.tokenizer.clone(),
                    self.storage.clone(),
                    now_ms,
                ),
                Hnsw::new(
                    name.to_string(),
                    field.clone(),
                    config,
                    self.storage.clone(),
                    now_ms,
                )
            )?;

            meta.bm25_hnsw_indexes
                .insert(name.to_string(), field.clone());
            self.bm25_hnsw_indexes.push((bm25, hnsw));
        }

        self.store(now_ms).await?;
        Ok(())
    }

    pub async fn add(&self, doc: Document) -> Result<Xid, DBError> {
        self.schema.validate(doc.fields())?;
        let id = doc.id().clone();
        if id.is_empty() {
            return Err(DBError::Schema {
                name: self.name.clone(),
                source: "document ID is empty".into(),
            });
        }

        let now_ms = unix_ms();
        let mut ids = BTreeSet::new();
        let mut btree_inserted: HashMap<&BTree, &FieldValue> = HashMap::new();
        let mut bm25_inserted: HashMap<&BM25, (&u64, &str)> = HashMap::new();
        let mut hnsw_inserted: HashMap<&Hnsw, &u64> = HashMap::new();

        let rt: Result<(), DBError> = (|| {
            for index in &self.btree_indexes {
                if let Some(fv) = doc.get_field(index.field_name()) {
                    btree_inserted.insert(index, fv);
                    index.insert(&id, fv, now_ms)?;
                }
            }

            for (bm25, hnsw) in &self.bm25_hnsw_indexes {
                if let Some(Fv::Array(segments)) = doc.get_field(bm25.field_name()) {
                    for seg in segments {
                        if let Some(id) = Segment::fv_exact_id(seg) {
                            if let Some(text) = Segment::fv_exact_text(seg) {
                                bm25_inserted.insert(bm25, (id, text));
                                bm25.insert(*id, text, now_ms)?;
                            }

                            if let Some(vector) = Segment::fv_exact_vec(seg) {
                                hnsw_inserted.insert(hnsw, id);
                                hnsw.insert(*id, vector.clone(), now_ms)?;
                            }
                            ids.insert(*id);
                        }
                    }
                }
            }
            Ok(())
        })();

        if let Err(err) = rt {
            // rollback indexes insertions
            for (k, v) in btree_inserted {
                k.remove(&id, v, now_ms);
            }
            for (k, v) in bm25_inserted {
                k.remove(*v.0, v.1, now_ms);
            }
            for (k, v) in hnsw_inserted {
                k.remove(*v, now_ms);
            }
            return Err(err);
        }

        let path = Self::doc_path(&id);
        if let Err(err) = self.storage.create(&path, &doc).await {
            // rollback indexes insertions
            for (k, v) in btree_inserted {
                k.remove(&id, v, now_ms);
            }
            for (k, v) in bm25_inserted {
                k.remove(*v.0, v.1, now_ms);
            }
            for (k, v) in hnsw_inserted {
                k.remove(*v, now_ms);
            }

            return Err(DBError::Storage {
                name: self.name.clone(),
                source: err.into(),
            });
        }

        // TODO: update doc segments

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

#[cfg(test)]
mod tests {
    

    #[test]
    fn test_rt() {}
}
