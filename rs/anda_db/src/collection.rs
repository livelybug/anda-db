use croaring::{Portable, Treemap};
use dashmap::DashMap;
use futures::try_join;
use object_store::path::Path;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Instant;

use crate::{
    database::AndaDB, error::DBError, index::*, query::*, schema::*, storage::Storage, unix_ms,
};

pub struct Collection {
    /// Collection name
    name: String,
    /// Collection metadata
    schema: Arc<Schema>,
    /// Storage backend
    storage: Storage,
    doc_segments: RwLock<BTreeMap<DocumentId, Vec<SegmentId>>>,
    inverted_doc_segments: DashMap<SegmentId, DocumentId>,
    btree_indexes: Vec<BTree>,
    bm25_hnsw_indexes: Vec<(BM25, Hnsw)>,
    metadata: RwLock<CollectionMetadata>,
    max_document_id: AtomicU64,
    max_segment_id: AtomicU64,
    search_count: AtomicU64,
    get_count: AtomicU64,
    tokenizer: TokenizerChain,
    doc_ids: RwLock<Treemap>,

    /// Maximum size of a bucket before creating a new one
    /// When a bucket's stored data exceeds this size,
    /// a new bucket should be created for new data
    bucket_overload_size: usize,
    current_bucket: RwLock<(BucketId, usize)>,
    dirty_buckets: RwLock<BTreeMap<BucketId, DocSegmentsBucket>>,
    read_only: AtomicBool,
    /// Last saved version of the collection
    last_saved_version: AtomicU64,
}

type BucketId = u32;
type DocSegmentsBucket = HashMap<DocumentId, Vec<SegmentId>>;

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
    pub max_document_id: u64,

    pub max_segment_id: u64,

    pub max_bucket_id: u32,

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

    pub read_only: bool,
}

impl Collection {
    const METADATA_PATH: &'static str = "meta.cbor";
    const IDS_PATH: &'static str = "ids.cbor";

    fn doc_path(id: DocumentId) -> String {
        format!("data/{}.cbor", id)
    }

    fn doc_segments_path(bucket: BucketId) -> String {
        format!("segments/{bucket}.cbor")
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
        let mut stats = CollectionStats::default();
        stats.version = 1;
        let metadata = CollectionMetadata {
            config: config.clone(),
            schema: schema.clone(),
            btree_indexes: BTreeMap::new(),
            bm25_hnsw_indexes: BTreeMap::new(),
            stats,
        };

        match storage.create(Self::METADATA_PATH, &metadata).await {
            Ok(_) => {
                // created successfully, and store storage metadata
                storage.store_metadata(unix_ms()).await?;
            }
            Err(err) => return Err(err),
        }

        let bucket_overload_size = storage.object_chunk_size();
        Ok(Self {
            name: config.name.clone(),
            schema: Arc::new(schema),
            storage,
            doc_segments: RwLock::new(BTreeMap::new()),
            inverted_doc_segments: DashMap::new(),
            btree_indexes: Vec::new(),
            bm25_hnsw_indexes: Vec::new(),
            max_document_id: AtomicU64::new(0),
            max_segment_id: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            tokenizer: default_tokenizer(),
            doc_ids: RwLock::new(Treemap::new()),
            bucket_overload_size,
            current_bucket: RwLock::new((0, 0)),
            dirty_buckets: RwLock::new(BTreeMap::new()),
            metadata: RwLock::new(metadata),
            read_only: AtomicBool::new(false),
            last_saved_version: AtomicU64::new(0),
        })
    }

    pub(crate) async fn open<F>(db: &AndaDB, name: String, f: F) -> Result<Self, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
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

        let bucket_overload_size = storage.object_chunk_size();
        let mut collection = Self {
            name,
            schema: Arc::new(metadata.schema.clone()),
            storage,
            doc_segments: RwLock::new(BTreeMap::new()),
            inverted_doc_segments: DashMap::new(),
            btree_indexes: Vec::new(),
            bm25_hnsw_indexes: Vec::new(),
            max_document_id: AtomicU64::new(metadata.stats.max_document_id),
            max_segment_id: AtomicU64::new(metadata.stats.max_segment_id),
            search_count: AtomicU64::new(metadata.stats.search_count),
            get_count: AtomicU64::new(metadata.stats.get_count),
            last_saved_version: AtomicU64::new(metadata.stats.version),
            tokenizer: default_tokenizer(),
            doc_ids: RwLock::new(Treemap::new()),
            bucket_overload_size,
            current_bucket: RwLock::new((metadata.stats.max_bucket_id, 0)),
            dirty_buckets: RwLock::new(BTreeMap::new()),
            metadata: RwLock::new(metadata),
            read_only: AtomicBool::new(false),
        };

        f(&mut collection).await?;
        collection.load_ids().await?;
        collection.load_doc_segments().await?;
        collection.load_indexes().await?;
        collection.auto_repair_indexes().await;
        Ok(collection)
    }

    async fn load_ids(&mut self) -> Result<(), DBError> {
        let (ids, _) = self.storage.fetch::<Vec<u8>>(Self::IDS_PATH).await?;

        let treemap =
            Treemap::try_deserialize::<Portable>(&ids).ok_or_else(|| DBError::Generic {
                name: self.name.clone(),
                source: "Failed to deserialize ids".into(),
            })?;
        *self.doc_ids.write() = treemap;
        Ok(())
    }

    async fn load_doc_segments(&mut self) -> Result<(), DBError> {
        let mut doc_segments: BTreeMap<DocumentId, Vec<SegmentId>> = BTreeMap::new();
        let max_bucket_id = self.metadata.read().stats.max_bucket_id;

        for i in 0..=max_bucket_id {
            let path = Self::doc_segments_path(i);
            let (bucket, _) = self.storage.fetch::<DocSegmentsBucket>(&path).await?;
            let doc_ids = self.doc_ids.read();
            for (id, segments) in bucket {
                if doc_ids.contains(id) {
                    for sid in &segments {
                        self.inverted_doc_segments.insert(*sid, id);
                    }
                    doc_segments.insert(id, segments);
                }
            }
        }

        self.doc_segments = RwLock::new(doc_segments);
        Ok(())
    }

    async fn load_indexes(&mut self) -> Result<(), DBError> {
        let meta = self.metadata.read().clone();
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

    async fn auto_repair_indexes(&mut self) {
        // TODO
    }

    pub fn set_read_only(&self, read_only: bool) {
        self.read_only.store(read_only, Ordering::Release);
        log::info!(
            action = "set_read_only",
            collection = self.name;
            "Collection is set to read-only: {read_only}",
        );
    }

    pub async fn close(&self) -> Result<(), DBError> {
        let start = Instant::now();
        match self.flush(unix_ms()).await {
            Ok(_) => {
                let elapsed = start.elapsed();
                log::info!(
                    action = "close",
                    collection = self.name,
                    elapsed = elapsed.as_millis();
                    "Collection closed successfully in {elapsed:?}",
                );
            }
            Err(err) => {
                let elapsed = start.elapsed();
                log::error!(
                    action = "close",
                    collection = self.name,
                    elapsed = elapsed.as_millis();
                    "Failed to close collection: {err:?}",
                );
                return Err(err);
            }
        }
        Ok(())
    }

    pub async fn flush(&self, now_ms: u64) -> Result<bool, DBError> {
        if !self.store_metadata(now_ms).await? {
            return Ok(false);
        }

        try_join!(
            self.store_ids(),
            self.store_dirty_segments(),
            self.store_indexes(now_ms),
        )?;

        Ok(true)
    }

    async fn store_metadata(&self, now_ms: u64) -> Result<bool, DBError> {
        let mut meta = self.metadata();
        let prev_saved_version = self
            .last_saved_version
            .fetch_max(meta.stats.version, Ordering::Release);
        if prev_saved_version >= meta.stats.version {
            // No need to save if the version is not updated
            return Ok(false);
        }

        meta.stats.last_saved = now_ms.max(meta.stats.last_saved);
        try_join!(
            self.storage.put(Self::METADATA_PATH, &meta, None),
            self.storage.store_metadata(now_ms),
        )?;

        self.update_metadata(|m| {
            m.stats.last_saved = meta.stats.last_saved.max(m.stats.last_saved);
        });

        Ok(true)
    }

    async fn store_ids(&self) -> Result<(), DBError> {
        let data = {
            let mut ids = self.doc_ids.read().clone();
            ids.run_optimize();
            ids.serialize::<Portable>()
        };
        self.storage.put(Self::IDS_PATH, &data, None).await?;
        Ok(())
    }

    async fn store_dirty_segments(&self) -> Result<(), DBError> {
        let mut dirty_buckets = BTreeMap::new();
        {
            // move the dirty nodes into a temporary variable
            // and release the lock
            dirty_buckets.append(&mut self.dirty_buckets.write());
        }

        while let Some((bucket_id, bucket)) = dirty_buckets.pop_first() {
            let path = Self::doc_segments_path(bucket_id);
            match self.storage.put(&path, &bucket, None).await {
                Ok(_) => {}
                Err(err) => {
                    // refund the unprocessed dirty buckets
                    dirty_buckets.insert(bucket_id, bucket);
                    self.dirty_buckets.write().append(&mut dirty_buckets);
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    async fn store_indexes(&self, now_ms: u64) -> Result<(), DBError> {
        // TODO: concurrently store indexes
        for index in &self.btree_indexes {
            index.flush(now_ms).await?;
        }

        for (bm25, hnsw) in &self.bm25_hnsw_indexes {
            bm25.flush(now_ms).await?;
            hnsw.flush(now_ms).await?;
        }

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
        metadata.stats.max_document_id = self.max_document_id.load(Ordering::Relaxed);
        metadata.stats.max_segment_id = self.max_segment_id.load(Ordering::Relaxed);
        metadata.stats.num_documents = self.doc_segments.read().len() as u64;
        metadata.stats.max_bucket_id = self.current_bucket.read().0;
        metadata.stats.search_count = self.search_count.load(Ordering::Relaxed);
        metadata.stats.get_count = self.get_count.load(Ordering::Relaxed);
        metadata.stats.read_only = self.read_only.load(Ordering::Relaxed);
        metadata
    }

    /// Gets current statistics about the collection
    pub fn stats(&self) -> CollectionStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.max_document_id = self.max_document_id.load(Ordering::Relaxed);
        stats.max_segment_id = self.max_segment_id.load(Ordering::Relaxed);
        stats.max_bucket_id = self.current_bucket.read().0;
        stats.num_documents = self.doc_segments.read().len() as u64;
        stats.search_count = self.search_count.load(Ordering::Relaxed);
        stats.get_count = self.get_count.load(Ordering::Relaxed);
        stats.read_only = self.read_only.load(Ordering::Relaxed);

        stats
    }

    pub fn new_document(&self) -> Document {
        Document::new(self.schema.clone())
    }

    pub fn obtain_segment_ids(&self, segments: &mut [Segment]) {
        let count = segments.len();
        if count == 0 {
            return;
        }
        let start = self
            .max_segment_id
            .fetch_add(count as u64, Ordering::Relaxed)
            + 1;
        for (i, seg) in segments.iter_mut().enumerate() {
            seg.id = start + i as u64;
        }
    }

    pub async fn create_btree_index(&mut self, name: &str, field: &str) -> Result<(), DBError> {
        validate_field_name(name)?;

        let now_ms = unix_ms();

        {
            if self.metadata.read().btree_indexes.contains_key(name) {
                return Err(DBError::AlreadyExists {
                    name: name.to_string(),
                    path: self.name.clone(),
                    source: "BTree index already exists".into(),
                });
            }
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

        let mut meta = self.metadata.write();
        meta.stats.version += 1;
        meta.btree_indexes.insert(name.to_string(), field.clone());
        if field.unique() {
            self.btree_indexes.insert(0, index);
        } else {
            self.btree_indexes.push(index);
        }

        Ok(())
    }

    pub async fn create_btree_index_nx(&mut self, name: &str, field: &str) -> Result<(), DBError> {
        match self.create_btree_index(name, field).await {
            Ok(_) => Ok(()),
            Err(DBError::AlreadyExists { .. }) => {
                // Ignore the error if the index already exists
                Ok(())
            }
            Err(err) => Err(err),
        }
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
            if self.metadata.read().bm25_hnsw_indexes.contains_key(name) {
                return Err(DBError::AlreadyExists {
                    name: name.to_string(),
                    path: self.name.clone(),
                    source: "Search (BM25 & HNSW) index already exists".into(),
                });
            }
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

        {
            let mut meta = self.metadata.write();
            meta.stats.version += 1;
            meta.bm25_hnsw_indexes
                .insert(name.to_string(), field.clone());
        }

        self.bm25_hnsw_indexes.push((bm25, hnsw));
        Ok(())
    }

    pub async fn create_search_index_nx(
        &mut self,
        name: &str,
        field: &str,
        config: HnswConfig,
    ) -> Result<(), DBError> {
        match self.create_search_index(name, field, config).await {
            Ok(_) => Ok(()),
            Err(DBError::AlreadyExists { .. }) => {
                // Ignore the error if the index already exists
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    pub async fn add(&self, mut doc: Document) -> Result<DocumentId, DBError> {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "Collection is read-only".into(),
            });
        }

        let id = self.max_document_id.fetch_add(1, Ordering::Acquire) + 1;
        doc.set_id(id);
        self.schema.validate(doc.fields())?;

        let now_ms = unix_ms();
        let mut segment_ids = BTreeSet::new();

        #[allow(clippy::mutable_key_type)]
        let mut btree_inserted: HashMap<&BTree, &FieldValue> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut bm25_inserted: HashMap<&BM25, (&u64, &str)> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut hnsw_inserted: HashMap<&Hnsw, &u64> = HashMap::new();

        let rt: Result<(), DBError> = (|| {
            for index in &self.btree_indexes {
                if let Some(fv) = doc.get_field(index.field_name()) {
                    if fv == &FieldValue::Null {
                        continue;
                    }

                    btree_inserted.insert(index, fv);
                    index.insert(id, fv, now_ms)?;
                }
            }

            for (bm25, hnsw) in &self.bm25_hnsw_indexes {
                if let Some(Fv::Array(segments)) = doc.get_field(bm25.field_name()) {
                    for seg in segments {
                        if let Some(sid) = Segment::id_from(seg) {
                            if let Some(text) = Segment::text_from(seg) {
                                bm25_inserted.insert(bm25, (sid, text));
                                bm25.insert(*sid, text, now_ms)?;
                            }

                            if let Some(vector) = Segment::vec_from(seg) {
                                hnsw_inserted.insert(hnsw, sid);
                                hnsw.insert(*sid, vector.clone(), now_ms)?;
                            }
                            segment_ids.insert(*sid);
                        }
                    }
                }
            }
            Ok(())
        })();

        let rollback_indexes = || {
            for (k, v) in btree_inserted {
                k.remove(id, v, now_ms);
            }
            for (k, v) in bm25_inserted {
                k.remove(*v.0, v.1, now_ms);
            }
            for (k, v) in hnsw_inserted {
                k.remove(*v, now_ms);
            }
        };

        if let Err(err) = rt {
            rollback_indexes();
            return Err(err);
        }

        let path = Self::doc_path(id);
        if let Err(err) = self.storage.create(&path, &doc).await {
            rollback_indexes();

            return Err(err);
        }

        let segment_ids: Vec<u64> = segment_ids.into_iter().collect();
        let max_bucket_id = {
            let mut current_bucket = self.current_bucket.write();
            let size_increase = CountingWriter::count_cbor(&(&id, &segment_ids)) + 2;
            if current_bucket.1 == 0 || current_bucket.1 + size_increase < self.bucket_overload_size
            {
                current_bucket.1 += size_increase;
            } else {
                current_bucket.0 += 1;
                current_bucket.1 = size_increase;
            }
            current_bucket.0
        };

        for sid in &segment_ids {
            self.inverted_doc_segments.insert(*sid, id);
        }
        self.doc_ids.write().add(id);
        self.doc_segments.write().insert(id, segment_ids.clone());

        let mut dirty_buckets = self.dirty_buckets.write();
        dirty_buckets
            .entry(max_bucket_id)
            .or_default()
            .insert(id, segment_ids);

        self.update_metadata(|meta| {
            meta.stats.last_inserted = now_ms;
            meta.stats.version += 1;
            meta.stats.insert_count += 1;
        });

        Ok(id)
    }

    pub async fn remove(&self, id: DocumentId) -> Result<(), DBError> {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "Collection is read-only".into(),
            });
        }

        let now_ms = unix_ms();
        {
            if !self.doc_ids.write().remove_checked(id) {
                return Ok(());
            }

            for bucket in self.dirty_buckets.write().values_mut() {
                bucket.remove(&id);
            }

            if let Some(segments) = self.doc_segments.write().remove(&id) {
                for sid in segments {
                    self.inverted_doc_segments.remove(&sid);
                }
            }
        }

        self.update_metadata(|meta| {
            meta.stats.last_deleted = now_ms;
            meta.stats.version += 1;
            meta.stats.delete_count += 1;
        });

        if let Ok((doc, _)) = self.storage.get::<DocumentOwned>(&Self::doc_path(id)).await {
            let doc = Document::try_from_doc(self.schema(), doc)?;
            for index in &self.btree_indexes {
                if let Some(fv) = doc.get_field(index.field_name()) {
                    index.remove(id, fv, now_ms);
                }
            }

            for (bm25, hnsw) in &self.bm25_hnsw_indexes {
                if let Some(Fv::Array(segments)) = doc.get_field(bm25.field_name()) {
                    for seg in segments {
                        if let Some(sid) = Segment::id_from(seg) {
                            if let Some(text) = Segment::text_from(seg) {
                                bm25.remove(*sid, text, now_ms);
                            }

                            if Segment::vec_from(seg).is_some() {
                                hnsw.remove(*sid, now_ms);
                            }
                        }
                    }
                }
            }
            let _ = self.storage.delete(&Self::doc_path(id)).await;
        }

        Ok(())
    }

    pub async fn search(&self, query: Query) -> Result<Vec<Document>, DBError> {
        let ids = self.search_ids(query).await?;
        let mut docs = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok((doc, _)) = self.storage.get::<DocumentOwned>(&Self::doc_path(id)).await {
                let doc = Document::try_from_doc(self.schema(), doc)?;
                docs.push(doc);
            }
        }
        Ok(docs)
    }

    pub async fn search_as<T>(&self, query: Query) -> Result<Vec<T>, DBError>
    where
        T: DeserializeOwned,
    {
        let docs = self.search(query).await?;
        let mut rt = Vec::with_capacity(docs.len());
        for doc in docs {
            rt.push(doc.try_into()?);
        }
        Ok(rt)
    }

    pub async fn search_ids(&self, query: Query) -> Result<Vec<DocumentId>, DBError> {
        let limit = query.limit.unwrap_or(10).min(1000);
        let top_k = limit * 3;
        let mut candidates = Vec::with_capacity(top_k);
        let mut result = Vec::new();

        self.search_count.fetch_add(1, Ordering::Relaxed);

        if let Some(params) = query.search {
            if let Some((bm25, hnsw)) = self
                .bm25_hnsw_indexes
                .iter()
                .find(|i| i.0.field_name() == params.field)
            {
                let mut results: Vec<Vec<u64>> = Vec::new();
                if let Some(ref text) = params.text {
                    let rt = if params.logical_search {
                        bm25.search_advanced(text, top_k, params.bm25_params)
                    } else {
                        bm25.search(text, top_k, params.bm25_params)
                    };
                    results.push(rt.into_iter().map(|r| r.0).collect());
                }

                if let Some(ref vector) = params.vector {
                    let rt = hnsw.search(vector, top_k);
                    results.push(rt.into_iter().map(|r| r.0).collect());
                }

                let reranker = params.reranker.unwrap_or_default();
                let reranked = reranker.rerank(&results);
                let mut seen = HashSet::new();
                for (sid, _) in reranked {
                    if let Some(entry) = self.inverted_doc_segments.get(&sid) {
                        let id = *entry;
                        if seen.insert(id) {
                            candidates.push(id);
                        }
                    }
                }
            }

            if candidates.is_empty() {
                return Ok(result);
            }
        }

        match query.filter {
            Some(filter) => {
                result = self.filter_by_field(filter, &candidates, top_k)?;
            }
            None => result = candidates,
        };

        result.truncate(limit);
        Ok(result)
    }

    fn filter_by_field(
        &self,
        filter: Filter,
        candidates: &[DocumentId],
        limit: usize,
    ) -> Result<Vec<DocumentId>, DBError> {
        let mut result = Vec::new();
        match filter {
            Filter::Field((field, filter)) => {
                if field == Schema::ID_KEY {
                    let filter: RangeQuery<u64> =
                        RangeQuery::try_convert_from(filter).map_err(|err| DBError::Generic {
                            name: self.name.clone(),
                            source: err,
                        })?;
                    let ids = self.filter_by_id(filter);
                    for id in ids {
                        if candidates.is_empty() || candidates.contains(&id) {
                            result.push(id);
                            if limit > 0 && result.len() >= limit {
                                return Ok(result);
                            }
                        }
                    }
                } else if let Some(index) =
                    self.btree_indexes.iter().find(|i| i.field_name() == field)
                {
                    let _: Vec<()> = index.search_range_with(filter, |_, ids| {
                        for id in ids {
                            if candidates.is_empty() || candidates.contains(id) {
                                result.push(*id);
                                if limit > 0 && result.len() >= limit {
                                    return (false, vec![]);
                                }
                            }
                        }
                        (true, vec![])
                    });
                } else {
                    return Err(DBError::Index {
                        name: self.name.clone(),
                        source: format!("BTree index not found for field {field:?}").into(),
                    });
                }
            }
            Filter::Or(queries) => {
                let mut seen = HashSet::new();
                for query in queries {
                    let ids = self.filter_by_field(*query, candidates, limit)?;
                    for id in ids {
                        if seen.insert(id) {
                            result.push(id);
                        }
                    }
                }
            }
            Filter::And(queries) => {
                let mut iter = queries.into_iter();
                if let Some(query) = iter.next() {
                    let mut intersection: BTreeSet<u64> =
                        self.filter_by_field(*query, &[], 0)?.into_iter().collect();

                    for query in iter {
                        let keys: BTreeSet<u64> =
                            self.filter_by_field(*query, &[], 0)?.into_iter().collect();
                        intersection = intersection
                            .intersection(&keys)
                            .cloned()
                            .collect::<BTreeSet<_>>();
                        if intersection.is_empty() {
                            return Ok(vec![]);
                        }
                    }

                    result.extend(intersection);
                }
            }
            Filter::Not(query) => {
                let exclude: Vec<u64> = self.filter_by_field(*query, &[], 0)?;
                for id in self.doc_segments.read().keys() {
                    if !exclude.contains(id) && (candidates.is_empty() || candidates.contains(id)) {
                        result.push(*id);
                        if result.len() >= limit {
                            return Ok(result);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn filter_by_id(&self, query: RangeQuery<DocumentId>) -> Vec<DocumentId> {
        let mut results: Vec<u64> = Vec::new();

        match query {
            RangeQuery::Eq(key) => {
                if self.doc_segments.read().contains_key(&key) {
                    results.push(key);
                }
            }
            RangeQuery::Gt(start_key) => {
                for (id, _) in self
                    .doc_segments
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                    .filter(|&(id, _)| *id > start_key)
                {
                    results.push(*id);
                }
            }
            RangeQuery::Ge(start_key) => {
                for (id, _) in self
                    .doc_segments
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                {
                    results.push(*id);
                }
            }
            RangeQuery::Lt(end_key) => {
                for (id, _) in self
                    .doc_segments
                    .read()
                    .range(std::ops::RangeTo { end: end_key })
                    .rev()
                {
                    results.push(*id);
                }
            }
            RangeQuery::Le(end_key) => {
                for (id, _) in self
                    .doc_segments
                    .read()
                    .range(std::ops::RangeToInclusive { end: end_key })
                    .rev()
                {
                    results.push(*id);
                }
            }
            RangeQuery::Between(start_key, end_key) => {
                for (id, _) in self.doc_segments.read().range(start_key..=end_key) {
                    results.push(*id);
                }
            }
            RangeQuery::Include(keys) => {
                let doc_segments = self.doc_segments.read();
                for k in keys.into_iter() {
                    if doc_segments.contains_key(&k) {
                        results.push(k);
                    }
                }
            }
            RangeQuery::And(queries) => {
                let mut iter = queries.into_iter();
                if let Some(query) = iter.next() {
                    let mut intersection: BTreeSet<u64> =
                        self.filter_by_id(*query).into_iter().collect();

                    for query in iter {
                        let keys: BTreeSet<u64> = self.filter_by_id(*query).into_iter().collect();
                        intersection = intersection
                            .intersection(&keys)
                            .cloned()
                            .collect::<BTreeSet<_>>();
                        if intersection.is_empty() {
                            return vec![];
                        }
                    }

                    results.extend(intersection);
                }
            }
            RangeQuery::Or(queries) => {
                let mut seen = HashSet::new();
                for query in queries {
                    let keys = self.filter_by_id(*query);
                    for k in keys {
                        if seen.insert(k) {
                            results.push(k);
                        }
                    }
                }
            }
            RangeQuery::Not(query) => {
                // 先收集要排除的 key，再遍历全集差集
                let exclude: Vec<u64> = self.filter_by_id(*query);
                for (id, _) in self.doc_segments.read().iter() {
                    if !exclude.contains(id) {
                        results.push(*id);
                    }
                }
            }
        }

        results
    }

    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut CollectionMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

/// Utility for counting the size of serialized CBOR data
pub struct CountingWriter {
    count: usize,
}

impl Default for CountingWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl CountingWriter {
    pub fn new() -> Self {
        CountingWriter { count: 0 }
    }

    pub fn size(&self) -> usize {
        self.count
    }

    // TODO: refactor this function to use a more efficient way to count the size
    pub fn count_cbor(val: &impl Serialize) -> usize {
        let mut writer = CountingWriter::new();
        let _ = ciborium::into_writer(val, &mut writer);
        writer.count
    }
}

impl std::io::Write for CountingWriter {
    /// Implements the write method for the Write trait
    /// This simply counts the bytes without actually writing them
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = buf.len();
        self.count += len;
        Ok(len)
    }

    /// Implements the flush method for the Write trait
    /// This is a no-op since we're not actually writing data
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {}
