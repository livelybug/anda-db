use anda_db_utils::UniqueVec;
use croaring::{Portable, Treemap};
use futures::{future::try_join_all, try_join as try_join_await};
use object_store::path::Path;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::{borrow::Cow, time::Instant};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
};

use crate::{
    database::AndaDB,
    error::DBError,
    index::*,
    query::*,
    schema::*,
    storage::{ObjectVersion, Storage},
    unix_ms,
};

/// A Collection represents a logical grouping of documents with the same schema.
/// It provides methods for document storage, retrieval, and indexing.
///
/// Collections manage:
/// - Document storage and retrieval
/// - Schema validation
/// - Index creation and maintenance
/// - Search functionality
pub struct Collection {
    /// Collection name
    name: String,
    /// Collection metadata
    schema: Arc<Schema>,
    /// Storage backend for persisting collection data
    storage: Storage,
    /// BTree indexes for efficient exact-match queries
    btree_indexes: Vec<BTree>,
    /// BM25 (text search) indexes
    bm25_indexes: Vec<BM25>,
    /// HNSW (vector search) indexes
    hnsw_indexes: Vec<Hnsw>,
    /// Collection metadata including statistics and configuration
    metadata: RwLock<CollectionMetadata>,
    /// Highest document ID assigned so far
    max_document_id: AtomicU64,
    /// Counter for search operations
    search_count: AtomicU64,
    /// Counter for get operations
    get_count: AtomicU64,
    /// Text tokenization chain for text analysis
    tokenizer: TokenizerChain,
    /// BTree index for document IDs
    doc_ids_index: RwLock<BTreeSet<DocumentId>>,
    /// Bitmap of document IDs for efficient membership tests
    doc_ids: RwLock<Treemap>,
    /// Whether the collection is in read-only mode
    read_only: AtomicBool,
    /// Last saved version of the collection
    last_saved_version: AtomicU64,

    metadata_version: RwLock<ObjectVersion>,
    ids_version: RwLock<ObjectVersion>,
    index_hooks: Arc<dyn IndexHooks>,
}

/// Collection configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionConfig {
    /// Collection name
    pub name: String,

    /// Collection description
    pub description: String,
}

/// Collection metadata containing configuration, schema, indexes, and statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// Collection configuration.
    pub config: CollectionConfig,

    /// Schema defining the structure of documents in this collection
    pub schema: Schema,

    /// Map of BTree index names to their field entries
    pub btree_indexes: BTreeMap<String, FieldEntry>,

    /// Map of BM25 index names to their field entries
    pub bm25_indexes: BTreeMap<String, FieldEntry>,

    /// Map of HNSW index names to their field entries
    pub hnsw_indexes: BTreeMap<String, FieldEntry>,

    /// Collection statistics.
    pub stats: CollectionStats,
}

/// Statistics about the collection's usage and state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionStats {
    /// Highest document ID assigned so far
    pub max_document_id: u64,

    /// Last insertion timestamp (unix ms).
    pub last_inserted: u64,

    /// Last update timestamp (unix ms).
    pub last_updated: u64,

    /// Last deletion timestamp (unix ms).
    pub last_deleted: u64,

    /// Last saved timestamp (unix ms).
    pub last_saved: u64,

    /// Updated version for the collection. It will be incremented when the collection is updated.
    pub version: u64,

    /// Number of documents in the collection.
    pub num_documents: u64,

    /// Number of search operations performed.
    pub search_count: u64,

    /// Number of get operations performed.
    pub get_count: u64,

    /// Number of insert operations performed.
    pub insert_count: u64,

    /// Number of update operations performed.
    pub update_count: u64,

    /// Number of delete operations performed.
    pub delete_count: u64,

    /// Whether the collection is in read-only mode
    pub read_only: bool,
}

impl Debug for Collection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Collection({})", self.name)
    }
}

impl Collection {
    /// Path to the collection metadata file
    const METADATA_PATH: &'static str = "meta.cbor";

    /// Path to the document IDs bitmap file
    const IDS_PATH: &'static str = "ids.cbor";

    /// Generates the storage path for a document with the given ID
    fn doc_path(id: DocumentId) -> String {
        format!("data/{id}.cbor")
    }

    /// Creates a new collection with the given schema and configuration.
    ///
    /// # Arguments
    /// * `db` - Reference to the database this collection belongs to
    /// * `schema` - Schema defining the structure of documents in this collection
    /// * `config` - Configuration parameters for the collection
    ///
    /// # Returns
    /// A new Collection instance or an error if creation fails
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
        let stats = CollectionStats {
            version: 1,
            ..Default::default()
        };
        let metadata = CollectionMetadata {
            config: config.clone(),
            schema: schema.clone(),
            btree_indexes: BTreeMap::new(),
            bm25_indexes: BTreeMap::new(),
            hnsw_indexes: BTreeMap::new(),
            stats,
        };

        let metadata_version = storage.create(Self::METADATA_PATH, &metadata).await?;
        let doc_ids = Treemap::new();
        let ids_data = {
            let mut ids = doc_ids.clone();
            ids.run_optimize();
            ids.serialize::<Portable>()
        };
        let ids_version = storage.create(Self::IDS_PATH, &ids_data).await?;

        // created successfully, and store storage metadata
        storage.store_metadata(0, unix_ms()).await?;

        Ok(Self {
            name: config.name.clone(),
            schema: Arc::new(schema),
            storage,
            btree_indexes: Vec::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_document_id: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            get_count: AtomicU64::new(0),
            tokenizer: default_tokenizer(),
            doc_ids_index: RwLock::new(BTreeSet::new()),
            doc_ids: RwLock::new(Treemap::new()),
            metadata: RwLock::new(metadata),
            read_only: AtomicBool::new(false),
            last_saved_version: AtomicU64::new(0),
            metadata_version: RwLock::new(metadata_version),
            ids_version: RwLock::new(ids_version),
            index_hooks: Arc::new(DefaultIndexHooks),
        })
    }

    /// Opens an existing collection.
    ///
    /// # Arguments
    /// * `db` - Reference to the database this collection belongs to
    /// * `name` - Name of the collection to open
    /// * `f` - Function to execute on the collection before it's fully loaded
    ///
    /// # Returns
    /// The opened Collection instance or an error if opening fails
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

        let (metadata, metadata_version) = storage
            .fetch::<CollectionMetadata>(Self::METADATA_PATH)
            .await?;

        let (ids, ids_version) = storage.fetch::<Vec<u8>>(Self::IDS_PATH).await?;
        let doc_ids =
            Treemap::try_deserialize::<Portable>(&ids).ok_or_else(|| DBError::Generic {
                name: name.clone(),
                source: "Failed to deserialize ids".into(),
            })?;
        let doc_ids_index = BTreeSet::from_iter(doc_ids.iter());

        let mut collection = Self {
            name,
            schema: Arc::new(metadata.schema.clone()),
            storage,
            btree_indexes: Vec::new(),
            bm25_indexes: Vec::new(),
            hnsw_indexes: Vec::new(),
            max_document_id: AtomicU64::new(metadata.stats.max_document_id),
            search_count: AtomicU64::new(metadata.stats.search_count),
            get_count: AtomicU64::new(metadata.stats.get_count),
            last_saved_version: AtomicU64::new(metadata.stats.version),
            tokenizer: default_tokenizer(),
            doc_ids_index: RwLock::new(doc_ids_index),
            doc_ids: RwLock::new(doc_ids),
            metadata: RwLock::new(metadata),
            read_only: AtomicBool::new(false),
            metadata_version: RwLock::new(metadata_version),
            ids_version: RwLock::new(ids_version),
            index_hooks: Arc::new(DefaultIndexHooks),
        };

        f(&mut collection).await?;
        collection.load_indexes().await?;
        let fixed = collection.auto_repair_indexes().await?;
        if fixed > 0 {
            log::info!(
                action = "auto_repair_indexes",
                collection = collection.name;
                "Auto-repaired {fixed} documents",
            );

            // flush the fixed documents
            collection.flush(unix_ms()).await?;
        }
        Ok(collection)
    }

    /// Loads all indexes from storage.
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if loading fails
    async fn load_indexes(&mut self) -> Result<(), DBError> {
        let meta = self.metadata.read().clone();
        let (btree_indexes, bm25_indexes, hnsw_indexes) = try_join_await!(
            async {
                let mut btree_indexes = Vec::new();
                for (name, field) in meta.btree_indexes.iter() {
                    let index =
                        BTree::bootstrap(name.clone(), field.r#type(), self.storage.clone())
                            .await?;
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
                for (name, _) in meta.bm25_indexes.iter() {
                    let index =
                        BM25::bootstrap(name.clone(), self.tokenizer.clone(), self.storage.clone())
                            .await?;

                    bm25_indexes.push(index);
                }
                Ok::<Vec<BM25>, DBError>(bm25_indexes)
            },
            async {
                let mut hnsw_indexes = Vec::new();
                for (name, _) in meta.hnsw_indexes.iter() {
                    let index = Hnsw::bootstrap(name.clone(), self.storage.clone()).await?;

                    hnsw_indexes.push(index);
                }
                Ok::<Vec<Hnsw>, DBError>(hnsw_indexes)
            },
        )?;

        self.btree_indexes = btree_indexes;
        self.bm25_indexes = bm25_indexes;
        self.hnsw_indexes = hnsw_indexes;
        Ok(())
    }

    /// Automatically repairs indexes if needed.
    /// This is called during collection opening to ensure index integrity.
    async fn auto_repair_indexes(&self) -> Result<usize, DBError> {
        let persisted_max_document_id = self.storage.stats().check_point;
        let maybe_max_document_id = self.max_document_id.load(Ordering::Relaxed);
        let now_ms = unix_ms();
        let mut id = persisted_max_document_id;
        let mut fixed = 0;

        loop {
            id += 1;
            match self
                .storage
                .fetch::<DocumentOwned>(&Self::doc_path(id))
                .await
            {
                Err(_) => {
                    if id > maybe_max_document_id {
                        return Ok(fixed);
                    }
                }
                Ok((doc, _)) => {
                    // dirty document exists
                    fixed += 1;
                    let doc = Document::try_from_doc(self.schema(), doc)?;
                    // try to repair indexes
                    for index in &self.btree_indexes {
                        if let Some(fv) = self.index_hooks.btree_index_value(index, &doc) {
                            if fv.as_ref() == &FieldValue::Null {
                                continue;
                            }
                            // ignore errors
                            let _ = index.insert(id, fv.into_owned(), now_ms);
                        }
                    }

                    for index in &self.bm25_indexes {
                        if let Some(text) = self.index_hooks.bm25_index_value(index, &doc) {
                            // ignore errors
                            let _ = index.insert(id, &text, now_ms);
                        }
                    }

                    for index in &self.hnsw_indexes {
                        if let Some(vector) = self.index_hooks.hnsw_index_value(index, &doc) {
                            // ignore errors
                            let _ = index.insert(id, vector.into_owned(), now_ms);
                        }
                    }

                    self.update_metadata(|meta| {
                        meta.stats.version += 1;
                    });
                }
            }
        }
    }

    /// Sets the collection to read-only mode.
    ///
    /// # Arguments
    /// * `read_only` - Whether to enable read-only mode
    pub fn set_read_only(&self, read_only: bool) {
        self.read_only.store(read_only, Ordering::Release);
        log::info!(
            action = "set_read_only",
            collection = self.name;
            "Collection is set to read-only: {read_only}",
        );
    }

    /// Closes the collection, ensuring all data is flushed to storage.
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if closing fails
    pub async fn close(&self) -> Result<(), DBError> {
        self.set_read_only(true);

        let start = Instant::now();
        let now_ms = unix_ms();
        let rt = self.flush(now_ms).await;
        let elapsed = start.elapsed();
        match rt {
            Ok(_) => {
                log::info!(
                    action = "close",
                    collection = self.name,
                    elapsed = elapsed.as_millis();
                    "Collection closed successfully in {elapsed:?}",
                );
                Ok(())
            }
            Err(err) => {
                log::error!(
                    action = "close",
                    collection = self.name,
                    elapsed = elapsed.as_millis();
                    "Failed to close collection: {err:?}",
                );
                Err(err)
            }
        }
    }

    /// Flushes all pending changes to storage.
    ///
    /// # Arguments
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    /// `true` if changes were flushed, `false` if no changes needed to be flushed
    pub async fn flush(&self, now_ms: u64) -> Result<bool, DBError> {
        let check_point = match self.store_metadata(now_ms).await? {
            Some(id) => id,
            None => return Ok(false),
        };

        self.store_ids().await?;
        self.store_indexes(now_ms).await?;

        // check_point is the last persisted document ID
        self.storage.store_metadata(check_point, now_ms).await?;
        Ok(true)
    }

    /// Stores collection metadata to storage if it has changed.
    ///
    /// # Arguments
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    /// `true` if metadata was stored, `false` if no changes needed to be stored
    async fn store_metadata(&self, now_ms: u64) -> Result<Option<DocumentId>, DBError> {
        let mut meta = self.metadata();
        let prev_saved_version = self
            .last_saved_version
            .fetch_max(meta.stats.version, Ordering::Release);
        if prev_saved_version >= meta.stats.version {
            // No need to save if the version is not updated
            return Ok(None);
        }

        meta.stats.last_saved = now_ms.max(meta.stats.last_saved);
        let ver = { self.metadata_version.read().clone() };
        let ver = self
            .storage
            .put(Self::METADATA_PATH, &meta, Some(ver))
            .await?;
        *self.metadata_version.write() = ver;
        self.update_metadata(|m| {
            m.stats.last_saved = meta.stats.last_saved.max(m.stats.last_saved);
        });

        Ok(Some(meta.stats.max_document_id))
    }

    /// Stores document IDs bitmap to storage.
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if storing fails
    async fn store_ids(&self) -> Result<(), DBError> {
        let data = {
            let mut ids = self.doc_ids.read().clone();
            ids.run_optimize();
            ids.serialize::<Portable>()
        };
        let ver = { self.ids_version.read().clone() };
        let ver = self.storage.put(Self::IDS_PATH, &data, Some(ver)).await?;
        *self.ids_version.write() = ver;
        Ok(())
    }

    /// Stores all indexes to storage.
    ///
    /// # Arguments
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if storing fails
    async fn store_indexes(&self, now_ms: u64) -> Result<(), DBError> {
        let _ = try_join_await!(
            try_join_all(self.btree_indexes.iter().map(|index| index.flush(now_ms))),
            try_join_all(self.bm25_indexes.iter().map(|index| index.flush(now_ms))),
            try_join_all(self.hnsw_indexes.iter().map(|index| index.flush(now_ms))),
        );

        Ok(())
    }

    /// Sets the tokenizer for text analysis.
    ///
    /// # Arguments
    /// * `tokenizer` - The tokenizer chain to use
    pub fn set_tokenizer(&mut self, tokenizer: TokenizerChain) {
        self.tokenizer = tokenizer;
    }

    pub fn set_index_hooks(&mut self, hooks: Arc<dyn IndexHooks>) {
        self.index_hooks = hooks;
    }

    /// Returns the collection name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the collection schema.
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Returns the collection metadata.
    /// This includes up-to-date statistics about the collection.
    pub fn metadata(&self) -> CollectionMetadata {
        let mut metadata = self.metadata.read().clone();
        metadata.stats.max_document_id = self.max_document_id.load(Ordering::Relaxed);
        metadata.stats.num_documents = self.doc_ids_index.read().len() as u64;
        metadata.stats.search_count = self.search_count.load(Ordering::Relaxed);
        metadata.stats.get_count = self.get_count.load(Ordering::Relaxed);
        metadata.stats.read_only = self.read_only.load(Ordering::Relaxed);
        metadata
    }

    /// Gets current statistics about the collection
    pub fn stats(&self) -> CollectionStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.max_document_id = self.max_document_id.load(Ordering::Relaxed);
        stats.num_documents = self.doc_ids_index.read().len() as u64;
        stats.search_count = self.search_count.load(Ordering::Relaxed);
        stats.get_count = self.get_count.load(Ordering::Relaxed);
        stats.read_only = self.read_only.load(Ordering::Relaxed);

        stats
    }

    /// Checks if a document with the given ID exists in the collection.
    ///
    /// # Arguments
    /// * `id` - The ID to check
    ///
    /// # Returns
    /// `true` if a document with the ID exists, `false` otherwise
    pub fn contains(&self, id: DocumentId) -> bool {
        self.doc_ids_index.read().contains(&id)
    }

    /// Gets the number of documents in the collection.
    ///
    /// # Returns
    /// The number of documents in the collection
    pub fn len(&self) -> usize {
        self.doc_ids_index.read().len()
    }

    /// Checks if the collection is empty.
    ///
    /// # Returns
    /// `true` if the collection contains no documents, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.doc_ids_index.read().is_empty()
    }

    /// Creates a new empty document with the collection's schema.
    pub fn new_document(&self) -> Document {
        Document::new(self.schema.clone())
    }

    /// Tokenizes the given text using the collection's tokenizer.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        BM25::collect_tokens(&self.tokenizer, text)
    }

    /// Creates a BTree index on the specified field.
    ///
    /// # Arguments
    /// * `fields` - Fields to index
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_btree_index(&mut self, fields: &[&str]) -> Result<(), DBError> {
        if fields.is_empty() {
            return Err(DBError::Schema {
                name: self.name.clone(),
                source: "BTree index requires at least one field".into(),
            });
        }

        let now_ms = unix_ms();
        let name = virtual_field_name(fields);

        {
            if self.metadata.read().btree_indexes.contains_key(&name) {
                return Err(DBError::AlreadyExists {
                    name: name.to_string(),
                    path: self.name.clone(),
                    source: "BTree index already exists".into(),
                });
            }
        }

        if fields.len() == 1 {
            let field = self.schema.get_field_or_err(fields[0])?;

            let index = BTree::new(field.clone(), self.storage.clone(), now_ms).await?;
            if field.unique() {
                self.btree_indexes.insert(0, index);
            } else {
                self.btree_indexes.push(index);
            }
            let mut meta = self.metadata.write();
            meta.btree_indexes.insert(name.to_string(), field.clone());
            meta.stats.version += 1;
        } else {
            for field in fields {
                self.schema.get_field_or_err(field)?;
            }
            let index = BTree::with_virtual_field(
                fields.iter().map(|s| s.to_string()).collect(),
                self.storage.clone(),
                now_ms,
            )
            .await?;

            let field = FieldEntry::new("_virtual_field_".to_string(), FieldType::Bytes)?
                .with_unique()
                .with_description(name.clone());
            self.btree_indexes.insert(0, index);
            let mut meta = self.metadata.write();
            meta.btree_indexes.insert(name, field);
            meta.stats.version += 1;
        }

        Ok(())
    }

    /// Creates a BTree index if it doesn't already exist.
    ///
    /// # Arguments
    /// * `fields` - Fields to index
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_btree_index_nx(&mut self, fields: &[&str]) -> Result<(), DBError> {
        match self.create_btree_index(fields).await {
            Ok(_) => Ok(()),
            Err(DBError::AlreadyExists { .. }) => {
                // Ignore the error if the index already exists
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Creates a BM25 text search index.
    ///
    /// # Arguments
    /// * `field` - Name of the field to index
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_bm25_index(&mut self, fields: &[&str]) -> Result<(), DBError> {
        if fields.is_empty() {
            return Err(DBError::Schema {
                name: self.name.clone(),
                source: "BM25 index requires at least one field".into(),
            });
        }

        let now_ms = unix_ms();
        let name = virtual_field_name(fields);

        {
            if self.metadata.read().bm25_indexes.contains_key(&name) {
                return Err(DBError::AlreadyExists {
                    name: name.clone(),
                    path: self.name.clone(),
                    source: "BM25 index already exists".into(),
                });
            }
        }

        for field in fields {
            self.schema.get_field_or_err(field)?;
        }

        let index = BM25::new(
            fields.iter().map(|s| s.to_string()).collect(),
            self.tokenizer.clone(),
            self.storage.clone(),
            now_ms,
        )
        .await?;

        {
            let mut meta = self.metadata.write();
            meta.stats.version += 1;
            let field = FieldEntry::new("_virtual_field_".to_string(), FieldType::Text)?
                .with_description(name.clone());
            meta.bm25_indexes.insert(name, field);
        }

        self.bm25_indexes.push(index);
        Ok(())
    }

    /// Creates a BM25 index if it doesn't already exist.
    ///
    /// # Arguments
    /// * `fields` - Fields to index
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_bm25_index_nx(&mut self, fields: &[&str]) -> Result<(), DBError> {
        match self.create_bm25_index(fields).await {
            Ok(_) => Ok(()),
            Err(DBError::AlreadyExists { .. }) => {
                // Ignore the error if the index already exists
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Creates a HNSW (vector) search index.
    ///
    /// # Arguments
    /// * `field` - Name of the field to index
    /// * `config` - HNSW index configuration
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_hnsw_index(
        &mut self,
        field: &str,
        config: HnswConfig,
    ) -> Result<(), DBError> {
        validate_field_name(field)?;

        let name = field.to_string();
        let now_ms = unix_ms();

        {
            if self.metadata.read().hnsw_indexes.contains_key(&name) {
                return Err(DBError::AlreadyExists {
                    name: name.clone(),
                    path: self.name.clone(),
                    source: "HNSW index already exists".into(),
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
        if field.r#type() != &FieldType::Vector {
            return Err(DBError::Schema {
                name: self.name.clone(),
                source: "The type of field for HNSW index should be FieldType::Vector".into(),
            });
        }

        let index = Hnsw::new(field, config, self.storage.clone(), now_ms).await?;

        {
            let mut meta = self.metadata.write();
            meta.stats.version += 1;
            meta.hnsw_indexes.insert(name, field.clone());
        }

        self.hnsw_indexes.push(index);
        Ok(())
    }

    /// Creates a HNSW index if it doesn't already exist.
    ///
    /// # Arguments
    /// * `field` - Name of the field to index
    /// * `config` - HNSW index configuration
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if creation fails
    pub async fn create_hnsw_index_nx(
        &mut self,
        field: &str,
        config: HnswConfig,
    ) -> Result<(), DBError> {
        match self.create_hnsw_index(field, config).await {
            Ok(_) => Ok(()),
            Err(DBError::AlreadyExists { .. }) => {
                // Ignore the error if the index already exists
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    pub fn get_btree_index(&self, fields: &[&str]) -> Result<&BTree, DBError> {
        let name = virtual_field_name(fields);
        if let Some(index) = self.btree_indexes.iter().find(|i| i.name() == name) {
            return Ok(index);
        }

        Err(DBError::Index {
            name,
            source: "BTree index not found".into(),
        })
    }

    pub fn get_bm25_index(&self, fields: &[&str]) -> Result<&BM25, DBError> {
        let name = virtual_field_name(fields);
        if let Some(index) = self.bm25_indexes.iter().find(|i| i.name() == name) {
            return Ok(index);
        }

        Err(DBError::Index {
            name,
            source: "BM25 index not found".into(),
        })
    }

    pub fn get_hnsw_index(&self, field: &str) -> Result<&Hnsw, DBError> {
        if let Some(index) = self.hnsw_indexes.iter().find(|i| i.field_name() == field) {
            return Ok(index);
        }

        Err(DBError::Index {
            name: field.to_string(),
            source: "HNSW index not found".into(),
        })
    }

    /// Adds a new document to the collection.
    ///
    /// This method:
    /// 1. Validates the document against the collection schema
    /// 2. Assigns a unique ID to the document
    /// 3. Updates all relevant indexes
    /// 4. Persists the document to storage
    ///
    /// # Arguments
    /// * `doc` - The document to add to the collection
    ///
    /// # Returns
    /// The ID of the newly added document, or an error if addition fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - The collection is in read-only mode
    /// - The document fails schema validation
    /// - Any index update fails
    /// - Storage operations fail
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
        #[allow(clippy::mutable_key_type)]
        let mut btree_inserted: HashMap<&BTree, Cow<FieldValue>> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut bm25_inserted: HashMap<&BM25, (u64, Cow<str>)> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut hnsw_inserted: HashMap<&Hnsw, u64> = HashMap::new();

        let rt: Result<(), DBError> = (|| {
            for index in &self.btree_indexes {
                if let Some(fv) = self.index_hooks.btree_index_value(index, &doc) {
                    if fv.as_ref() == &FieldValue::Null {
                        continue;
                    }

                    btree_inserted.insert(index, fv.clone());
                    index.insert(id, fv.into_owned(), now_ms)?;
                }
            }

            for index in &self.bm25_indexes {
                if let Some(text) = self.index_hooks.bm25_index_value(index, &doc) {
                    index.insert(id, &text, now_ms)?;
                    bm25_inserted.insert(index, (id, text));
                }
            }

            for index in &self.hnsw_indexes {
                if let Some(vector) = self.index_hooks.hnsw_index_value(index, &doc) {
                    hnsw_inserted.insert(index, id);
                    index.insert(id, vector.into_owned(), now_ms)?;
                }
            }

            Ok(())
        })();

        let rollback_indexes = || {
            for (k, v) in btree_inserted {
                k.remove(id, &v, now_ms);
            }
            for (k, v) in bm25_inserted {
                k.remove(v.0, &v.1, now_ms);
            }
            for (k, v) in hnsw_inserted {
                k.remove(v, now_ms);
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

        self.doc_ids.write().add(id);
        self.doc_ids_index.write().insert(id);

        self.update_metadata(|meta| {
            meta.stats.last_inserted = now_ms;
            meta.stats.version += 1;
            meta.stats.insert_count += 1;
        });

        Ok(id)
    }

    /// Adds a new document to the collection from a serializable value.
    ///
    /// This method:
    /// 1. Converts the value into a Document using the collection's schema
    /// 2. Validates the document against the schema
    /// 3. Assigns a unique ID to the document
    /// 4. Updates all relevant indexes
    /// 5. Persists the document to storage
    /// # Arguments
    /// * `val` - The value to convert into a document
    ///
    /// # Returns
    /// The ID of the newly added document, or an error if addition fails
    pub async fn add_from<T>(&self, val: &T) -> Result<DocumentId, DBError>
    where
        T: Serialize,
    {
        let doc = Document::try_from(self.schema(), val)?;
        self.add(doc).await
    }

    /// Updates an existing document with new field values.
    ///
    /// # Arguments
    /// * `id` - The ID of the document to update
    /// * `fields` - The new field values to apply
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if update fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - The collection is in read-only mode
    /// - The document doesn't exist
    /// - The updated document fails schema validation
    /// - The updated document version not matching the stored version because of concurrent update
    /// - Any index update fails
    /// - Storage operations fail
    pub async fn update(
        &self,
        id: DocumentId,
        fields: BTreeMap<String, Fv>,
    ) -> Result<(), DBError> {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "Collection is read-only".into(),
            });
        }

        if !self.doc_ids.read().contains(id) {
            return Err(DBError::NotFound {
                name: id.to_string(),
                path: self.name.clone(),
                source: format!("Document with ID {id} not found").into(),
            });
        }

        if fields.is_empty() {
            return Ok(());
        }

        let (doc, ver) = self
            .storage
            .get::<DocumentOwned>(&Self::doc_path(id))
            .await?;
        let mut doc = Document::try_from_doc(self.schema(), doc)?;
        let old_doc = doc.clone();

        // apply the new values
        let mut fields_keys = HashSet::new();
        for (field_name, fv) in fields {
            doc.set_field(&field_name, fv)?;
            fields_keys.insert(field_name);
        }

        // validate the updated document
        self.schema.validate(doc.fields())?;

        let now_ms = unix_ms();

        // record the updated and removed indexes for rollback
        #[allow(clippy::mutable_key_type)]
        let mut btree_inserted: HashMap<&BTree, Cow<FieldValue>> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut bm25_inserted: HashMap<&BM25, (u64, Cow<str>)> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut hnsw_inserted: HashMap<&Hnsw, u64> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut btree_removed: HashMap<&BTree, Cow<FieldValue>> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut bm25_removed: HashMap<&BM25, (u64, Cow<str>)> = HashMap::new();
        #[allow(clippy::mutable_key_type)]
        let mut hnsw_removed: HashMap<&Hnsw, (u64, Cow<Vector>)> = HashMap::new();

        // update the indexes
        let rt: Result<(), DBError> = (|| {
            for index in &self.btree_indexes {
                let fields = index.virtual_field();
                if fields_keys.iter().any(|v| fields.contains(v)) {
                    if let Some(fv) = self.index_hooks.btree_index_value(index, &old_doc) {
                        if fv.as_ref() != &FieldValue::Null {
                            index.remove(id, &fv, now_ms);
                            btree_removed.insert(index, fv);
                        }
                    }

                    if let Some(fv) = self.index_hooks.btree_index_value(index, &doc) {
                        if fv.as_ref() != &FieldValue::Null {
                            index.insert(id, fv.clone().into_owned(), now_ms)?;
                            btree_inserted.insert(index, fv);
                        }
                    }
                }
            }

            for index in &self.bm25_indexes {
                let fields = index.virtual_field();
                if fields_keys.iter().any(|v| fields.contains(v)) {
                    if let Some(text) = self.index_hooks.bm25_index_value(index, &old_doc) {
                        index.remove(id, &text, now_ms);
                        bm25_removed.insert(index, (id, text));
                    }

                    if let Some(text) = self.index_hooks.bm25_index_value(index, &doc) {
                        index.insert(id, &text, now_ms)?;
                        bm25_inserted.insert(index, (id, text));
                    }
                }
            }

            for index in &self.hnsw_indexes {
                let field_name = index.field_name();
                if fields_keys.contains(field_name) {
                    if let Some(vector) = self.index_hooks.hnsw_index_value(index, &old_doc) {
                        index.remove(id, now_ms);
                        hnsw_removed.insert(index, (id, vector));
                    }

                    if let Some(vector) = self.index_hooks.hnsw_index_value(index, &doc) {
                        hnsw_inserted.insert(index, id);
                        index.insert(id, vector.into_owned(), now_ms)?;
                    }
                }
            }

            Ok(())
        })();

        let rollback_indexes = || {
            for (k, v) in btree_inserted {
                k.remove(id, &v, now_ms);
            }
            for (k, v) in bm25_inserted {
                k.remove(v.0, &v.1, now_ms);
            }
            for (k, v) in hnsw_inserted {
                k.remove(v, now_ms);
            }
            for (k, v) in btree_removed {
                let _ = k.insert(id, v.into_owned(), now_ms);
            }
            for (k, v) in bm25_removed {
                let _ = k.insert(v.0, &v.1, now_ms);
            }
            for (k, v) in hnsw_removed {
                let _ = k.insert(v.0, v.1.to_vec(), now_ms);
            }
        };

        if let Err(err) = rt {
            rollback_indexes();
            return Err(err);
        }

        // persist the updated document with update version
        let path = Self::doc_path(id);
        if let Err(err) = self.storage.put(&path, &doc, Some(ver)).await {
            rollback_indexes();
            return Err(err);
        }

        self.update_metadata(|meta| {
            meta.stats.last_updated = now_ms;
            meta.stats.version += 1;
            meta.stats.update_count += 1;
        });

        Ok(())
    }

    /// Removes a document from the collection by its ID.
    ///
    /// This method:
    /// 1. Removes the document ID from the bitmap
    /// 2. Removes document segments from memory
    /// 3. Updates all relevant indexes
    /// 4. Deletes the document from storage
    ///
    /// # Arguments
    /// * `id` - The ID of the document to remove
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if removal fails
    ///
    /// # Errors
    /// Returns an error if:
    /// - The collection is in read-only mode
    /// - Any index update fails
    /// - Storage operations fail
    pub async fn remove(&self, id: DocumentId) -> Result<(), DBError> {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "Collection is read-only".into(),
            });
        }

        let now_ms = unix_ms();
        {
            if !self.doc_ids_index.write().remove(&id) {
                return Ok(());
            }

            self.doc_ids.write().remove(id);
        }

        self.update_metadata(|meta| {
            meta.stats.last_deleted = now_ms;
            meta.stats.version += 1;
            meta.stats.delete_count += 1;
        });

        if let Ok((doc, _)) = self.storage.get::<DocumentOwned>(&Self::doc_path(id)).await {
            let doc = Document::try_from_doc(self.schema(), doc)?;
            for index in &self.btree_indexes {
                if let Some(fv) = self.index_hooks.btree_index_value(index, &doc) {
                    if fv.as_ref() != &FieldValue::Null {
                        index.remove(id, &fv, now_ms);
                    }
                }
            }

            for index in &self.bm25_indexes {
                if let Some(text) = self.index_hooks.bm25_index_value(index, &doc) {
                    index.remove(id, &text, now_ms);
                }
            }

            for index in &self.hnsw_indexes {
                if self.index_hooks.hnsw_index_value(index, &doc).is_some() {
                    index.remove(id, now_ms);
                }
            }
            let _ = self.storage.delete(&Self::doc_path(id)).await;
        }

        Ok(())
    }

    /// Searches for documents matching the given query and returns them.
    ///
    /// # Arguments
    /// * `query` - The search query parameters
    ///
    /// # Returns
    /// A vector of matching documents, or an error if the search fails
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

    /// Searches for documents matching the given query and deserializes them into the specified type.
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize documents into
    ///
    /// # Arguments
    /// * `query` - The search query parameters
    ///
    /// # Returns
    /// A vector of deserialized objects of type T, or an error if the search or deserialization fails
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

    /// Searches for documents matching the given query and returns only their IDs.
    ///
    /// This is more efficient than retrieving full documents when only IDs are needed.
    ///
    /// # Arguments
    /// * `query` - The search query parameters
    ///
    /// # Returns
    /// A vector of matching document IDs, or an error if the search fails
    pub async fn search_ids(&self, query: Query) -> Result<Vec<DocumentId>, DBError> {
        let limit = query.limit.unwrap_or(10).min(1000);
        let top_k = limit * 3;
        let mut candidates = Vec::with_capacity(top_k);
        let mut result = Vec::new();

        self.search_count.fetch_add(1, Ordering::Relaxed);
        if let Some(params) = query.search {
            let mut results: Vec<Vec<u64>> = Vec::new();

            if let Some(ref text) = params.text {
                for index in self.bm25_indexes.iter() {
                    let rt = if params.logical_search {
                        index.search_advanced(text, top_k, params.bm25_params.clone())
                    } else {
                        index.search(text, top_k, params.bm25_params.clone())
                    };
                    results.push(rt.into_iter().map(|r| r.0).collect());
                }
            }

            if let Some(ref vector) = params.vector {
                for index in self.hnsw_indexes.iter() {
                    let rt = index.search(vector, top_k);
                    results.push(rt.into_iter().map(|r| r.0).collect());
                }
            }

            let reranker = params.reranker.unwrap_or_default();
            let reranked = reranker.rerank(&results);
            candidates.extend(reranked.into_iter().map(|(id, _)| id));

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

    /// Queries document IDs based on a filter condition.
    ///
    /// # Arguments
    /// * `filter` - The filter condition to apply
    /// * `limit` - Maximum number of results to return (0 means no limit)
    /// # Returns
    /// A vector of document IDs matching the filter, or an error if filtering fails.
    pub async fn query_ids(
        &self,
        filter: Filter,
        limit: Option<usize>,
    ) -> Result<Vec<DocumentId>, DBError> {
        self.search_count.fetch_add(1, Ordering::Relaxed);
        self.filter_by_field(filter, &[], limit.unwrap_or(0))
    }

    /// Gets a document by its ID.
    ///
    /// # Arguments
    /// * `id` - The ID of the document to retrieve
    ///
    /// # Returns
    /// The document if found, or an error if retrieval fails
    pub async fn get(&self, id: DocumentId) -> Result<Document, DBError> {
        if self.doc_ids.read().contains(id) {
            self.get_count.fetch_add(1, Ordering::Relaxed);

            let path = Self::doc_path(id);
            if let Ok((doc, _)) = self.storage.get::<DocumentOwned>(&path).await {
                let doc = Document::try_from_doc(self.schema(), doc)?;
                return Ok(doc);
            }
        }
        Err(DBError::NotFound {
            name: id.to_string(),
            path: self.name.clone(),
            source: format!("Document with ID {id} not found").into(),
        })
    }

    /// Gets a document by its ID and deserializes it into the specified type.
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize the document into
    ///
    /// # Arguments
    /// * `id` - The ID of the document to retrieve
    ///
    /// # Returns
    /// The deserialized object of type T if found, or an error if retrieval or deserialization fails
    pub async fn get_as<T>(&self, id: DocumentId) -> Result<T, DBError>
    where
        T: DeserializeOwned,
    {
        let doc = self.get(id).await?;
        let obj = doc.try_into()?;
        Ok(obj)
    }

    /// Filters documents by a field condition.
    ///
    /// # Arguments
    /// * `filter` - The filter condition to apply
    /// * `candidates` - Optional list of document IDs to filter (if empty, all documents are considered)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// A vector of document IDs matching the filter, or an error if filtering fails
    fn filter_by_field(
        &self,
        filter: Filter,
        candidates: &[DocumentId],
        limit: usize,
    ) -> Result<Vec<DocumentId>, DBError> {
        let mut result = Vec::new();
        match filter {
            Filter::Field((index_name, filter)) => {
                if index_name == Schema::ID_KEY {
                    let filter: RangeQuery<u64> =
                        RangeQuery::try_convert_from(filter).map_err(|err| DBError::Generic {
                            name: self.name.clone(),
                            source: err,
                        })?;
                    Ok(self.filter_by_id(filter, candidates, limit))
                } else if let Some(index) =
                    self.btree_indexes.iter().find(|i| i.name() == index_name)
                {
                    result.reserve_exact(limit);
                    let _: Vec<()> = index.range_query_with(filter, |_, ids| {
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
                    Ok(result)
                } else {
                    Err(DBError::Index {
                        name: self.name.clone(),
                        source: format!("BTree index {index_name:?} not found").into(),
                    })
                }
            }
            Filter::Or(queries) => {
                let mut rt: UniqueVec<u64> = UniqueVec::with_capacity(limit);
                for query in queries {
                    let ids = self.filter_by_field(*query, candidates, limit)?;
                    rt.extend(ids);
                    if limit > 0 && rt.len() >= limit {
                        break;
                    }
                }

                result = rt.into();
                if limit > 0 && result.len() > limit {
                    result.truncate(limit);
                }
                Ok(result)
            }
            Filter::And(queries) => {
                let mut iter = queries.into_iter();
                if let Some(query) = iter.next() {
                    let mut rt: UniqueVec<u64> = self.filter_by_field(*query, &[], 0)?.into();

                    for query in iter {
                        let keys: UniqueVec<u64> = self.filter_by_field(*query, &[], 0)?.into();
                        rt.intersect_with(&keys);
                        if rt.is_empty() {
                            return Ok(vec![]);
                        }
                    }

                    result = rt.into();
                    if limit > 0 && result.len() > limit {
                        result.truncate(limit);
                    }
                }
                Ok(result)
            }
            Filter::Not(query) => {
                result.reserve_exact(limit);
                let exclude: HashSet<u64> =
                    self.filter_by_field(*query, &[], 0)?.into_iter().collect();
                for id in self.doc_ids_index.read().iter() {
                    if !exclude.contains(id) && (candidates.is_empty() || candidates.contains(id)) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            break;
                        }
                    }
                }
                Ok(result)
            }
        }
    }

    /// Filters documents by ID using a range query.
    ///
    /// # Arguments
    /// * `query` - The range query to apply to document IDs
    ///
    /// # Returns
    /// A vector of document IDs matching the range query
    fn filter_by_id(
        &self,
        query: RangeQuery<DocumentId>,
        candidates: &[DocumentId],
        limit: usize,
    ) -> Vec<DocumentId> {
        let mut result = Vec::new();
        match query {
            RangeQuery::Eq(id) => {
                if self.doc_ids_index.read().contains(&id)
                    && (candidates.is_empty() || candidates.contains(&id))
                {
                    result.push(id);
                }
            }
            RangeQuery::Gt(start_key) => {
                result.reserve_exact(limit);
                for id in self
                    .doc_ids_index
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                    .filter(|&id| *id > start_key)
                {
                    if candidates.is_empty() || candidates.contains(id) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::Ge(start_key) => {
                result.reserve_exact(limit);
                for id in self
                    .doc_ids_index
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                {
                    if candidates.is_empty() || candidates.contains(id) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::Lt(end_key) => {
                result.reserve_exact(limit);
                for id in self
                    .doc_ids_index
                    .read()
                    .range(std::ops::RangeTo { end: end_key })
                    .rev()
                {
                    if candidates.is_empty() || candidates.contains(id) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::Le(end_key) => {
                result.reserve_exact(limit);
                for id in self
                    .doc_ids_index
                    .read()
                    .range(std::ops::RangeToInclusive { end: end_key })
                    .rev()
                {
                    if candidates.is_empty() || candidates.contains(id) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::Between(start_key, end_key) => {
                result.reserve_exact(limit.min((1 + end_key - start_key) as usize));
                for id in self.doc_ids_index.read().range(start_key..=end_key) {
                    if candidates.is_empty() || candidates.contains(id) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::Include(ids) => {
                result.reserve_exact(limit.min(ids.len()));
                let doc_ids_index = self.doc_ids_index.read();
                for id in ids.into_iter() {
                    if doc_ids_index.contains(&id)
                        && (candidates.is_empty() || candidates.contains(&id))
                    {
                        result.push(id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
            RangeQuery::And(queries) => {
                let mut iter = queries.into_iter();
                if let Some(query) = iter.next() {
                    let mut rt: UniqueVec<u64> = self.filter_by_id(*query, candidates, 0).into();

                    for query in iter {
                        let keys: UniqueVec<u64> = self.filter_by_id(*query, candidates, 0).into();
                        rt.intersect_with(&keys);
                        if rt.is_empty() {
                            return vec![];
                        }
                    }

                    result = rt.into();
                    if limit > 0 && result.len() > limit {
                        result.truncate(limit);
                    }
                }
            }
            RangeQuery::Or(queries) => {
                let mut rt = UniqueVec::new();
                for query in queries {
                    let keys = self.filter_by_id(*query, candidates, 0);
                    rt.extend(keys);
                    if limit > 0 && rt.len() > limit {
                        break;
                    }
                }

                result = rt.into();
                if limit > 0 && result.len() > limit {
                    result.truncate(limit);
                }
            }
            RangeQuery::Not(query) => {
                result.reserve_exact(limit);
                // 先收集要排除的 key，再遍历全集差集
                let exclude: HashSet<u64> = self.filter_by_id(*query, &[], 0).into_iter().collect();
                for id in self.doc_ids_index.read().iter() {
                    if !exclude.contains(id) && (candidates.is_empty() || candidates.contains(id)) {
                        result.push(*id);
                        if limit > 0 && result.len() >= limit {
                            return result;
                        }
                    }
                }
            }
        }

        result
    }

    /// Updates the collection metadata with the provided function.
    ///
    /// # Arguments
    /// * `f` - A function that modifies the collection metadata
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
mod tests {
    use super::*;
    use crate::{
        database::{AndaDB, DBConfig},
        error::DBError,
        index::HnswConfig,
        query::{Filter, Query, RangeQuery, Search},
        schema::{AndaDBSchema, Document, Fv, Json, Schema, Vector},
        storage::StorageConfig,
    };
    use object_store::memory::InMemory;
    use serde::{Deserialize, Serialize};
    use std::{collections::BTreeMap, sync::Arc};

    // 测试用的文档结构
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, AndaDBSchema)]
    struct TestDoc {
        pub _id: u64,
        pub name: String,
        pub age: u32,
        pub tags: Vec<String>,
        pub metadata: BTreeMap<String, Json>,
        pub vector: Vector,
    }

    // 创建测试数据库和集合的辅助函数
    async fn setup_test_db() -> Result<AndaDB, DBError> {
        let object_store = Arc::new(InMemory::new());

        let db_config = DBConfig {
            name: "test_db".to_string(),
            description: "Test database".to_string(),
            storage: StorageConfig {
                compress_level: 0,
                ..Default::default()
            },
            lock: None,
        };

        let db = AndaDB::connect(object_store, db_config).await?;
        Ok(db)
    }

    // 创建测试集合的辅助函数
    async fn create_test_collection<F>(db: &AndaDB, f: F) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
        // 创建测试文档的模式
        let schema = TestDoc::schema()?;
        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test collection".to_string(),
        };

        let collection = db
            .open_or_create_collection(schema, collection_config, f)
            .await?;

        Ok(collection)
    }

    // 创建测试文档的辅助函数
    fn create_test_doc(_id: u64, name: &str, age: u32, tags: Vec<&str>) -> TestDoc {
        TestDoc {
            _id,
            name: name.to_string(),
            age,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            metadata: BTreeMap::new(),
            vector: vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                .into_iter()
                .map(bf16::from_f32)
                .collect(),
        }
    }

    #[tokio::test]
    async fn test_collection_create() -> Result<(), DBError> {
        let db = setup_test_db().await?;

        let collection = create_test_collection(&db, async |c| {
            c.create_bm25_index_nx(&["name", "tags", "metadata"])
                .await?;
            c.create_hnsw_index_nx("vector", HnswConfig::default())
                .await?;
            Ok(())
        })
        .await?;

        assert_eq!(collection.name(), "test_collection");
        assert_eq!(collection.metadata().config.description, "Test collection");

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_open() -> Result<(), DBError> {
        let db = setup_test_db().await?;

        // 首先创建集合
        {
            let collection = create_test_collection(&db, async |_| Ok(())).await?;
            assert_eq!(collection.name(), "test_collection");

            // 添加一个文档以确保有数据可以在重新打开时加载
            let doc = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
            let doc_obj = Document::try_from(collection.schema(), &doc)?;
            let id = collection.add(doc_obj).await?;
            assert_eq!(id, 1);

            // 刷新以确保数据被持久化
            collection.flush(unix_ms()).await?;
        }

        // 关闭并重新打开数据库
        db.close().await?;
        let db = AndaDB::connect(
            db.object_store(),
            DBConfig {
                name: "test_db".to_string(),
                description: "Test database".to_string(),
                storage: StorageConfig {
                    compress_level: 0,
                    ..Default::default()
                },
                lock: None,
            },
        )
        .await?;

        // 重新打开集合
        let collection = db
            .open_collection("test_collection".to_string(), async |_| Ok(()))
            .await?;

        assert_eq!(collection.name(), "test_collection");
        assert_eq!(collection.metadata().stats.num_documents, 1);

        // 验证文档是否正确加载
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "_id".to_string(),
                    RangeQuery::Eq(Fv::U64(1)),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Alice");

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_document_operations() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |_| Ok(())).await?;

        // 添加文档
        let doc1 = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
        let doc_obj1 = Document::try_from(collection.schema(), &doc1)?;
        let id1 = collection.add(doc_obj1).await?;
        assert_eq!(id1, 1);

        let doc2 = create_test_doc(0, "Bob", 25, vec!["tall", "quiet"]);
        let doc_obj2 = Document::try_from(collection.schema(), &doc2)?;
        let id2 = collection.add(doc_obj2).await?;
        assert_eq!(id2, 2);

        // 获取文档
        let result: TestDoc = collection.get_as(id1).await?;
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);

        // 删除文档
        collection.remove(id2).await?;

        // 验证删除
        let result = collection.get(id2).await;
        assert!(result.is_err());

        // 验证集合统计信息
        let stats = collection.stats();
        assert_eq!(stats.num_documents, 1);

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_index_operations() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |collection| {
            // 创建索引
            collection.create_btree_index_nx(&["name"]).await?;
            collection.create_btree_index_nx(&["age"]).await?;
            collection.create_btree_index_nx(&["tags"]).await?;

            // 创建搜索索引
            collection
                .create_bm25_index_nx(&["name", "tags", "metadata"])
                .await?;
            collection
                .create_hnsw_index_nx(
                    "vector",
                    HnswConfig {
                        dimension: 10,
                        ..Default::default()
                    },
                )
                .await?;
            Ok(())
        })
        .await?;

        // 添加测试文档
        for (name, age, tags) in [
            ("Alice", 30, vec!["smart", "friendly"]),
            ("Bob", 25, vec!["tall", "quiet"]),
            ("Charlie", 35, vec!["smart", "tall"]),
            ("David", 40, vec!["friendly", "quiet"]),
        ] {
            let doc = create_test_doc(0, name, age, tags);
            let doc_obj = Document::try_from(collection.schema(), &doc)?;
            collection.add(doc_obj).await?;
        }

        // 刷新以确保索引更新
        collection.flush(unix_ms()).await?;

        // 测试精确匹配查询
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "name".to_string(),
                    RangeQuery::Eq(Fv::Text("Alice".to_string())),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Alice");

        // 测试范围查询
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "age".to_string(),
                    RangeQuery::Gt(Fv::U64(30)),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|doc| doc.name == "Charlie"));
        assert!(result.iter().any(|doc| doc.name == "David"));

        // 测试数组字段查询
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "tags".to_string(),
                    RangeQuery::Eq(Fv::Text("smart".to_string())),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|doc| doc.name == "Alice"));
        assert!(result.iter().any(|doc| doc.name == "Charlie"));

        // 测试文本搜索
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                search: Some(Search {
                    text: Some("Alice".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Alice");

        // 测试向量搜索
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                search: Some(Search {
                    vector: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await?;

        assert!(!result.is_empty());

        // 测试复合查询
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                search: Some(Search {
                    text: Some("tall".to_string()),
                    ..Default::default()
                }),
                filter: Some(Filter::Field((
                    "age".to_string(),
                    RangeQuery::Lt(Fv::U64(30)),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Bob");

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_persistence() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let object_store = db.object_store();

        // 创建集合并添加文档
        {
            let collection = create_test_collection(&db, async |collection| {
                // 创建索引
                collection.create_btree_index_nx(&["name"]).await?;
                collection.create_btree_index_nx(&["age"]).await?;
                collection.create_btree_index_nx(&["tags"]).await?;

                // 创建搜索索引
                collection
                    .create_bm25_index_nx(&["name", "tags", "metadata"])
                    .await?;
                collection
                    .create_hnsw_index_nx(
                        "vector",
                        HnswConfig {
                            dimension: 10,
                            ..Default::default()
                        },
                    )
                    .await?;
                Ok(())
            })
            .await?;

            // 添加文档
            let doc = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
            let doc_obj = Document::try_from(collection.schema(), &doc)?;
            collection.add(doc_obj).await?;

            // 刷新以确保数据被持久化
            // collection.flush(unix_ms()).await?;

            // 关闭集合
            // collection.close().await?;
        }

        // 关闭并持久化数据库
        db.close().await?;

        // 重新打开数据库和集合
        let db = AndaDB::connect(
            object_store.clone(),
            DBConfig {
                name: "test_db".to_string(),
                description: "Test database".to_string(),
                storage: StorageConfig {
                    compress_level: 0,
                    ..Default::default()
                },
                lock: None,
            },
        )
        .await?;

        let collection = db
            .open_collection("test_collection".to_string(), async |_| Ok(()))
            .await?;

        // 验证文档是否正确加载
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "name".to_string(),
                    RangeQuery::Eq(Fv::Text("Alice".to_string())),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "Alice");
        assert_eq!(result[0].age, 30);

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_read_only_mode() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |_| Ok(())).await?;

        // 添加一个文档
        let doc = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
        let doc_obj = Document::try_from(collection.schema(), &doc)?;
        collection.add(doc_obj).await?;

        // 设置为只读模式
        collection.set_read_only(true);

        // 尝试添加另一个文档，应该失败
        let doc2 = create_test_doc(0, "Bob", 25, vec!["tall", "quiet"]);
        let doc_obj2 = Document::try_from(collection.schema(), &doc2)?;
        let result = collection.add(doc_obj2).await;

        assert!(result.is_err());

        // 验证读取操作仍然有效
        let result: TestDoc = collection.get_as(1).await?;
        assert_eq!(result.name, "Alice");

        // 恢复为读写模式
        collection.set_read_only(false);

        // 现在应该可以添加文档
        let doc3 = create_test_doc(0, "Charlie", 35, vec!["smart", "tall"]);
        let doc_obj3 = Document::try_from(collection.schema(), &doc3)?;
        let id = collection.add(doc_obj3).await?;
        assert_eq!(id, 2);

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_error_handling() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |collection| {
            // 测试创建已存在的索引
            collection.create_btree_index_nx(&["name"]).await?;
            let result = collection.create_btree_index(&["name"]).await;
            assert!(result.is_err());
            Ok(())
        })
        .await?;

        // 测试获取不存在的文档
        let result = collection.get(999).await;
        assert!(result.is_err());

        // 测试删除不存在的文档
        let result = collection.remove(999).await;
        assert!(result.is_ok());

        // 测试无效的查询
        let result: Result<Vec<TestDoc>, DBError> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "non_existent_field".to_string(),
                    RangeQuery::Eq(Fv::Text("value".to_string())),
                ))),
                ..Default::default()
            })
            .await;

        assert!(result.is_err());

        db.close().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_operations() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |collection| {
            // 创建索引
            collection.create_btree_index_nx(&["name"]).await?;
            Ok(())
        })
        .await?;

        // 并发添加多个文档
        let mut handles = Vec::new();
        for i in 0..10 {
            let collection_clone = collection.clone();
            let handle = tokio::spawn(async move {
                let doc = create_test_doc(0, &format!("Person{i}"), 20 + i, vec!["tag"]);
                let doc_obj = Document::try_from(collection_clone.schema(), &doc).unwrap();
                collection_clone.add(doc_obj).await
            });
            handles.push(handle);
        }

        // 等待所有任务完成
        let mut ids = Vec::new();
        for handle in handles {
            let id = handle.await.unwrap()?;
            ids.push(id);
        }

        // 验证所有文档都被添加
        assert_eq!(ids.len(), 10);
        // 验证文档数量
        let stats = collection.stats();
        assert_eq!(stats.num_documents, 10);

        // 并发获取文档
        let mut handles = Vec::new();
        for id in ids {
            let collection_clone = collection.clone();
            let handle = tokio::spawn(async move { collection_clone.get_as::<TestDoc>(id).await });
            handles.push(handle);
        }

        // 等待所有任务完成
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_metadata_updates() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |_| Ok(())).await?;

        // 记录初始版本
        let initial_version = collection.metadata().stats.version;

        // 添加文档应该更新元数据
        let doc = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
        let doc_obj = Document::try_from(collection.schema(), &doc)?;
        collection.add(doc_obj).await?;

        // 验证版本已更新
        let new_version = collection.metadata().stats.version;
        assert!(new_version > initial_version);

        // 验证统计信息已更新
        let stats = collection.stats();
        assert_eq!(stats.num_documents, 1);
        assert_eq!(stats.insert_count, 1);

        // 删除文档应该更新元数据
        collection.remove(1).await?;

        // 验证统计信息已更新
        let stats = collection.stats();
        assert_eq!(stats.num_documents, 0);
        assert_eq!(stats.delete_count, 1);

        db.close().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_document_updates() -> Result<(), DBError> {
        let db = setup_test_db().await?;
        let collection = create_test_collection(&db, async |collection| {
            // 创建索引以测试更新对索引的影响
            collection.create_btree_index_nx(&["name"]).await?;
            collection.create_btree_index_nx(&["age"]).await?;
            collection.create_btree_index_nx(&["tags"]).await?;
            Ok(())
        })
        .await?;

        // 添加文档
        let doc = create_test_doc(0, "Alice", 30, vec!["smart", "friendly"]);
        let doc_obj = Document::try_from(collection.schema(), &doc)?;
        let id = collection.add(doc_obj).await?;

        // 更新文档
        let mut update_fields = BTreeMap::new();
        update_fields.insert("name".to_string(), Fv::Text("Alice Updated".to_string()));
        update_fields.insert("age".to_string(), Fv::U64(31));
        update_fields.insert(
            "tags".to_string(),
            Fv::Array(vec![
                Fv::Text("smart".to_string()),
                Fv::Text("friendly".to_string()),
                Fv::Text("updated".to_string()),
            ]),
        );

        collection.update(id, update_fields.clone()).await?;

        // 获取并验证更新后的文档
        let updated_doc: TestDoc = collection.get_as(id).await?;
        assert_eq!(updated_doc.name, "Alice Updated");
        assert_eq!(updated_doc.age, 31);
        assert_eq!(updated_doc.tags.len(), 3);
        assert!(updated_doc.tags.contains(&"updated".to_string()));

        // 通过索引验证更新是否生效
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "name".to_string(),
                    RangeQuery::Eq(Fv::Text("Alice Updated".to_string())),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].age, 31);

        // 验证原来的值不再能被索引查询到
        let result: Vec<TestDoc> = collection
            .search_as(Query {
                filter: Some(Filter::Field((
                    "name".to_string(),
                    RangeQuery::Eq(Fv::Text("Alice".to_string())),
                ))),
                ..Default::default()
            })
            .await?;

        assert_eq!(result.len(), 0);

        // 测试部分更新
        let mut partial_update = BTreeMap::new();
        partial_update.insert("age".to_string(), Fv::U64(32));

        collection.update(id, partial_update).await?;

        let partially_updated: TestDoc = collection.get_as(id).await?;
        assert_eq!(partially_updated.name, "Alice Updated"); // 未更改
        assert_eq!(partially_updated.age, 32); // 已更改

        // 测试更新不存在的文档
        let result = collection.update(999, update_fields.clone()).await;
        assert!(result.is_err());

        // 测试只读模式下的更新失败
        collection.set_read_only(true);
        let result = collection.update(id, update_fields.clone()).await;
        assert!(result.is_err());

        // 恢复读写模式
        collection.set_read_only(false);

        // 测试更新元数据字段
        let mut metadata_update = BTreeMap::new();
        let mut metadata_map = BTreeMap::new();
        metadata_map.insert("key1".to_string(), Fv::Text("value1".to_string()));
        metadata_map.insert("key2".to_string(), Fv::U64(42));
        metadata_update.insert("metadata".to_string(), Fv::Map(metadata_map));

        collection.update(id, metadata_update).await?;

        let doc_with_metadata: TestDoc = collection.get_as(id).await?;
        assert_eq!(doc_with_metadata.metadata.len(), 2);
        assert!(
            matches!(doc_with_metadata.metadata.get("key1"), Some(Json::String(s)) if s == "value1")
        );
        assert!(
            matches!(doc_with_metadata.metadata.get("key2"), Some(Json::Number(n)) if n.as_i64() == Some(42))
        );

        // 验证统计信息已更新
        let stats = collection.stats();
        assert_eq!(stats.update_count, 3); // 初始更新 + 部分更新 + 元数据更新，只读失败不计数

        db.close().await?;
        Ok(())
    }
}
