use futures::future::join_all;
use ic_auth_types::ByteBufB64;
use object_store::ObjectStore;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::{
    fmt::Debug,
    time::{Duration, Instant},
};
use tokio_util::sync::CancellationToken;

use crate::{
    collection::{Collection, CollectionConfig},
    error::DBError,
    schema::*,
    storage::{Storage, StorageConfig},
    unix_ms,
};

/// Main database structure that manages collections and storage.
///
/// AndaDB provides a high-level interface for creating, opening, and managing
/// collections of documents. It handles persistence through an object store
/// and maintains metadata about the database and its collections.
#[derive(Clone)]
pub struct AndaDB {
    inner: Arc<InnerDB>,
}

struct InnerDB {
    /// Database name
    name: String,
    /// Underlying object storage implementation
    object_store: Arc<dyn ObjectStore>,
    /// Storage layer for database operations
    storage: Storage,
    /// Database metadata protected by a read-write lock
    metadata: RwLock<DBMetadata>,
    /// Map of collection names to collection instances
    collections: RwLock<BTreeMap<String, Arc<Collection>>>,
    /// Flag indicating whether the database is in read-only mode
    read_only: AtomicBool,
}

/// Database configuration parameters.
///
/// Contains settings that define the database's behavior and properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBConfig {
    /// Database name
    pub name: String,

    /// Database description
    pub description: String,

    /// Storage configuration settings
    pub storage: StorageConfig,

    /// Optional opaque bytes as lock for the database
    pub lock: Option<ByteBufB64>,
}

impl Default for DBConfig {
    fn default() -> Self {
        Self {
            name: "anda_db".to_string(),
            description: "Anda DB".to_string(),
            storage: StorageConfig::default(),
            lock: None,
        }
    }
}

/// Database metadata.
///
/// Contains the database configuration and a set of collection names
/// that belong to this database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBMetadata {
    /// Database configuration
    pub config: DBConfig,

    /// Set of collection names in this database
    pub collections: BTreeSet<String>,
}

impl Debug for AndaDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AndaDB({})", self.inner.name)
    }
}

impl AndaDB {
    /// Path where database metadata is stored
    const METADATA_PATH: &'static str = "db_meta.cbor";

    /// Creates a new database with the given configuration.
    ///
    /// This method initializes a new database with the specified configuration
    /// and object store. It validates the database name, connects to storage,
    /// and creates the initial metadata.
    ///
    /// # Arguments
    /// * `object_store` - The object store implementation to use for persistence
    /// * `config` - The database configuration
    ///
    /// # Returns
    /// A Result containing either the new AndaDB instance or an error
    pub async fn create(
        object_store: Arc<dyn ObjectStore>,
        config: DBConfig,
    ) -> Result<Self, DBError> {
        validate_field_name(config.name.as_str())?;

        let storage = Storage::connect(
            config.name.clone(),
            object_store.clone(),
            config.storage.clone(),
        )
        .await?;

        let metadata = DBMetadata {
            config,
            collections: BTreeSet::new(),
        };

        match storage.create(Self::METADATA_PATH, &metadata).await {
            Ok(_) => {
                // DB created successfully, and store storage metadata
                storage.store_metadata(0, unix_ms()).await?;
            }
            Err(err) => return Err(err),
        }

        Ok(Self {
            inner: Arc::new(InnerDB {
                name: metadata.config.name.clone(),
                object_store,
                storage,
                metadata: RwLock::new(metadata),
                collections: RwLock::new(BTreeMap::new()),
                read_only: AtomicBool::new(false),
            }),
        })
    }

    /// Connects to an existing database or creates a new one if it doesn't exist.
    ///
    /// This method attempts to connect to an existing database with the given
    /// configuration. If the database doesn't exist, it creates a new one.
    ///
    /// # Arguments
    /// * `object_store` - The object store implementation to use for persistence
    /// * `config` - The database configuration
    ///
    /// # Returns
    /// A Result containing either the AndaDB instance or an error
    pub async fn connect(
        object_store: Arc<dyn ObjectStore>,
        config: DBConfig,
    ) -> Result<Self, DBError> {
        validate_field_name(config.name.as_str())?;

        let storage = Storage::connect(
            config.name.clone(),
            object_store.clone(),
            config.storage.clone(),
        )
        .await?;

        match storage.fetch::<DBMetadata>(Self::METADATA_PATH).await {
            Ok((metadata, _)) => {
                let set_lock = match (&metadata.config.lock, config.lock) {
                    (None, Some(lock)) => Some(lock),
                    (Some(existing_lock), lock) => {
                        if lock.as_ref() != Some(existing_lock) {
                            return Err(DBError::Storage {
                                name: config.name.clone(),
                                source: "Database lock mismatch".into(),
                            });
                        }
                        None
                    }
                    _ => None,
                };

                let this = Self {
                    inner: Arc::new(InnerDB {
                        name: metadata.config.name.clone(),
                        object_store,
                        storage,
                        metadata: RwLock::new(metadata),
                        collections: RwLock::new(BTreeMap::new()),
                        read_only: AtomicBool::new(false),
                    }),
                };

                if let Some(lock) = set_lock {
                    this.set_lock(lock).await?;
                }

                Ok(this)
            }
            Err(DBError::NotFound { .. }) => {
                let metadata = DBMetadata {
                    config,
                    collections: BTreeSet::new(),
                };

                match storage.create(Self::METADATA_PATH, &metadata).await {
                    Ok(_) => {
                        // DB created successfully, and store storage metadata
                        storage.store_metadata(0, unix_ms()).await?;
                    }
                    Err(err) => return Err(err),
                }

                Ok(Self {
                    inner: Arc::new(InnerDB {
                        name: metadata.config.name.clone(),
                        object_store,
                        storage,
                        metadata: RwLock::new(metadata),
                        collections: RwLock::new(BTreeMap::new()),
                        read_only: AtomicBool::new(false),
                    }),
                })
            }
            Err(err) => Err(err),
        }
    }

    /// Returns the name of the database.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns a clone of the database metadata.
    pub fn metadata(&self) -> DBMetadata {
        self.inner.metadata.read().clone()
    }

    /// Sets the database to read-only mode.
    ///
    /// When in read-only mode, operations that modify the database will fail.
    /// This setting is propagated to all collections in the database.
    ///
    /// # Arguments
    /// * `read_only` - Whether to enable read-only mode
    pub fn set_read_only(&self, read_only: bool) {
        self.inner.read_only.store(read_only, Ordering::Release);
        log::info!(
            action = "set_read_only",
            database = self.inner.name;
            "Database is set to read-only: {read_only}"
        );

        for collection in self.inner.collections.read().values() {
            collection.set_read_only(read_only);
        }
    }

    /// Closes the database, ensuring all data is flushed to storage.
    ///
    /// This method sets the database to read-only mode, closes all collections,
    /// and flushes any pending changes to storage.
    ///
    /// # Returns
    /// A Result indicating success or an error
    pub async fn close(&self) -> Result<(), DBError> {
        self.set_read_only(true);
        let collections = self
            .inner
            .collections
            .read()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let _rt = join_all(collections.iter().map(|collection| collection.close())).await;
        let start = Instant::now();
        match self.flush(unix_ms()).await {
            Ok(_) => {
                let elapsed = start.elapsed();
                log::info!(
                    action = "close",
                    database = self.inner.name,
                    elapsed = elapsed.as_millis();
                    "Database closed successfully in {elapsed:?}",
                );
            }
            Err(err) => {
                let elapsed = start.elapsed();
                log::error!(
                    action = "close",
                    database = self.inner.name,
                    elapsed = elapsed.as_millis();
                    "Failed to close database: {err:?}",
                );
                return Err(err);
            }
        }
        Ok(())
    }

    /// Automatically flushes the database at regular intervals.
    ///
    /// This method runs in a loop, waiting for the specified interval
    /// before flushing the database. When the cancellation token is triggered,
    /// the loop will exit and the database will be closed.
    ///
    /// # Arguments
    /// * `cancel_token` - A cancellation token to stop the loop
    /// * `interval` - The time interval between flushes
    ///
    pub async fn auto_flush(&self, cancel_token: CancellationToken, interval: Duration) {
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    let _ = self.close().await;
                    return;
                }
                _ = tokio::time::sleep(interval) => {}
            };

            let collections = self
                .inner
                .collections
                .read()
                .values()
                .cloned()
                .collect::<Vec<_>>();
            let _rt = join_all(collections.iter().map(|collection| collection.close())).await;
            let start = Instant::now();
            match self.flush(unix_ms()).await {
                Ok(_) => {
                    let elapsed = start.elapsed();
                    log::info!(
                        action = "auto_flush",
                        database = self.inner.name,
                        elapsed = elapsed.as_millis();
                        "Database auto-flushed successfully in {elapsed:?}",
                    );
                }
                Err(err) => {
                    let elapsed = start.elapsed();
                    log::error!(
                        action = "auto_flush",
                        database = self.inner.name,
                        elapsed = elapsed.as_millis();
                        "Failed to auto-flush database: {err:?}",
                    );
                }
            }
        }
    }

    /// Creates a new collection in the database.
    ///
    /// This method creates a new collection with the given schema and configuration.
    /// It also executes the provided function on the collection before finalizing creation.
    ///
    /// # Arguments
    /// * `schema` - The schema defining the structure of documents in the collection
    /// * `config` - The collection configuration
    /// * `f` - A function to execute on the collection during creation
    ///
    /// # Returns
    /// A Result containing either the new Collection or an error
    pub async fn create_collection<F>(
        &self,
        schema: Schema,
        config: CollectionConfig,
        f: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
        if self.inner.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.inner.name.clone(),
                source: "database is read-only".into(),
            });
        }

        {
            if self.inner.collections.read().contains_key(&config.name) {
                return Err(DBError::AlreadyExists {
                    name: config.name,
                    path: self.inner.name.clone(),
                    source: "collection already exists".into(),
                    _id: 0,
                });
            }
        }

        let start = Instant::now();
        // self.metadata.collections will check it exists again in Collection::create
        let mut collection = Collection::create(self.clone(), schema, config).await?;
        f(&mut collection).await?;
        let collection = Arc::new(collection);
        {
            let mut collections = self.inner.collections.write();
            collections.insert(collection.name().to_string(), collection.clone());
            self.inner
                .metadata
                .write()
                .collections
                .insert(collection.name().to_string());
        }

        let now = unix_ms();
        collection.flush(now).await?;
        self.flush(now).await?;
        let elapsed = start.elapsed();
        log::info!(
            action = "create_collection",
            database = self.inner.name,
            collection = collection.name(),
            elapsed = elapsed.as_millis();
            "Create a collection successfully in {elapsed:?}",
        );
        Ok(collection)
    }

    /// Opens an existing collection or creates a new one if it doesn't exist.
    ///
    /// This method attempts to open an existing collection with the given name.
    /// If the collection doesn't exist, it creates a new one with the provided
    /// schema and configuration.
    ///
    /// # Arguments
    /// * `schema` - The schema to use if creating a new collection
    /// * `config` - The collection configuration
    /// * `f` - A function to execute on the collection during opening/creation
    ///
    /// # Returns
    /// A Result containing either the Collection or an error
    pub async fn open_or_create_collection<F>(
        &self,
        schema: Schema,
        config: CollectionConfig,
        f: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
        if self.inner.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.inner.name.clone(),
                source: "database is read-only".into(),
            });
        }

        {
            if let Some(collection) = self.inner.collections.read().get(&config.name) {
                return Ok(collection.clone());
            }
        }

        {
            if !self
                .inner
                .metadata
                .read()
                .collections
                .contains(&config.name)
            {
                return self.create_collection(schema, config, f).await;
            }
        }

        self.open_collection(config.name, f).await
    }

    /// Opens an existing collection.
    ///
    /// This method attempts to open an existing collection with the given name.
    /// It fails if the collection doesn't exist.
    ///
    /// # Arguments
    /// * `name` - The name of the collection to open
    /// * `f` - A function to execute on the collection during opening
    ///
    /// # Returns
    /// A Result containing either the Collection or an error
    pub async fn open_collection<F>(&self, name: String, f: F) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnOnce(&mut Collection) -> Result<(), DBError>,
    {
        {
            if let Some(collection) = self.inner.collections.read().get(&name) {
                return Ok(collection.clone());
            }
        }
        {
            if !self.inner.metadata.read().collections.contains(&name) {
                return Err(DBError::NotFound {
                    name,
                    path: self.inner.name.clone(),
                    source: "collection not found".into(),
                    _id: 0,
                });
            }
        }

        let collection = Collection::open(self.clone(), name, f).await?;
        let collection = Arc::new(collection);
        {
            let mut collections = self.inner.collections.write();
            collections.insert(collection.name().to_string(), collection.clone());
        }
        let now = unix_ms();
        collection.flush(now).await?;
        Ok(collection)
    }

    async fn set_lock(&self, lock: ByteBufB64) -> Result<(), DBError> {
        {
            self.inner.metadata.write().config.lock = Some(lock);
        }

        let metadata = self.metadata();
        self.inner
            .storage
            .put(Self::METADATA_PATH, &metadata, None)
            .await?;
        Ok(())
    }

    /// Flushes database metadata to storage.
    ///
    /// This method writes the current database metadata to storage and
    /// updates the storage metadata with the current timestamp.
    ///
    /// # Arguments
    /// * `now_ms` - The current timestamp in milliseconds
    ///
    /// # Returns
    /// A Result indicating success or an error
    async fn flush(&self, now_ms: u64) -> Result<(), DBError> {
        let metadata = self.metadata();

        self.inner
            .storage
            .put(Self::METADATA_PATH, &metadata, None)
            .await?;
        self.inner.storage.store_metadata(0, now_ms).await?;
        Ok(())
    }

    /// Returns a clone of the object store.
    ///
    /// This method is used internally by collections to access the object store.
    pub(crate) fn object_store(&self) -> Arc<dyn ObjectStore> {
        self.inner.object_store.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Fe, Ft, Schema};
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn test_database_creation() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();

        let db = AndaDB::create(object_store, config).await.unwrap();
        assert_eq!(db.name(), "anda_db");
        assert!(db.metadata().collections.is_empty());
    }

    #[tokio::test]
    async fn test_database_connection() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig {
            name: "test_db".to_string(),
            description: "Test Database".to_string(),
            storage: StorageConfig::default(),
            lock: None,
        };

        // First create the database
        {
            let _db = AndaDB::create(object_store.clone(), config.clone())
                .await
                .unwrap();
        }

        // Then connect to it
        let db = AndaDB::connect(object_store, config).await.unwrap();
        assert_eq!(db.name(), "test_db");
    }

    #[tokio::test]
    async fn test_create_collection() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();
        let db = AndaDB::create(object_store, config).await.unwrap();

        let mut schema = Schema::builder();
        schema
            .add_field(Fe::new("name".to_string(), Ft::Text).unwrap())
            .unwrap();
        let schema = schema.build().unwrap();

        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test Collection".to_string(),
        };

        let collection = db
            .create_collection(schema.clone(), collection_config.clone(), async |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(collection.name(), "test_collection");
        assert!(db.metadata().collections.contains("test_collection"));
    }

    #[tokio::test]
    async fn test_open_collection() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();
        let db = AndaDB::create(object_store, config).await.unwrap();

        let mut schema = Schema::builder();
        schema
            .add_field(Fe::new("name".to_string(), Ft::Text).unwrap())
            .unwrap();
        let schema = schema.build().unwrap();

        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test Collection".to_string(),
        };

        // Create collection first
        db.create_collection(schema.clone(), collection_config.clone(), async |_| Ok(()))
            .await
            .unwrap();

        // Then open it
        let collection = db
            .open_collection("test_collection".to_string(), async |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(collection.name(), "test_collection");
    }

    #[tokio::test]
    async fn test_open_or_create_collection() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();
        let db = AndaDB::create(object_store, config).await.unwrap();

        let mut schema = Schema::builder();
        schema
            .add_field(Fe::new("name".to_string(), Ft::Text).unwrap())
            .unwrap();
        let schema = schema.build().unwrap();

        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test Collection".to_string(),
        };

        // First call should create the collection
        let collection1 = db
            .open_or_create_collection(schema.clone(), collection_config.clone(), async |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(collection1.name(), "test_collection");

        // Second call should open the existing collection
        let collection2 = db
            .open_or_create_collection(schema.clone(), collection_config.clone(), async |_| Ok(()))
            .await
            .unwrap();

        assert_eq!(collection2.name(), "test_collection");
    }

    #[tokio::test]
    async fn test_read_only_mode() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();
        let db = AndaDB::create(object_store, config).await.unwrap();

        let mut schema = Schema::builder();
        schema
            .add_field(Fe::new("name".to_string(), Ft::Text).unwrap())
            .unwrap();
        let schema = schema.build().unwrap();

        // Create collection while DB is writable
        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test Collection".to_string(),
        };
        let _collection = db
            .create_collection(schema.clone(), collection_config.clone(), async |_| Ok(()))
            .await
            .unwrap();

        // Set database to read-only
        db.set_read_only(true);

        // Attempt to create another collection should fail
        let collection_config2 = CollectionConfig {
            name: "test_collection2".to_string(),
            description: "Test Collection 2".to_string(),
        };
        let result = db
            .create_collection(schema, collection_config2, async |_| Ok(()))
            .await;

        assert!(result.is_err());
        match result {
            Err(DBError::Generic { .. }) => (),
            _ => panic!("Expected Generic error due to read-only mode"),
        }
    }

    #[tokio::test]
    async fn test_database_close() {
        let object_store = Arc::new(InMemory::new());
        let config = DBConfig::default();
        let db = AndaDB::create(object_store, config).await.unwrap();

        let mut schema = Schema::builder();
        schema
            .add_field(Fe::new("name".to_string(), Ft::Text).unwrap())
            .unwrap();
        let schema = schema.build().unwrap();

        let collection_config = CollectionConfig {
            name: "test_collection".to_string(),
            description: "Test Collection".to_string(),
        };

        db.create_collection(schema, collection_config, async |_| Ok(()))
            .await
            .unwrap();

        // Close the database
        db.close().await.unwrap();

        // Database should be in read-only mode after closing
        assert!(db.inner.read_only.load(Ordering::Relaxed));
    }
}
