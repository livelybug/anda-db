use futures::{future::JoinAll, try_join};
use object_store::ObjectStore;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use crate::{
    collection::{Collection, CollectionConfig},
    error::DBError,
    schema::*,
    storage::{Storage, StorageConfig},
    unix_ms,
};

pub struct AndaDB {
    /// Database name
    name: String,
    object_store: Arc<dyn ObjectStore>,
    storage: Storage,
    metadata: RwLock<DBMetadata>,
    collections: RwLock<BTreeMap<String, Arc<Collection>>>,
    read_only: AtomicBool,
}

/// Collection configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBConfig {
    /// Index name
    pub name: String,

    /// Collection description
    pub description: String,

    pub storage: StorageConfig,

    /// Auto-commit interval in seconds (if enabled)
    pub auto_commit_interval: Option<u64>,
}

impl Default for DBConfig {
    fn default() -> Self {
        Self {
            name: "anda_db".to_string(),
            description: "Anda DB".to_string(),
            storage: StorageConfig::default(),
            auto_commit_interval: None,
        }
    }
}

/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBMetadata {
    pub config: DBConfig,

    pub collections: BTreeSet<String>,
}

impl AndaDB {
    const METADATA_PATH: &'static str = "db_meta.cbor";

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
                storage.store_metadata(unix_ms()).await?;
            }
            Err(err) => return Err(err),
        }

        Ok(Self {
            name: metadata.config.name.clone(),
            object_store,
            storage,
            metadata: RwLock::new(metadata),
            collections: RwLock::new(BTreeMap::new()),
            read_only: AtomicBool::new(false),
        })
    }

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
            Ok((metadata, _)) => Ok(Self {
                name: metadata.config.name.clone(),
                object_store,
                storage,
                metadata: RwLock::new(metadata),
                collections: RwLock::new(BTreeMap::new()),
                read_only: AtomicBool::new(false),
            }),
            Err(DBError::NotFound { .. }) => {
                let metadata = DBMetadata {
                    config,
                    collections: BTreeSet::new(),
                };

                match storage.create(Self::METADATA_PATH, &metadata).await {
                    Ok(_) => {
                        // DB created successfully, and store storage metadata
                        storage.store_metadata(unix_ms()).await?;
                    }
                    Err(err) => return Err(err),
                }

                Ok(Self {
                    name: metadata.config.name.clone(),
                    object_store,
                    storage,
                    metadata: RwLock::new(metadata),
                    collections: RwLock::new(BTreeMap::new()),
                    read_only: AtomicBool::new(false),
                })
            }
            Err(err) => Err(err),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn metadata(&self) -> DBMetadata {
        self.metadata.read().clone()
    }

    pub fn set_read_only(&self, read_only: bool) {
        self.read_only.store(read_only, Ordering::Release);
        log::info!(
            action = "set_read_only",
            database = self.name;
            "Database is set to read-only: {}",
            read_only
        );

        for collection in self.collections.read().values() {
            collection.set_read_only(read_only);
        }
    }

    pub async fn close(&self) -> Result<(), DBError> {
        self.set_read_only(true);
        let collections = self
            .collections
            .read()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let _rt = JoinAll::from_iter(collections.iter().map(|collection| collection.close())).await;
        let start = Instant::now();
        match self.store_metadata(unix_ms()).await {
            Ok(_) => {
                let elapsed = start.elapsed();
                log::info!(
                    action = "close",
                    database = self.name,
                    elapsed = elapsed.as_millis();
                    "Database closed successfully in {elapsed:?}",
                );
            }
            Err(err) => {
                let elapsed = start.elapsed();
                log::error!(
                    action = "close",
                    database = self.name,
                    elapsed = elapsed.as_millis();
                    "Failed to close database: {err:?}",
                );
                return Err(err);
            }
        }
        Ok(())
    }

    pub async fn create_collection<F>(
        &self,
        schema: Schema,
        config: CollectionConfig,
        mut f: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnMut(&mut Collection) -> Result<(), BoxError>,
    {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "database is read-only".into(),
            });
        }

        {
            if self.collections.read().contains_key(&config.name) {
                return Err(DBError::AlreadyExists {
                    name: config.name,
                    path: self.name.clone(),
                    source: "collection already exists".into(),
                });
            }
        }

        let start = Instant::now();
        // self.metadata.collections will check it exists again in Collection::create
        let mut collection = Collection::create(self, schema, config).await?;
        f(&mut collection)
            .await
            .map_err(|err| DBError::Collection {
                name: collection.name().to_string(),
                source: err,
            })?;
        let collection = Arc::new(collection);
        {
            let mut collections = self.collections.write();
            collections.insert(collection.name().to_string(), collection.clone());
            self.metadata
                .write()
                .collections
                .insert(collection.name().to_string());
        }

        self.store_metadata(unix_ms()).await?;
        let elapsed = start.elapsed();
        log::info!(
            action = "create_collection",
            database = self.name,
            collection = collection.name(),
            elapsed = elapsed.as_millis();
            "Create a collection successfully in {elapsed:?}",
        );
        Ok(collection)
    }

    pub async fn open_or_create_collection<F>(
        &self,
        schema: Schema,
        config: CollectionConfig,
        mut f: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnMut(&mut Collection) -> Result<(), BoxError>,
    {
        if self.read_only.load(Ordering::Relaxed) {
            return Err(DBError::Generic {
                name: self.name.clone(),
                source: "database is read-only".into(),
            });
        }

        {
            if let Some(collection) = self.collections.read().get(&config.name) {
                return Ok(collection.clone());
            }
        }

        {
            if !self.metadata.read().collections.contains(&config.name) {
                return self.create_collection(schema, config, f).await;
            }
        }

        let mut collection = Collection::open(self, config.name).await?;
        f(&mut collection)
            .await
            .map_err(|err| DBError::Collection {
                name: collection.name().to_string(),
                source: err,
            })?;
        let collection = Arc::new(collection);
        let mut collections = self.collections.write();
        collections.insert(collection.name().to_string(), collection.clone());
        Ok(collection)
    }

    pub async fn open_collection<F>(
        &self,
        name: String,
        mut f: F,
    ) -> Result<Arc<Collection>, DBError>
    where
        F: AsyncFnMut(&mut Collection) -> Result<(), BoxError>,
    {
        {
            if let Some(collection) = self.collections.read().get(&name) {
                return Ok(collection.clone());
            }
        }
        {
            if !self.metadata.read().collections.contains(&name) {
                return Err(DBError::NotFound {
                    name,
                    path: self.name.clone(),
                    source: "collection not found".into(),
                });
            }
        }

        let mut collection = Collection::open(self, name).await?;
        f(&mut collection)
            .await
            .map_err(|err| DBError::Collection {
                name: collection.name().to_string(),
                source: err,
            })?;
        let collection = Arc::new(collection);
        let mut collections = self.collections.write();
        collections.insert(collection.name().to_string(), collection.clone());
        Ok(collection)
    }

    async fn store_metadata(&self, now_ms: u64) -> Result<(), DBError> {
        let metadata = self.metadata();

        try_join!(
            self.storage.put(Self::METADATA_PATH, &metadata, None),
            self.storage.store_metadata(now_ms)
        )?;

        Ok(())
    }

    pub(crate) fn object_store(&self) -> Arc<dyn ObjectStore> {
        self.object_store.clone()
    }
}
