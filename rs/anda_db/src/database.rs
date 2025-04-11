use futures::try_join;
use object_store::ObjectStore;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::{
    collection::{Collection, CollectionConfig},
    error::DbError,
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
    const METADATA_PATH: &'static str = "db_metadata.cbor";

    pub async fn create(
        object_store: Arc<dyn ObjectStore>,
        config: DBConfig,
    ) -> Result<Self, DbError> {
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
                storage.store(unix_ms()).await?;
            }
            Err(err) => return Err(err),
        }

        Ok(Self {
            name: metadata.config.name.clone(),
            object_store,
            storage,
            metadata: RwLock::new(metadata),
            collections: RwLock::new(BTreeMap::new()),
        })
    }

    pub async fn connect(
        object_store: Arc<dyn ObjectStore>,
        config: DBConfig,
    ) -> Result<Self, DbError> {
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
            }),
            Err(DbError::NotFound { .. }) => {
                let metadata = DBMetadata {
                    config,
                    collections: BTreeSet::new(),
                };

                match storage.create(Self::METADATA_PATH, &metadata).await {
                    Ok(_) => {
                        // DB created successfully, and store storage metadata
                        storage.store(unix_ms()).await?;
                    }
                    Err(err) => return Err(err),
                }

                Ok(Self {
                    name: metadata.config.name.clone(),
                    object_store,
                    storage,
                    metadata: RwLock::new(metadata),
                    collections: RwLock::new(BTreeMap::new()),
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

    pub async fn create_collection(
        &self,
        schema: Schema,
        config: CollectionConfig,
    ) -> Result<Arc<Collection>, DbError> {
        let mut collections = self.collections.write();
        if collections.contains_key(&config.name) {
            return Err(DbError::AlreadyExists {
                name: config.name,
                path: self.name.clone(),
                source: "collection already exists".into(),
            });
        }

        // self.metadata.collections will check it exists again in Collection::create
        let collection = Collection::create(self, schema, config).await?;
        let collection = Arc::new(collection);
        collections.insert(collection.name().to_string(), collection.clone());
        {
            self.metadata
                .write()
                .collections
                .insert(collection.name().to_string());
        }
        self.store().await?;
        Ok(collection)
    }

    pub async fn open_or_create_collection(
        &self,
        schema: Schema,
        config: CollectionConfig,
    ) -> Result<Arc<Collection>, DbError> {
        {
            if let Some(collection) = self.collections.read().get(&config.name) {
                return Ok(collection.clone());
            }
        }
        {
            if !self.metadata.read().collections.contains(&config.name) {
                return self.create_collection(schema, config).await;
            }
        }

        let mut collections = self.collections.write();
        let collection = Collection::open(self, config.name).await?;
        let collection = Arc::new(collection);
        collections.insert(collection.name().to_string(), collection.clone());
        Ok(collection)
    }

    pub async fn open_collection(&self, name: String) -> Result<Arc<Collection>, DbError> {
        {
            if let Some(collection) = self.collections.read().get(&name) {
                return Ok(collection.clone());
            }
        }
        {
            if !self.metadata.read().collections.contains(&name) {
                return Err(DbError::NotFound {
                    name,
                    path: self.name.clone(),
                    source: "collection not found".into(),
                });
            }
        }

        let mut collections = self.collections.write();
        let collection = Collection::open(self, name).await?;
        let collection = Arc::new(collection);
        collections.insert(collection.name().to_string(), collection.clone());
        Ok(collection)
    }

    async fn store(&self) -> Result<(), DbError> {
        let metadata = self.metadata();

        try_join!(
            self.storage.put(Self::METADATA_PATH, &metadata, None),
            self.storage.store(unix_ms())
        )?;

        Ok(())
    }

    pub(crate) fn object_store(&self) -> Arc<dyn ObjectStore> {
        self.object_store.clone()
    }
}
