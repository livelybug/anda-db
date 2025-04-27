use anda_db_hnsw::HnswIndex;
use bytes::Bytes;
use std::{fmt::Debug, hash::Hash};

pub use anda_db_hnsw::{HnswConfig, HnswError, HnswMetadata, HnswStats};

use crate::{
    error::DBError,
    schema::{Fe, SegmentId, Vector},
    storage::{PutMode, Storage},
};

pub struct Hnsw {
    name: String,
    field: Fe,
    index: HnswIndex,
    storage: Storage, // 与 Collection 共享同一个 Storage 实例
}

impl Debug for Hnsw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HnswIndex({})", self.name)
    }
}

impl PartialEq for &Hnsw {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for &Hnsw {}
impl Hash for &Hnsw {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Hnsw {
    fn metadata_path(name: &str) -> String {
        format!("hnsw_indexes/{name}/meta.cbor")
    }

    fn ids_path(name: &str) -> String {
        format!("hnsw_indexes/{name}/ids.cbor")
    }

    fn node_path(name: &str, node: u64) -> String {
        format!("hnsw_indexes/{name}/n_{node}.cbor")
    }

    pub async fn new(
        name: String,
        field: Fe,
        config: HnswConfig,
        storage: Storage,
        now_ms: u64,
    ) -> Result<Self, DBError> {
        let path = Hnsw::metadata_path(&name);
        let index = HnswIndex::new(name.clone(), Some(config));
        let mut data = Vec::new();
        index.store_metadata(&mut data, now_ms)?;
        storage
            .put_bytes(&path, data.into(), PutMode::Create)
            .await?;
        Ok(Self {
            name,
            field,
            index,
            storage,
        })
    }

    pub async fn bootstrap(name: String, field: Fe, storage: Storage) -> Result<Self, DBError> {
        let (metadata, _) = storage.fetch_raw(&Hnsw::metadata_path(&name)).await?;
        let (ids, _) = storage.fetch_raw(&Hnsw::ids_path(&name)).await?;
        let index = HnswIndex::load_all(&metadata[..], &ids[..], async |id: u64| {
            let path = Hnsw::node_path(&name, id);
            match storage.fetch_raw(&path).await {
                Ok((data, _)) => Ok(Some(data.into())),
                Err(DBError::NotFound { .. }) => Ok(None),
                Err(e) => Err(e.into()),
            }
        })
        .await?;

        Ok(Self {
            name,
            field,
            index,
            storage,
        })
    }

    pub async fn flush(&self, now_ms: u64) -> Result<bool, DBError> {
        let mut data = Vec::new();
        if !self.index.store_metadata(&mut data, now_ms)? {
            return Ok(false);
        }

        let path = Hnsw::metadata_path(&self.name);
        self.storage
            .put_bytes(&path, Bytes::copy_from_slice(&data[..]), PutMode::Overwrite)
            .await?;

        data.clear();
        self.index.store_ids(&mut data)?;
        let path = Hnsw::ids_path(&self.name);
        self.storage
            .put_bytes(&path, Bytes::copy_from_slice(&data[..]), PutMode::Overwrite)
            .await?;
        self.index
            .store_dirty_nodes(async |id, data| {
                let path = Hnsw::node_path(&self.name, id);
                let _ = self
                    .storage
                    .put_bytes(&path, Bytes::copy_from_slice(data), PutMode::Overwrite)
                    .await?;
                Ok(true)
            })
            .await?;

        Ok(true)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn field_name(&self) -> &str {
        self.field.name()
    }

    pub fn stats(&self) -> HnswStats {
        self.index.stats()
    }

    pub fn metadata(&self) -> HnswMetadata {
        self.index.metadata()
    }

    pub fn insert(&self, id: SegmentId, vector: Vector, now_ms: u64) -> Result<(), DBError> {
        self.index.insert(id, vector, now_ms)?;
        Ok(())
    }

    pub fn remove(&self, id: SegmentId, now_ms: u64) -> bool {
        self.index.remove(id, now_ms)
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        self.index.search_f32(query, top_k).unwrap_or_default()
    }
}
