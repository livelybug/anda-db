use bytes::Bytes;
use ciborium::{from_reader, into_writer};
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use moka::future::Cache;
use object_store::{
    GetOptions, ObjectMeta, ObjectStore, PutMode, PutOptions, PutResult, UpdateVersion, path::Path,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use zstd_safe::{compress, decompress};

use crate::{error::DbError, schema::validate_field_name};

/// 基于 object_store 实现的 Anda DB 存储层
#[derive(Clone)]
pub struct Storage {
    inner: Arc<InnerStorage>,
}

struct InnerStorage {
    /// Object store reference
    object_store: Arc<dyn ObjectStore>,
    /// Base path for all documents
    base_path: Path,
    compress_level: i32,
    object_chunk_size: usize,
    // max object size that can be stored in a single put operation
    // oherwise, it will be split into multiple parts by fixed object_chunk_size size
    max_small_object_size: usize,
    // max_large_object_size = object_chunk_size * max_object_parts
    max_object_parts: usize,
    // object cache
    stats: StorageStatsAtomic,
    metadata: StorageMetadata,
    cache: Option<Cache<Path, Arc<(Bytes, ObjectVersion)>>>,
}

/// Configuration for object store storage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    /// Cache max capacity in items, 0 表示不启用缓存
    pub cache_max_capacity: u64,
    /// Compression level, 0 表示不压缩，1-22 表示压缩级别，默认为 3，22 表示最高压缩率
    pub compress_level: i32,
    pub object_chunk_size: usize,
    pub max_small_object_size: usize,
    pub max_object_parts: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            cache_max_capacity: 10000,
            compress_level: 3,
            object_chunk_size: 256 * 1024,
            max_small_object_size: 2000 * 1024,
            max_object_parts: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    pub name: String,

    /// Storage configuration.
    pub config: StorageConfig,

    /// Storage statistics.
    pub stats: StorageStats,
}

struct StorageStatsAtomic {
    version: AtomicU64,
    last_saved: AtomicU64,
    total_get_count: AtomicU64,
    total_fetch_count: AtomicU64,
    total_fetch_bytes: AtomicU64,
    total_put_count: AtomicU64,
    total_put_bytes: AtomicU64,
    total_delete_count: AtomicU64,
}

/// Storage statistics
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StorageStats {
    pub version: u64,
    pub last_saved: u64,
    pub total_get_count: u64,
    pub total_fetch_count: u64,
    pub total_fetch_bytes: u64,
    pub total_put_count: u64,
    pub total_put_bytes: u64,
    pub total_delete_count: u64,
}

impl From<&StorageStatsAtomic> for StorageStats {
    fn from(stats: &StorageStatsAtomic) -> Self {
        StorageStats {
            version: stats.version.load(Ordering::Relaxed),
            last_saved: stats.last_saved.load(Ordering::Relaxed),
            total_get_count: stats.total_get_count.load(Ordering::Relaxed),
            total_fetch_count: stats.total_fetch_count.load(Ordering::Relaxed),
            total_fetch_bytes: stats.total_fetch_bytes.load(Ordering::Relaxed),
            total_put_count: stats.total_put_count.load(Ordering::Relaxed),
            total_put_bytes: stats.total_put_bytes.load(Ordering::Relaxed),
            total_delete_count: stats.total_delete_count.load(Ordering::Relaxed),
        }
    }
}

impl From<&StorageStats> for StorageStatsAtomic {
    fn from(stats: &StorageStats) -> Self {
        StorageStatsAtomic {
            version: AtomicU64::new(stats.version),
            last_saved: AtomicU64::new(stats.last_saved),
            total_get_count: AtomicU64::new(stats.total_get_count),
            total_fetch_count: AtomicU64::new(stats.total_fetch_count),
            total_fetch_bytes: AtomicU64::new(stats.total_fetch_bytes),
            total_put_count: AtomicU64::new(stats.total_put_count),
            total_put_bytes: AtomicU64::new(stats.total_put_bytes),
            total_delete_count: AtomicU64::new(stats.total_delete_count),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ObjectVersion {
    /// The unique identifier for the newly created object
    ///
    /// <https://datatracker.ietf.org/doc/html/rfc9110#name-etag>
    pub e_tag: Option<String>,
    /// 很多实现 ObjectStore 的库没有使用这个字段
    pub version: Option<String>,
}

impl From<ObjectVersion> for UpdateVersion {
    fn from(version: ObjectVersion) -> Self {
        UpdateVersion {
            e_tag: version.e_tag,
            version: version.version,
        }
    }
}

impl From<PutResult> for ObjectVersion {
    fn from(version: PutResult) -> Self {
        ObjectVersion {
            e_tag: version.e_tag,
            version: version.version,
        }
    }
}

impl From<&ObjectMeta> for ObjectVersion {
    fn from(meta: &ObjectMeta) -> Self {
        ObjectVersion {
            e_tag: meta.e_tag.clone(),
            version: meta.version.clone(),
        }
    }
}

impl Storage {
    const METADATA_PATH: &'static str = "storage_metadata.cbor";

    pub async fn connect(
        name: String,
        object_store: Arc<dyn ObjectStore>,
        config: StorageConfig,
    ) -> Result<Storage, DbError> {
        validate_field_name(name.as_str())?;

        let stats = StorageStats::default();
        let metadata = StorageMetadata {
            name,
            config,
            stats,
        };

        let storage = Storage::new(object_store.clone(), metadata)?;
        match storage.fetch(Storage::METADATA_PATH).await {
            Ok((metadata, _)) => Storage::new(object_store, metadata),
            Err(DbError::NotFound { .. }) => Ok(storage),
            Err(err) => Err(err),
        }
    }

    pub async fn store(&self, now_ms: u64) -> Result<(), DbError> {
        let prev = self.inner.stats.last_saved.load(Ordering::Acquire);
        if prev >= now_ms {
            // Don't save if the last saved time is greater than now
            return Ok(());
        }

        let mut metadata = self.metadata();
        metadata.stats.last_saved = now_ms;

        let mut buf: Vec<u8> = Vec::new();

        into_writer(&metadata, &mut buf).map_err(|err| DbError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        self.put(Storage::METADATA_PATH, &metadata, None)
            .await?;

        self.inner.stats.version.fetch_add(1, Ordering::Relaxed);
        self.inner
            .stats
            .last_saved
            .fetch_max(now_ms, Ordering::Relaxed);

        Ok(())
    }

    fn new(
        object_store: Arc<dyn ObjectStore>,
        metadata: StorageMetadata,
    ) -> Result<Storage, DbError> {
        validate_field_name(metadata.name.as_str())?;
        // Create a cache with a size limit
        let cache = if metadata.config.cache_max_capacity > 0 {
            Some(
                Cache::builder()
                    .max_capacity(metadata.config.cache_max_capacity)
                    .build(),
            )
        } else {
            None
        };

        Ok(Storage {
            inner: Arc::new(InnerStorage {
                object_store,
                base_path: Path::from(metadata.name.as_str()),
                compress_level: metadata.config.compress_level,
                object_chunk_size: metadata.config.object_chunk_size,
                max_small_object_size: metadata.config.max_small_object_size,
                max_object_parts: metadata.config.max_object_parts,
                stats: (&metadata.stats).into(),
                metadata,
                cache,
            }),
        })
    }

    pub fn metadata(&self) -> StorageMetadata {
        StorageMetadata {
            name: self.inner.metadata.name.clone(),
            config: self.inner.metadata.config.clone(),
            stats: self.stats(),
        }
    }

    pub fn base_path(&self) -> &Path {
        &self.inner.base_path
    }

    pub fn stats(&self) -> StorageStats {
        (&self.inner.stats).into()
    }

    pub async fn fetch<T>(&self, doc_path: &str) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        let path = self.inner.base_path.child(doc_path);
        // Try to get the document
        self.inner_fetch(&path).await
    }

    async fn inner_fetch<T>(&self, path: &Path) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        // Try to get the document
        let result = self
            .inner
            .object_store
            .get_opts(path, GetOptions::default())
            .await
            .map_err(DbError::from)?;

        let size = result.meta.size;
        let version: ObjectVersion = (&result.meta).into();
        let bytes = result.bytes().await.map_err(DbError::from)?;
        if (bytes.len() as u64) < size {
            // TODO: get_range
        }

        let bytes = if self.inner.compress_level > 0 {
            try_decompress(bytes)
        } else {
            bytes
        };

        self.inner
            .stats
            .total_fetch_count
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .stats
            .total_fetch_bytes
            .fetch_add(size, Ordering::Relaxed);

        let doc: T = from_reader(&bytes[..]).map_err(|err| DbError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        if let Some(cache) = &self.inner.cache {
            cache
                .insert(path.clone(), Arc::new((bytes, version.clone())))
                .await;
        }

        Ok((doc, version))
    }

    pub async fn get<T>(&self, doc_path: &str) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        let path = self.inner.base_path.child(doc_path);

        self.inner_get(&path).await
    }

    async fn inner_get<T>(&self, path: &Path) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        if let Some(cache) = &self.inner.cache {
            if let Some(arc) = cache.get(path).await {
                let doc: T = from_reader(&arc.0[..]).map_err(|err| DbError::Serialization {
                    name: self.inner.base_path.to_string(),
                    source: err.into(),
                })?;
                self.inner
                    .stats
                    .total_get_count
                    .fetch_add(1, Ordering::Relaxed);
                return Ok((doc, arc.1.clone()));
            }
        }

        self.inner_fetch(path).await
    }

    pub async fn create<T>(&self, doc_path: &str, doc: &T) -> Result<ObjectVersion, DbError>
    where
        T: Serialize,
    {
        let path = self.inner.base_path.child(doc_path);
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DbError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        let buf_len = buf.len();
        if buf_len > self.inner.max_small_object_size {
            // TODO: multipart upload
            return Err(DbError::Generic {
                name: self.inner.base_path.to_string(),
                source: "Payload size exceeds limit".into(),
            });
        }

        let buf = if self.inner.compress_level > 0 {
            try_compress(buf, self.inner.compress_level)
        } else {
            buf
        };

        let buf_len = buf.len();
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        let result = self
            .inner
            .object_store
            .put_opts(&path, buf.into(), opts)
            .await
            .map_err(DbError::from)?;

        self.inner
            .stats
            .total_put_count
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .stats
            .total_put_bytes
            .fetch_add(buf_len as u64, Ordering::Relaxed);

        Ok(result.into())
    }

    pub async fn put<T>(
        &self,
        doc_path: &str,
        doc: &T,
        version: Option<ObjectVersion>,
    ) -> Result<ObjectVersion, DbError>
    where
        T: Serialize,
    {
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DbError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;
        self.put_bytes(doc_path, buf, version).await
    }

    pub async fn put_bytes(
        &self,
        doc_path: &str,
        data: Vec<u8>,
        version: Option<ObjectVersion>,
    ) -> Result<ObjectVersion, DbError> {
        let data_len = data.len();
        if data_len > self.inner.max_small_object_size {
            // TODO: multipart upload
            return Err(DbError::Generic {
                name: self.inner.base_path.to_string(),
                source: "Payload size exceeds limit".into(),
            });
        }
        let path = self.inner.base_path.child(doc_path);
        let data = if self.inner.compress_level > 0 {
            try_compress(data, self.inner.compress_level)
        } else {
            data
        };

        let data_len = data.len();
        let opts = PutOptions {
            mode: if let Some(v) = version {
                PutMode::Update(UpdateVersion {
                    e_tag: v.e_tag,
                    version: v.version,
                })
            } else {
                PutMode::Overwrite
            },
            ..Default::default()
        };

        let result = self
            .inner
            .object_store
            .put_opts(&path, data.into(), opts)
            .await
            .map_err(DbError::from)?;

        if let Some(cache) = &self.inner.cache {
            cache.remove(&path).await;
        }

        self.inner
            .stats
            .total_put_count
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .stats
            .total_put_bytes
            .fetch_add(data_len as u64, Ordering::Relaxed);

        Ok(result.into())
    }

    async fn delete(&self, doc_path: &str) -> Result<(), DbError> {
        let path = self.inner.base_path.child(doc_path);

        self.inner
            .object_store
            .delete(&path)
            .await
            .map_err(DbError::from)?;

        if let Some(cache) = &self.inner.cache {
            cache.remove(&path).await;
        }

        self.inner
            .stats
            .total_delete_count
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn list<T>(
        &self,
        prefix: Option<&str>,
        offset: Option<&str>,
    ) -> BoxStream<Result<(T, ObjectVersion), DbError>>
    where
        T: DeserializeOwned + Send,
    {
        let path_prefix = if let Some(p) = prefix {
            self.inner.base_path.child(p)
        } else {
            self.inner.base_path.clone()
        };

        let offset = offset.map(|o| self.inner.base_path.child(o));

        let stream = if let Some(offset) = offset {
            self.inner
                .object_store
                .list_with_offset(Some(&path_prefix), &offset)
        } else {
            self.inner.object_store.list(Some(&path_prefix))
        };
        let stream = stream
            .map_err(DbError::from)
            .try_filter_map(|meta| {
                let this = self.clone();
                async move {
                    let result = this.inner_get(&meta.location).await?;
                    Ok(Some(result))
                }
            })
            .boxed();
        stream
    }
}

fn try_decompress(data: Bytes) -> Bytes {
    let mut buf = Vec::with_capacity(data.len() * 2);
    match decompress(&mut buf, &data[..]) {
        Ok(_) => buf.into(),
        Err(_) => data,
    }
}

fn try_compress(data: Vec<u8>, compress_level: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() / 3);
    match compress(&mut buf, &data[..], compress_level) {
        Ok(_) => buf,
        Err(_) => data,
    }
}
