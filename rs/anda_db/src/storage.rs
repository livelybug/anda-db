use bytes::Bytes;
use ciborium::{from_reader, into_writer};
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use moka::future::Cache;
use object_store::{
    GetOptions, ObjectMeta, ObjectStore, PutMode, PutOptions, PutResult, UpdateVersion,
    memory::InMemory, path::Path,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use zstd_safe::{compress, decompress};

use crate::error::DbError;

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

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStatsOwned {
    pub total_get_count: u64,
    pub total_fetch_count: u64,
    pub total_fetch_bytes: u64,
    pub total_put_count: u64,
    pub total_put_bytes: u64,
    pub total_delete_count: u64,
}

/// Configuration for object store storage
#[derive(Debug, Clone)]
pub struct StorageBuilder {
    /// The object store instance
    object_store: Arc<dyn ObjectStore>,
    /// Base path within the object store
    base_path: Path,
    /// Cache max capacity in items, 0 表示不启用缓存
    cache_max_capacity: u64,
    /// Compression level, 0 表示不压缩，1-22 表示压缩级别，默认为 3，22 表示最高压缩率
    compress_level: i32,
}

impl Default for StorageBuilder {
    fn default() -> Self {
        Self {
            object_store: Arc::new(InMemory::new()),
            base_path: Path::from("_sys"),
            cache_max_capacity: 10000,
            compress_level: 3,
        }
    }
}

impl StorageBuilder {
    pub fn with_object_store(mut self, object_store: Arc<dyn ObjectStore>) -> Self {
        self.object_store = object_store;
        self
    }

    pub fn with_base_path<P: Into<Path>>(mut self, base_path: P) -> Self {
        self.base_path = base_path.into();
        self
    }

    pub fn with_cache_max_capacity(mut self, cache_max_capacity: u64) -> Self {
        self.cache_max_capacity = cache_max_capacity;
        self
    }

    pub fn with_compress_level(mut self, compress_level: i32) -> Self {
        self.compress_level = compress_level;
        self
    }

    pub fn build(self) -> Result<Storage, DbError> {
        // Create a cache with a size limit
        let cache = if self.cache_max_capacity > 0 {
            Some(Arc::new(
                Cache::builder()
                    .max_capacity(self.cache_max_capacity)
                    .build(),
            ))
        } else {
            None
        };

        Ok(Storage {
            object_store: self.object_store,
            base_path: self.base_path,
            compress_level: self.compress_level,
            chunk_size: 256 * 1024,
            max_payload_size: 2000 * 1024,
            max_parts: 1024,
            cache,
            stats: Arc::new(StorageStats {
                total_get_count: AtomicU64::new(0),
                total_fetch_count: AtomicU64::new(0),
                total_fetch_bytes: AtomicU64::new(0),
                total_put_count: AtomicU64::new(0),
                total_put_bytes: AtomicU64::new(0),
                total_delete_count: AtomicU64::new(0),
            }),
        })
    }
}

/// 基于 object_store 实现的 Anda DB 存储层
#[derive(Clone)]
pub struct Storage {
    /// Object store reference
    object_store: Arc<dyn ObjectStore>,
    /// Base path for all documents
    base_path: Path,
    compress_level: i32,
    chunk_size: usize,
    max_payload_size: usize,
    max_parts: usize,
    /// Document cache
    cache: Option<Arc<Cache<Path, Arc<(Bytes, ObjectVersion)>>>>,
    stats: Arc<StorageStats>,
}

struct StorageStats {
    total_get_count: AtomicU64,
    total_fetch_count: AtomicU64,
    total_fetch_bytes: AtomicU64,
    total_put_count: AtomicU64,
    total_put_bytes: AtomicU64,
    total_delete_count: AtomicU64,
}

impl Storage {
    pub async fn fetch<T>(&self, doc_path: &str) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        let path = self.base_path.child(doc_path);
        // Try to get the document
        self.inner_fetch(&path).await
    }

    async fn inner_fetch<T>(&self, path: &Path) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        // Try to get the document
        let result = self
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

        let bytes = if self.compress_level > 0 {
            try_decompress(bytes)
        } else {
            bytes
        };

        self.stats.total_fetch_count.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_fetch_bytes
            .fetch_add(size, Ordering::Relaxed);

        let doc: T = from_reader(&bytes[..]).map_err(|err| DbError::Serialization {
            name: self.base_path.to_string(),
            source: err.into(),
        })?;

        if let Some(cache) = &self.cache {
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
        let path = self.base_path.child(doc_path);

        self.inner_get(&path).await
    }

    async fn inner_get<T>(&self, path: &Path) -> Result<(T, ObjectVersion), DbError>
    where
        T: DeserializeOwned,
    {
        if let Some(cache) = &self.cache {
            if let Some(arc) = cache.get(path).await {
                let doc: T = from_reader(&arc.0[..]).map_err(|err| DbError::Serialization {
                    name: self.base_path.to_string(),
                    source: err.into(),
                })?;
                self.stats.total_get_count.fetch_add(1, Ordering::Relaxed);
                return Ok((doc, arc.1.clone()));
            }
        }

        self.inner_fetch(path).await
    }

    pub async fn create<T>(&self, doc_path: &str, doc: &T) -> Result<ObjectVersion, DbError>
    where
        T: Serialize,
    {
        let path = self.base_path.child(doc_path);
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DbError::Serialization {
            name: self.base_path.to_string(),
            source: err.into(),
        })?;

        let buf_len = buf.len();
        if buf_len > self.max_payload_size {
            // TODO: multipart upload
            return Err(DbError::Generic {
                name: self.base_path.to_string(),
                source: "Payload size exceeds limit".into(),
            });
        }

        let buf = if self.compress_level > 0 {
            try_compress(buf, self.compress_level)
        } else {
            buf
        };

        let buf_len = buf.len();
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        let result = self
            .object_store
            .put_opts(&path, buf.into(), opts)
            .await
            .map_err(DbError::from)?;

        self.stats.total_put_count.fetch_add(1, Ordering::Relaxed);
        self.stats
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
        let path = self.base_path.child(doc_path);
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DbError::Serialization {
            name: self.base_path.to_string(),
            source: err.into(),
        })?;

        let buf_len = buf.len();
        if buf_len > self.max_payload_size {
            // TODO: multipart upload
            return Err(DbError::Generic {
                name: self.base_path.to_string(),
                source: "Payload size exceeds limit".into(),
            });
        }

        let buf = if self.compress_level > 0 {
            try_compress(buf, self.compress_level)
        } else {
            buf
        };

        let buf_len = buf.len();
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
            .object_store
            .put_opts(&path, buf.into(), opts)
            .await
            .map_err(DbError::from)?;

        if let Some(cache) = &self.cache {
            cache.remove(&path).await;
        }

        self.stats.total_put_count.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_put_bytes
            .fetch_add(buf_len as u64, Ordering::Relaxed);

        Ok(result.into())
    }

    async fn delete(&self, doc_path: &str) -> Result<(), DbError> {
        let path = self.base_path.child(doc_path);

        self.object_store
            .delete(&path)
            .await
            .map_err(DbError::from)?;

        if let Some(cache) = &self.cache {
            cache.remove(&path).await;
        }

        self.stats
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
            self.base_path.child(p)
        } else {
            self.base_path.clone()
        };

        let offset = offset.map(|o| self.base_path.child(o));

        let stream = if let Some(offset) = offset {
            self.object_store
                .list_with_offset(Some(&path_prefix), &offset)
        } else {
            self.object_store.list(Some(&path_prefix))
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

    pub fn stats(&self) -> StorageStatsOwned {
        StorageStatsOwned {
            total_get_count: self.stats.total_get_count.load(Ordering::Relaxed),
            total_fetch_count: self.stats.total_fetch_count.load(Ordering::Relaxed),
            total_fetch_bytes: self.stats.total_fetch_bytes.load(Ordering::Relaxed),
            total_put_count: self.stats.total_put_count.load(Ordering::Relaxed),
            total_put_bytes: self.stats.total_put_bytes.load(Ordering::Relaxed),
            total_delete_count: self.stats.total_delete_count.load(Ordering::Relaxed),
        }
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
