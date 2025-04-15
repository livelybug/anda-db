use async_compression::tokio::{bufread::ZstdDecoder, write::ZstdEncoder};
use async_trait::async_trait;
use bytes::Bytes;
use ciborium::{from_reader, into_writer};
use futures::{
    StreamExt, TryStreamExt,
    future::{BoxFuture, FutureExt},
    stream::BoxStream,
};
use moka::future::Cache;
use object_store::{
    GetOptions, ObjectMeta, ObjectStore, PutOptions, PutResult, UpdateVersion,
    buffered::{BufReader, BufWriter},
    path::Path,
};
use pin_project_lite::pin_project;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use zstd_safe::{compress, decompress};

pub use object_store::PutMode;

use crate::{error::DBError, schema::validate_field_name};

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
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            cache_max_capacity: 10000,
            compress_level: 3,
            object_chunk_size: 256 * 1024,
            max_small_object_size: 2000 * 1024,
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
    const METADATA_PATH: &'static str = "storage_meta.cbor";

    pub async fn connect(
        name: String,
        object_store: Arc<dyn ObjectStore>,
        config: StorageConfig,
    ) -> Result<Storage, DBError> {
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
            Err(DBError::NotFound { .. }) => Ok(storage),
            Err(err) => Err(err),
        }
    }

    pub async fn store(&self, now_ms: u64) -> Result<(), DBError> {
        let prev = self.inner.stats.last_saved.load(Ordering::Acquire);
        if prev >= now_ms {
            // Don't save if the last saved time is greater than now
            return Ok(());
        }

        let mut metadata = self.metadata();
        metadata.stats.last_saved = now_ms;

        let mut buf: Vec<u8> = Vec::new();

        into_writer(&metadata, &mut buf).map_err(|err| DBError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        self.put(Storage::METADATA_PATH, &metadata, None).await?;

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
    ) -> Result<Storage, DBError> {
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

    pub fn object_chunk_size(&self) -> usize {
        self.inner.object_chunk_size
    }

    pub fn stats(&self) -> StorageStats {
        (&self.inner.stats).into()
    }

    pub async fn fetch<T>(&self, doc_path: &str) -> Result<(T, ObjectVersion), DBError>
    where
        T: DeserializeOwned,
    {
        let path = self.inner.base_path.child(doc_path);
        // Try to get the document
        let (bytes, version) = self.inner_fetch(&path).await?;
        let doc: T = from_reader(&bytes[..]).map_err(|err| DBError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        if let Some(cache) = &self.inner.cache {
            if bytes.len() < self.inner.max_small_object_size {
                // Cache the document if it is small enough
                cache
                    .insert(path.clone(), Arc::new((bytes, version.clone())))
                    .await;
            }
        }

        Ok((doc, version))
    }

    pub async fn fetch_raw(&self, doc_path: &str) -> Result<(Bytes, ObjectVersion), DBError> {
        let path = self.inner.base_path.child(doc_path);
        // Try to get the document
        self.inner_fetch(&path).await
    }

    async fn inner_fetch(&self, path: &Path) -> Result<(Bytes, ObjectVersion), DBError> {
        // Try to get the document
        let result = self
            .inner
            .object_store
            .get_opts(path, GetOptions::default())
            .await
            .map_err(DBError::from)?;

        let size = result.meta.size;
        let version: ObjectVersion = (&result.meta).into();
        let bytes = result.bytes().await.map_err(DBError::from)?;
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

        Ok((bytes, version))
    }

    pub async fn get<T>(&self, doc_path: &str) -> Result<(T, ObjectVersion), DBError>
    where
        T: DeserializeOwned,
    {
        let path = self.inner.base_path.child(doc_path);

        self.inner_get(&path).await
    }

    async fn inner_get<T>(&self, path: &Path) -> Result<(T, ObjectVersion), DBError>
    where
        T: DeserializeOwned,
    {
        if let Some(cache) = &self.inner.cache {
            if let Some(arc) = cache.get(path).await {
                let doc: T = from_reader(&arc.0[..]).map_err(|err| DBError::Serialization {
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

        let (bytes, version) = self.inner_fetch(path).await?;
        let doc: T = from_reader(&bytes[..]).map_err(|err| DBError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        if let Some(cache) = &self.inner.cache {
            if bytes.len() < self.inner.max_small_object_size {
                // Cache the document if it is small enough
                cache
                    .insert(path.clone(), Arc::new((bytes, version.clone())))
                    .await;
            }
        }

        Ok((doc, version))
    }

    pub async fn stream_reader(&self, doc_path: &str) -> Result<StreamReader, DBError> {
        let path = self.inner.base_path.child(doc_path);
        let meta = self
            .inner
            .object_store
            .head(&path)
            .await
            .map_err(DBError::from)?;
        let reader = BufReader::with_capacity(
            self.inner.object_store.clone(),
            &meta,
            self.inner.object_chunk_size,
        );
        let empty = BufReader::new(Arc::new(NotImplementedObjectStore), &meta);
        if self.inner.compress_level > 0 {
            Ok(StreamReader {
                inner: self.inner.clone(),
                reader: empty,
                compression: ZstdDecoder::new(reader),
                with_compression: true,
                path,
            })
        } else {
            Ok(StreamReader {
                inner: self.inner.clone(),
                reader,
                compression: ZstdDecoder::new(empty),
                with_compression: false,
                path,
            })
        }
    }

    pub async fn create<T>(&self, doc_path: &str, doc: &T) -> Result<ObjectVersion, DBError>
    where
        T: Serialize,
    {
        let path = self.inner.base_path.child(doc_path);
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DBError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;

        self.inner.put(path, buf.into(), PutMode::Create).await
    }

    pub async fn put<T>(
        &self,
        doc_path: &str,
        doc: &T,
        version: Option<ObjectVersion>,
    ) -> Result<ObjectVersion, DBError>
    where
        T: Serialize,
    {
        let mut buf: Vec<u8> = Vec::new();

        into_writer(doc, &mut buf).map_err(|err| DBError::Serialization {
            name: self.inner.base_path.to_string(),
            source: err.into(),
        })?;
        let path = self.inner.base_path.child(doc_path);
        let mode = if let Some(version) = version {
            PutMode::Update(version.into())
        } else {
            PutMode::Overwrite
        };
        self.inner.put(path, buf.into(), mode).await
    }

    pub async fn put_bytes(
        &self,
        doc_path: &str,
        data: Bytes,
        version: Option<ObjectVersion>,
    ) -> Result<ObjectVersion, DBError> {
        let path = self.inner.base_path.child(doc_path);
        let mode = if let Some(version) = version {
            PutMode::Update(version.into())
        } else {
            PutMode::Overwrite
        };
        self.inner.put(path, data, mode).await
    }

    pub fn to_writer(&self, doc_path: &str, mode: PutMode) -> SingleWriter {
        let path = self.inner.base_path.child(doc_path);
        SingleWriter {
            inner: self.inner.clone(),
            path,
            buf: Vec::new(),
            mode,
            flushing: None,
        }
    }

    pub fn stream_writer(&self, doc_path: &str) -> StreamWriter {
        let path = self.inner.base_path.child(doc_path);
        let writer = BufWriter::with_capacity(
            self.inner.object_store.clone(),
            path.clone(),
            self.inner.object_chunk_size,
        );

        let empty: BufWriter = BufWriter::new(Arc::new(NotImplementedObjectStore), path.clone());
        let level = async_compression::Level::Precise(self.inner.compress_level);
        if self.inner.compress_level > 0 {
            StreamWriter {
                inner: self.inner.clone(),
                writer: empty,
                compression: ZstdEncoder::with_quality(writer, level),
                with_compression: true,
                path,
                bytes_written_total: 0,
            }
        } else {
            StreamWriter {
                inner: self.inner.clone(),
                writer,
                compression: ZstdEncoder::new(empty),
                with_compression: false,
                path,
                bytes_written_total: 0,
            }
        }
    }

    async fn delete(&self, doc_path: &str) -> Result<(), DBError> {
        let path = self.inner.base_path.child(doc_path);

        self.inner
            .object_store
            .delete(&path)
            .await
            .map_err(DBError::from)?;

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
    ) -> BoxStream<Result<(T, ObjectVersion), DBError>>
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
            .map_err(DBError::from)
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

impl InnerStorage {
    async fn put(&self, path: Path, data: Bytes, mode: PutMode) -> Result<ObjectVersion, DBError> {
        let data = if self.compress_level > 0 {
            try_compress(data, self.compress_level)
        } else {
            data
        };

        let data_len = data.len();
        if data_len > self.max_small_object_size {
            return Err(DBError::Generic {
                name: self.base_path.to_string(),
                source: "Payload size exceeds limit, please use `stream_writer`".into(),
            });
        }

        let result = self
            .object_store
            .put_opts(
                &path,
                data.into(),
                PutOptions {
                    mode,
                    ..Default::default()
                },
            )
            .await
            .map_err(DBError::from)?;

        if let Some(cache) = &self.cache {
            cache.remove(&path).await;
        }

        self.stats.total_put_count.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_put_bytes
            .fetch_add(data_len as u64, Ordering::Relaxed);

        Ok(result.into())
    }
}

// SingleWriter 仅支持将 max_small_object_size 以下的对象写入到 object_store 中。
// 如果对象超过了这个大小，应该使用 StreamWriter 来写入。
pub struct SingleWriter {
    inner: Arc<InnerStorage>,
    path: Path,
    buf: Vec<u8>,
    mode: PutMode,
    flushing: Option<BoxFuture<'static, Result<ObjectVersion, DBError>>>,
}

impl futures::io::AsyncWrite for SingleWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.buf.extend_from_slice(buf);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(io::Write::write_vectored(&mut self.buf, bufs))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // 获取 self 的可变引用
        let this = self.as_mut().get_mut();

        // 如果没有正在进行的 flush 操作且 buffer 不为空，则创建一个
        if this.flushing.is_none() && !this.buf.is_empty() {
            let buf = std::mem::take(&mut this.buf);
            let inner = this.inner.clone();
            let path = this.path.clone();
            let mode = this.mode.clone();

            this.flushing = Some(Box::pin(
                async move { inner.put(path, buf.into(), mode).await },
            ));
        }

        // 如果有正在进行的 flush 操作，轮询它
        if let Some(fut) = &mut this.flushing {
            match fut.poll_unpin(cx) {
                Poll::Ready(Ok(_)) => {
                    this.flushing = None;
                    Poll::Ready(Ok(()))
                }
                Poll::Ready(Err(e)) => {
                    this.flushing = None;
                    Poll::Ready(Err(io::Error::new(io::ErrorKind::Other, e)))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            // 没有需要 flush 的数据
            Poll::Ready(Ok(()))
        }
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

pin_project! {
    pub struct StreamReader {
        #[pin]
        reader: BufReader,
        #[pin]
        compression: ZstdDecoder<BufReader>,
        with_compression: bool,
        path: Path,
        inner: Arc<InnerStorage>,
    }
}

impl futures::io::AsyncRead for StreamReader {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        slice: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let inner = self.inner.clone();

        let mut buf = tokio::io::ReadBuf::new(slice);
        if self.with_compression {
            core::task::ready!(tokio::io::AsyncRead::poll_read(
                self.project().compression,
                cx,
                &mut buf
            ))?;
        } else {
            core::task::ready!(tokio::io::AsyncRead::poll_read(
                self.project().reader,
                cx,
                &mut buf
            ))?;
        }

        let filled = buf.filled().len();
        inner
            .stats
            .total_fetch_count
            .fetch_add(1, Ordering::Relaxed);
        inner
            .stats
            .total_fetch_bytes
            .fetch_add(filled as u64, Ordering::Relaxed);
        Poll::Ready(Ok(filled))
    }
}

pin_project! {
    pub struct StreamWriter {
        #[pin]
        writer: BufWriter,
        #[pin]
        compression: ZstdEncoder<BufWriter>,
        with_compression: bool,
        bytes_written_total: usize,
        path: Path,
        inner: Arc<InnerStorage>,
    }
}

impl futures::io::AsyncWrite for StreamWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.as_mut().get_mut();
        this.bytes_written_total += buf.len();

        if self.with_compression {
            tokio::io::AsyncWrite::poll_write(self.project().compression, cx, buf)
        } else {
            tokio::io::AsyncWrite::poll_write(self.project().writer, cx, buf)
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.with_compression {
            tokio::io::AsyncWrite::poll_flush(self.project().compression, cx)
        } else {
            tokio::io::AsyncWrite::poll_flush(self.project().writer, cx)
        }
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let inner = self.inner.clone();
        let bytes_written = self.bytes_written_total;
        let rt = if self.with_compression {
            tokio::io::AsyncWrite::poll_shutdown(self.project().compression, cx)
        } else {
            tokio::io::AsyncWrite::poll_shutdown(self.project().writer, cx)
        };

        match rt {
            Poll::Ready(Ok(_)) => {
                inner.stats.total_put_count.fetch_add(1, Ordering::Relaxed);
                // If the writer is closed successfully, we can return the total bytes written
                inner
                    .stats
                    .total_put_bytes
                    .fetch_add(bytes_written as u64, Ordering::Relaxed);
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
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

fn try_compress(data: Bytes, compress_level: i32) -> Bytes {
    let mut buf = Vec::with_capacity(data.len() / 3);
    match compress(&mut buf, &data[..], compress_level) {
        Ok(_) => buf.into(),
        Err(_) => data,
    }
}

#[derive(Debug, Clone)]
struct NotImplementedObjectStore;

impl std::fmt::Display for NotImplementedObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NotImplementedObjectStore")
    }
}

#[async_trait]
impl ObjectStore for NotImplementedObjectStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: object_store::PutPayload,
        _opts: object_store::PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        Err(object_store::Error::NotImplemented)
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: object_store::PutMultipartOpts,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        Err(object_store::Error::NotImplemented)
    }

    async fn get_opts(
        &self,
        _location: &Path,
        _options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        Err(object_store::Error::NotImplemented)
    }

    async fn get_ranges(
        &self,
        _location: &Path,
        _ranges: &[std::ops::Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        Err(object_store::Error::NotImplemented)
    }

    /// Delete the object at the specified location.
    async fn delete(&self, _location: &Path) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented)
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let stream = futures::stream::iter(vec![]);
        stream.boxed()
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> object_store::Result<object_store::ListResult> {
        Err(object_store::Error::NotImplemented)
    }

    async fn copy(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented)
    }

    async fn copy_if_not_exists(&self, _from: &Path, _to: &Path) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented)
    }
}
