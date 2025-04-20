use anda_db_tfs::BM25Index;
use bytes::Bytes;
use std::{fmt::Debug, hash::Hash};

pub use anda_db_tfs::{
    BM25Config, BM25Error, BM25Metadata, BM25Params, BM25Stats, TokenizerChain, jieba_tokenizer,
};

use crate::{
    error::DBError,
    schema::Fe,
    storage::{PutMode, Storage},
};

pub(crate) struct BM25 {
    name: String,
    field: Fe,
    index: BM25Index<TokenizerChain>,
    storage: Storage, // 与 Collection 共享同一个 Storage 实例
}

impl Debug for BM25 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BM25Index({})", self.name)
    }
}

impl PartialEq for &BM25 {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for &BM25 {}
impl Hash for &BM25 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl BM25 {
    fn metadata_path(name: &str) -> String {
        format!("bm25_indexes/{name}/meta.cbor")
    }

    fn segment_path(name: &str, bucket: u32) -> String {
        format!("bm25_indexes/{name}/s_{bucket}.cbor")
    }

    fn posting_path(name: &str, bucket: u32) -> String {
        format!("bm25_indexes/{name}/p_{bucket}.cbor")
    }

    pub async fn new(
        name: String,
        field: Fe,
        tokenizer: TokenizerChain,
        storage: Storage,
        now_ms: u64,
    ) -> Result<Self, DBError> {
        let config = BM25Config {
            bucket_overload_size: storage.object_chunk_size() as u32 * 2,
            ..Default::default()
        };
        let path = BM25::metadata_path(&name);
        let index = BM25Index::new(name.clone(), tokenizer, Some(config));
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

    pub async fn bootstrap(
        name: String,
        field: Fe,
        tokenizer: TokenizerChain,
        storage: Storage,
    ) -> Result<Self, DBError> {
        let path = BM25::metadata_path(&name);
        let (metadata, _) = storage.fetch_raw(&path).await?;
        let index = BM25Index::load_all(
            tokenizer,
            &metadata[..],
            async |id: u32| {
                let path = BM25::segment_path(&name, id);
                let (data, _) = storage.fetch_raw(&path).await?;
                Ok(data.into())
            },
            async |id: u32| {
                let path = BM25::posting_path(&name, id);
                let (data, _) = storage.fetch_raw(&path).await?;
                Ok(data.into())
            },
        )
        .await?;

        Ok(Self {
            name,
            field,
            index,
            storage,
        })
    }

    pub async fn flush(&self, now_ms: u64) -> Result<(), DBError> {
        let path = BM25::metadata_path(&self.name);
        let mut data = Vec::new();
        self.index.store_metadata(&mut data, now_ms)?;
        self.storage
            .put_bytes(&path, data.into(), PutMode::Overwrite)
            .await?;
        self.index
            .store_dirty_segments(async |id, data| {
                let path = BM25::segment_path(&self.name, id);
                let _ = self
                    .storage
                    .put_bytes(&path, Bytes::copy_from_slice(data), PutMode::Overwrite)
                    .await?;
                Ok(true)
            })
            .await?;
        self.index
            .store_dirty_postings(async |id, data| {
                let path = BM25::posting_path(&self.name, id);
                let _ = self
                    .storage
                    .put_bytes(&path, Bytes::copy_from_slice(data), PutMode::Overwrite)
                    .await?;
                Ok(true)
            })
            .await?;

        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn field_name(&self) -> &str {
        self.field.name()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn stats(&self) -> BM25Stats {
        self.index.stats()
    }

    pub fn metadata(&self) -> BM25Metadata {
        self.index.metadata()
    }

    pub fn insert(&self, id: u64, text: &str, now_ms: u64) -> Result<(), DBError> {
        self.index.insert(id, text, now_ms)?;
        Ok(())
    }

    pub fn remove(&self, id: u64, text: &str, now_ms: u64) -> bool {
        self.index.remove(id, text, now_ms)
    }

    pub fn search(&self, query: &str, top_k: usize, params: Option<BM25Params>) -> Vec<(u64, f32)> {
        self.index.search(query, top_k, params)
    }

    pub fn search_advanced(
        &self,
        query: &str,
        top_k: usize,
        params: Option<BM25Params>,
    ) -> Vec<(u64, f32)> {
        self.index.search_advanced(query, top_k, params)
    }
}
