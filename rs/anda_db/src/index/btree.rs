use anda_db_btree::BTreeIndex;
use bytes::Bytes;
use serde::{Serialize, de::DeserializeOwned};
use std::{fmt::Debug, hash::Hash};

pub use anda_db_btree::{BTreeConfig, BTreeError, BTreeMetadata, BTreeStats, RangeQuery};

use crate::{
    error::DBError,
    schema::{Fe, Ft, Fv, Xid},
    storage::{PutMode, Storage},
};

pub(crate) enum BTree {
    U64(InnerBTree<u64>),
    I64(InnerBTree<i64>),
    String(InnerBTree<String>),
    Bytes(InnerBTree<Vec<u8>>),
}

impl Debug for BTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BTree::I64(btree) => write!(f, "BTreeIndex<I64>({})", btree.name),
            BTree::U64(btree) => write!(f, "BTreeIndex<U64>({})", btree.name),
            BTree::String(btree) => write!(f, "BTreeIndex<String>({})", btree.name),
            BTree::Bytes(btree) => write!(f, "BTreeIndex<Bytes>({})", btree.name),
        }
    }
}

impl PartialEq for &BTree {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BTree::I64(a), BTree::I64(b)) => a.name == b.name,
            (BTree::U64(a), BTree::U64(b)) => a.name == b.name,
            (BTree::String(a), BTree::String(b)) => a.name == b.name,
            (BTree::Bytes(a), BTree::Bytes(b)) => a.name == b.name,
            _ => false,
        }
    }
}

impl Eq for &BTree {}
impl Hash for &BTree {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BTree::I64(btree) => btree.name.hash(state),
            BTree::U64(btree) => btree.name.hash(state),
            BTree::String(btree) => btree.name.hash(state),
            BTree::Bytes(btree) => btree.name.hash(state),
        }
    }
}

pub(crate) struct InnerBTree<FV>
where
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
{
    name: String,
    field: Fe,
    index: BTreeIndex<Xid, FV>,
    storage: Storage, // 与 Collection 共享同一个 Storage 实例
}

impl BTree {
    fn metadata_path(name: &str) -> String {
        format!("btree_indexes/{name}/meta.cbor")
    }

    fn posting_path(name: &str, bucket: u32) -> String {
        format!("btree_indexes/{name}/p_{bucket}.cbor")
    }

    pub async fn new(
        name: String,
        field: Fe,
        storage: Storage,
        now_ms: u64,
    ) -> Result<Self, DBError> {
        let config = BTreeConfig {
            bucket_overload_size: storage.object_chunk_size() as u32 * 2,
            allow_duplicates: !field.unique(),
        };
        let btree = match *field.r#type() {
            Ft::U64 => BTree::U64(InnerBTree::new(name, field, config, storage, now_ms).await?),
            Ft::I64 => BTree::I64(InnerBTree::new(name, field, config, storage, now_ms).await?),
            Ft::Text => BTree::String(InnerBTree::new(name, field, config, storage, now_ms).await?),
            Ft::Bytes => BTree::Bytes(InnerBTree::new(name, field, config, storage, now_ms).await?),
            _ => {
                return Err(DBError::Index {
                    name,
                    source: format!("BTree: unsupported field: {:?}", field).into(),
                });
            }
        };

        Ok(btree)
    }

    pub async fn bootstrap(name: String, field: Fe, storage: Storage) -> Result<Self, DBError> {
        match field.r#type() {
            Ft::U64 => {
                let btree = InnerBTree::<u64>::bootstrap(name, field, storage).await?;
                Ok(BTree::U64(btree))
            }
            Ft::I64 => {
                let btree = InnerBTree::<i64>::bootstrap(name, field, storage).await?;
                Ok(BTree::I64(btree))
            }
            Ft::Text => {
                let btree = InnerBTree::<String>::bootstrap(name, field, storage).await?;
                Ok(BTree::String(btree))
            }
            Ft::Bytes => {
                let btree = InnerBTree::<Vec<u8>>::bootstrap(name, field, storage).await?;
                Ok(BTree::Bytes(btree))
            }
            _ => Err(DBError::Index {
                name,
                source: format!("BTree: unsupported field: {:?}", field).into(),
            }),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            BTree::I64(btree) => &btree.name,
            BTree::U64(btree) => &btree.name,
            BTree::String(btree) => &btree.name,
            BTree::Bytes(btree) => &btree.name,
        }
    }

    pub fn field_name(&self) -> &str {
        match self {
            BTree::I64(btree) => btree.field.name(),
            BTree::U64(btree) => btree.field.name(),
            BTree::String(btree) => btree.field.name(),
            BTree::Bytes(btree) => btree.field.name(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BTree::I64(btree) => btree.index.len(),
            BTree::U64(btree) => btree.index.len(),
            BTree::String(btree) => btree.index.len(),
            BTree::Bytes(btree) => btree.index.len(),
        }
    }

    pub fn stats(&self) -> BTreeStats {
        match self {
            BTree::I64(btree) => btree.index.stats(),
            BTree::U64(btree) => btree.index.stats(),
            BTree::String(btree) => btree.index.stats(),
            BTree::Bytes(btree) => btree.index.stats(),
        }
    }

    pub fn metadata(&self) -> BTreeMetadata {
        match self {
            BTree::I64(btree) => btree.index.metadata(),
            BTree::U64(btree) => btree.index.metadata(),
            BTree::String(btree) => btree.index.metadata(),
            BTree::Bytes(btree) => btree.index.metadata(),
        }
    }

    pub fn insert(&self, doc_id: &Xid, field_value: &Fv, now_ms: u64) -> Result<bool, DBError> {
        match (&self, field_value) {
            (BTree::I64(btree), Fv::I64(val)) => btree
                .index
                .insert(doc_id.clone(), *val, now_ms)
                .map_err(DBError::from),
            (BTree::U64(btree), Fv::U64(val)) => btree
                .index
                .insert(doc_id.clone(), *val, now_ms)
                .map_err(DBError::from),
            (BTree::String(btree), Fv::Text(val)) => btree
                .index
                .insert(doc_id.clone(), val.clone(), now_ms)
                .map_err(DBError::from),
            (BTree::Bytes(btree), Fv::Bytes(val)) => btree
                .index
                .insert(doc_id.clone(), val.clone(), now_ms)
                .map_err(DBError::from),
            _ => Err(DBError::Index {
                name: self.name().to_string(),
                source: format!(
                    "BTree: field value type mismatch: expected {:?}, found {:?}",
                    self.field_name(),
                    field_value
                )
                .into(),
            }),
        }
    }

    pub fn batch_insert<I>(&self, items: I, now_ms: u64) -> Result<usize, DBError>
    where
        I: IntoIterator<Item = (Xid, Fv)>,
    {
        match &self {
            BTree::I64(btree) => btree
                .index
                .batch_insert(
                    items
                        .into_iter()
                        .filter_map(|(id, val)| val.try_into().ok().map(|v| (id, v))),
                    now_ms,
                )
                .map_err(DBError::from),
            BTree::U64(btree) => btree
                .index
                .batch_insert(
                    items
                        .into_iter()
                        .filter_map(|(id, val)| val.try_into().ok().map(|v| (id, v))),
                    now_ms,
                )
                .map_err(DBError::from),
            BTree::String(btree) => btree
                .index
                .batch_insert(
                    items
                        .into_iter()
                        .filter_map(|(id, val)| val.try_into().ok().map(|v| (id, v))),
                    now_ms,
                )
                .map_err(DBError::from),
            BTree::Bytes(btree) => btree
                .index
                .batch_insert(
                    items
                        .into_iter()
                        .filter_map(|(id, val)| val.try_into().ok().map(|v| (id, v))),
                    now_ms,
                )
                .map_err(DBError::from),
        }
    }

    pub fn remove(&self, doc_id: &Xid, field_value: &Fv, now_ms: u64) -> bool {
        match (&self, field_value) {
            (BTree::I64(btree), Fv::I64(val)) => btree.index.remove(doc_id.clone(), *val, now_ms),
            (BTree::U64(btree), Fv::U64(val)) => btree.index.remove(doc_id.clone(), *val, now_ms),
            (BTree::String(btree), Fv::Text(val)) => {
                btree.index.remove(doc_id.clone(), val.clone(), now_ms)
            }
            (BTree::Bytes(btree), Fv::Bytes(val)) => {
                btree.index.remove(doc_id.clone(), val.clone(), now_ms)
            }
            _ => false,
        }
    }

    pub fn search_with<F, R>(&self, field_value: &Fv, f: F) -> Option<R>
    where
        F: FnOnce(&Vec<Xid>) -> Option<R>,
    {
        match (self, field_value) {
            (BTree::I64(btree), Fv::I64(val)) => btree.index.search_with(val, f),
            (BTree::U64(btree), Fv::U64(val)) => btree.index.search_with(val, f),
            (BTree::String(btree), Fv::Text(val)) => btree.index.search_with(val, f),
            (BTree::Bytes(btree), Fv::Bytes(val)) => btree.index.search_with(val, f),
            _ => None,
        }
    }

    pub fn search_range_with<'a, F, R>(&'a self, query: RangeQuery<'a, Fv>, mut f: F) -> Vec<R>
    where
        F: FnMut(Fv, &Vec<Xid>) -> (bool, Option<R>),
    {
        match self {
            BTree::I64(btree) => match RangeQuery::<'a, i64>::try_convert_from(query) {
                Ok(q) => btree
                    .index
                    .search_range_with(q, |fv, pks| f(Fv::I64(*fv), pks)),
                Err(_) => {
                    vec![]
                }
            },
            BTree::U64(btree) => match RangeQuery::<'a, u64>::try_convert_from(query) {
                Ok(q) => btree
                    .index
                    .search_range_with(q, |fv, pks| f(Fv::U64(*fv), pks)),
                Err(_) => {
                    vec![]
                }
            },
            BTree::String(btree) => match RangeQuery::<'a, String>::try_convert_from(query) {
                Ok(q) => btree
                    .index
                    .search_range_with(q, |fv, pks| f(Fv::Text(fv.to_owned()), pks)),
                Err(_) => {
                    vec![]
                }
            },
            BTree::Bytes(btree) => match RangeQuery::<'a, Vec<u8>>::try_convert_from(query) {
                Ok(q) => btree
                    .index
                    .search_range_with(q, |fv, pks| f(Fv::Bytes(fv.clone()), pks)),
                Err(_) => {
                    vec![]
                }
            },
        }
    }

    pub async fn flush(&self, now_ms: u64) -> Result<(), DBError> {
        match self {
            BTree::I64(btree) => btree.flush(now_ms).await,
            BTree::U64(btree) => btree.flush(now_ms).await,
            BTree::String(btree) => btree.flush(now_ms).await,
            BTree::Bytes(btree) => btree.flush(now_ms).await,
        }
    }
}

impl<FV> InnerBTree<FV>
where
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
{
    async fn new(
        name: String,
        field: Fe,
        config: BTreeConfig,
        storage: Storage,
        now_ms: u64,
    ) -> Result<Self, DBError> {
        let path = BTree::metadata_path(&name);
        let index = BTreeIndex::new(name.clone(), Some(config));
        let mut data = Vec::new();
        index.store_metadata(&mut data, now_ms)?;
        storage
            .put_bytes(&path, data.into(), PutMode::Create)
            .await?;
        Ok(InnerBTree {
            name,
            field,
            index,
            storage,
        })
    }

    async fn bootstrap(name: String, field: Fe, storage: Storage) -> Result<Self, DBError> {
        let path = BTree::metadata_path(&name);
        let (metadata, _) = storage.fetch_raw(&path).await?;
        let index = BTreeIndex::<Xid, FV>::load_all(&metadata[..], async |id: u32| {
            let path = BTree::posting_path(&name, id);
            let (data, _) = storage.fetch_raw(&path).await?;
            Ok(data.into())
        })
        .await?;

        Ok(Self {
            name,
            field,
            index,
            storage,
        })
    }

    async fn flush(&self, now_ms: u64) -> Result<(), DBError> {
        let path = BTree::metadata_path(&self.name);
        let mut data = Vec::new();
        self.index.store_metadata(&mut data, now_ms)?;
        self.storage
            .put_bytes(&path, data.into(), PutMode::Overwrite)
            .await?;
        self.index
            .store_dirty_postings(async |id, data| {
                let path = BTree::posting_path(&self.name, id);
                let _ = self
                    .storage
                    .put_bytes(&path, Bytes::copy_from_slice(data), PutMode::Overwrite)
                    .await?;
                Ok(true)
            })
            .await?;

        Ok(())
    }
}
