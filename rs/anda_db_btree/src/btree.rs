//! # Anda-DB B-tree Index Library
//!
//! This module provides a B-tree based index implementation for Anda-DB.
//! It supports indexing fields of various types including u64, i64, String, and binary data.
//! The implementation is optimized for concurrent access and efficient range queries.
//!
//! ## Features
//! - Thread-safe concurrent access
//! - Efficient range queries
//! - Support for various data types
//! - Bucket-based storage for better incremental persistent
//! - Efficient serialization and deserialization in CBOR format

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    io::{Read, Write},
    sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering},
};

use crate::{BTreeError, BoxError};

/// B-tree index for efficient key-value lookups
///
/// This structure provides a thread-safe B-tree index implementation
/// that supports concurrent reads and writes, as well as efficient range queries.
pub struct BTreeIndex<PK, FV>
where
    PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
{
    /// Index name
    name: String,

    /// Index configuration
    config: BTreeConfig,

    /// Buckets store information about where posting entries are stored and their current state
    /// The mapping is: bucket_id -> (bucket_size, is_dirty, vec<field_values>)
    /// - bucket_size: Current size of the bucket in bytes
    /// - is_dirty: Indicates if the bucket has new data that needs to be persisted
    /// - field_values: List of field values stored in this bucket
    buckets: DashMap<u32, (u32, bool, Vec<FV>)>,

    /// Inverted index mapping field values to posting values
    postings: DashMap<FV, PostingValue<PK>>,

    /// B-tree set for efficient range queries
    btree: RwLock<BTreeSet<FV>>,

    /// Index metadata
    metadata: RwLock<BTreeMetadata>,

    /// Maximum bucket ID currently in use
    max_bucket_id: AtomicU32,

    /// Maximum size of a bucket before creating a new one
    /// When a bucket's stored data exceeds this size,
    /// a new bucket should be created for new data
    bucket_overload_size: u32,

    /// Number of search operations performed
    search_count: AtomicU64,
}

/// Type alias for posting values: (bucket id, update version, Vec<document id>)
/// - bucket_id: The bucket where this posting is stored
/// - update_version: Version number that increases with each update
/// - document_ids: List of document IDs associated with this field value
type PostingValue<PK> = (u32, u64, Vec<PK>);

/// Configuration parameters for the B-tree index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeConfig {
    /// Maximum size of a bucket before creating a new one (in bytes)
    pub bucket_overload_size: u32,

    /// Whether to allow duplicate keys
    /// If false, attempting to insert a duplicate key will result in an error
    pub allow_duplicates: bool,
}

impl Default for BTreeConfig {
    fn default() -> Self {
        BTreeConfig {
            bucket_overload_size: 1024 * 512,
            allow_duplicates: true,
        }
    }
}

/// Index metadata containing configuration and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeMetadata {
    /// Index name
    pub name: String,

    /// Index configuration
    pub config: BTreeConfig,

    /// Index statistics
    pub stats: BTreeStats,
}

/// Index statistics for monitoring and diagnostics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BTreeStats {
    /// Last insertion timestamp (unix ms)
    pub last_inserted: u64,

    /// Last deletion timestamp (unix ms)
    pub last_deleted: u64,

    /// Last saved timestamp (unix ms)
    pub last_saved: u64,

    /// Updated version for the index. It will be incremented when the index is updated.
    pub version: u64,

    /// Number of elements in the index
    pub num_elements: u64,

    /// Number of search operations performed
    pub search_count: u64,

    /// Number of insert operations performed
    pub insert_count: u64,

    /// Number of delete operations performed
    pub delete_count: u64,

    /// Maximum bucket ID currently in use
    pub max_bucket_id: u32,
}

/// Serializable B-tree index structure (owned version)
// #[derive(Debug, Clone, Serialize, Deserialize)]
// struct BTreeIndexOwned<PK, FV>
// where
//     FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
//     PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
// {
//     // #[serde(skip)]
//     postings: DashMap<String, PostingValue<PK>>,
//     metadata: BTreeMetadata,
// }

// Helper structure for serialization and deserialization of index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BTreeIndexOwnedHelper {
    // Serialized postings map (not used for actual data, just for structure)
    postings: HashMap<String, String>,

    // Index metadata
    metadata: BTreeMetadata,
}

// Reference structure for serializing the index
#[derive(Serialize)]
struct BTreeIndexRef<'a, PK, FV>
where
    PK: Ord + Debug + Clone + Serialize,
    FV: Eq + Hash + Debug + Clone + Serialize,
{
    postings: &'a DashMap<FV, PostingValue<PK>>,
    metadata: &'a BTreeMetadata,
}

/// Range query specification for flexible querying
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeQuery<FV> {
    /// Equal to a specific key
    Eq(FV),

    /// Greater than a specific key
    Gt(FV),

    /// Greater than or equal to a specific key
    Ge(FV),

    /// Less than a specific key
    Lt(FV),

    /// Less than or equal to a specific key
    Le(FV),

    /// Between two keys (inclusive)
    Between(FV, FV),

    /// Include specific keys
    Include(Vec<FV>),

    /// A logical OR query that requires at least one subquery to match
    Or(Vec<Box<RangeQuery<FV>>>),

    /// A logical AND query that requires all subqueries to match
    And(Vec<Box<RangeQuery<FV>>>),

    /// A logical NOT query that negates the result of its subquery
    Not(Box<RangeQuery<FV>>),
}

impl<FV> RangeQuery<FV> {
    pub fn try_convert_from<FV1>(value: RangeQuery<FV1>) -> Result<Self, BoxError>
    where
        FV: Ord,
        FV: TryFrom<FV1, Error = BoxError>,
    {
        match value {
            RangeQuery::Eq(key) => Ok(RangeQuery::Eq(key.try_into()?)),
            RangeQuery::Gt(key) => Ok(RangeQuery::Gt(key.try_into()?)),
            RangeQuery::Ge(key) => Ok(RangeQuery::Ge(key.try_into()?)),
            RangeQuery::Lt(key) => Ok(RangeQuery::Lt(key.try_into()?)),
            RangeQuery::Le(key) => Ok(RangeQuery::Le(key.try_into()?)),
            RangeQuery::Between(start_key, end_key) => Ok(RangeQuery::Between(
                start_key.try_into()?,
                end_key.try_into()?,
            )),
            RangeQuery::Include(keys) => {
                let converted_keys = keys
                    .into_iter()
                    .map(|key| key.try_into())
                    .collect::<Result<Vec<FV>, _>>()?;
                Ok(RangeQuery::Include(converted_keys))
            }
            RangeQuery::And(queries) => {
                let converted_queries = queries
                    .into_iter()
                    .map(|query| RangeQuery::try_convert_from(*query))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(RangeQuery::And(
                    converted_queries.into_iter().map(Box::new).collect(),
                ))
            }
            RangeQuery::Or(queries) => {
                let converted_queries = queries
                    .into_iter()
                    .map(|query| RangeQuery::try_convert_from(*query))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(RangeQuery::Or(
                    converted_queries.into_iter().map(Box::new).collect(),
                ))
            }
            RangeQuery::Not(query) => {
                let converted_query = RangeQuery::try_convert_from(*query)?;
                Ok(RangeQuery::Not(Box::new(converted_query)))
            }
        }
    }
}

impl<PK, FV> BTreeIndex<PK, FV>
where
    PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
{
    /// Creates a new empty B-tree index with the given configuration
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the index
    /// * `config` - Optional B-tree configuration parameters
    ///
    /// # Returns
    ///
    /// * `BTreeIndex` - A new instance of the B-tree index
    pub fn new(name: String, config: Option<BTreeConfig>) -> Self {
        let config = config.unwrap_or_default();
        let bucket_overload_size = config.bucket_overload_size;
        BTreeIndex {
            name: name.clone(),
            config: config.clone(),
            postings: DashMap::new(),
            buckets: DashMap::from_iter(vec![(0, (0, true, Vec::new()))]),
            btree: RwLock::new(BTreeSet::new()),
            metadata: RwLock::new(BTreeMetadata {
                name,
                config,
                stats: BTreeStats::default(),
            }),
            bucket_overload_size,
            max_bucket_id: AtomicU32::new(0),
            search_count: AtomicU64::new(0),
        }
    }

    /// Loads an index from metadata reader and a closure for loading buckets.
    ///
    /// # Arguments
    ///
    /// * `metadata` - Metadata reader
    /// * `f` - Closure for loading buckets
    ///
    /// # Returns
    ///
    /// * `Result<Self, BTreeError>` - Loaded index or error.
    pub async fn load_all<R: Read, F>(metadata: R, f: F) -> Result<Self, BTreeError>
    where
        F: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
    {
        let mut index = Self::load_metadata(metadata)?;
        index.load_postings(f).await?;
        Ok(index)
    }

    /// Loads an index from a reader
    /// This only loads metadata, you need to call [`Self::load_buckets`] to load the actual posting data
    ///
    /// # Arguments
    ///
    /// * `r` - Any type implementing the [`Read`] trait
    ///
    /// # Returns
    ///
    /// * `Result<Self, Error>` - Loaded index or error
    pub fn load_metadata<R: Read>(r: R) -> Result<Self, BTreeError> {
        // Deserialize the index metadata
        let index: BTreeIndexOwnedHelper =
            ciborium::from_reader(r).map_err(|err| BTreeError::Serialization {
                name: "unknown".to_string(),
                source: err.into(),
            })?;

        // Extract configuration values
        let bucket_overload_size = index.metadata.config.bucket_overload_size;
        let max_bucket_id = AtomicU32::new(index.metadata.stats.max_bucket_id);
        let search_count = AtomicU64::new(index.metadata.stats.search_count);
        Ok(BTreeIndex {
            name: index.metadata.name.clone(),
            config: index.metadata.config.clone(),
            postings: DashMap::with_capacity(index.metadata.stats.num_elements as usize),
            buckets: DashMap::from_iter(vec![(0, (0, true, Vec::new()))]),
            btree: RwLock::new(BTreeSet::new()),
            metadata: RwLock::new(index.metadata),
            bucket_overload_size,
            search_count,
            max_bucket_id,
        })
    }

    /// Loads posting data from buckets using the provided async function
    /// This function should be called during database startup to load all bucket posting data
    /// and form a complete posting index
    ///
    /// # Arguments
    ///
    /// * `f` - Async function that reads posting data from a specified bucket.
    ///   `F: AsyncFn(u32) -> Result<Vec<u8>, BTreeError>`
    ///   The function should take a bucket ID as input and return a vector of bytes
    ///   containing the serialized posting data.
    ///
    /// # Returns
    ///
    /// * `Result<(), BTreeError>` - Success or error
    pub async fn load_postings<F>(&mut self, mut f: F) -> Result<(), BTreeError>
    where
        F: AsyncFnMut(u32) -> Result<Option<Vec<u8>>, BoxError>,
    {
        for i in 0..=self.max_bucket_id.load(AtomicOrdering::Relaxed) {
            let data = f(i).await.map_err(|err| BTreeError::Generic {
                name: self.name.clone(),
                source: err,
            })?;
            if let Some(data) = data {
                let postings: HashMap<FV, PostingValue<PK>> = ciborium::from_reader(&data[..])
                    .map_err(|err| BTreeError::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    })?;
                let bks = postings.keys().cloned().collect::<Vec<_>>();
                self.btree.write().extend(bks.iter().cloned());
                // Update bucket information
                // Larger buckets have the most recent state and can override smaller buckets
                self.buckets.insert(i, (data.len() as u32, false, bks));
                self.postings.extend(postings);
            }
        }

        Ok(())
    }

    /// Returns the number of keys in the index
    pub fn len(&self) -> usize {
        self.postings.len()
    }

    /// Returns whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.postings.is_empty()
    }

    /// Returns the index name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the index metadata
    /// This includes up-to-date statistics about the index
    pub fn metadata(&self) -> BTreeMetadata {
        let mut metadata = self.metadata.read().clone();
        metadata.stats.num_elements = self.postings.len() as u64;
        metadata.stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        metadata.stats.max_bucket_id = self.max_bucket_id.load(AtomicOrdering::Relaxed);
        metadata
    }

    /// Gets current statistics about the index
    pub fn stats(&self) -> BTreeStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.num_elements = self.postings.len() as u64;
        stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        stats.max_bucket_id = self.max_bucket_id.load(AtomicOrdering::Relaxed);
        stats
    }

    /// Inserts a document_id-field_value pair to the index
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Document identifier
    /// * `field_value` - Key to index
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` if the document_id-field_value pair was successfully added
    /// * `Err(BTreeError)` if failed
    pub fn insert(&self, doc_id: PK, field_value: FV, now_ms: u64) -> Result<bool, BTreeError> {
        let bucket = self.max_bucket_id.load(AtomicOrdering::Acquire);

        // Calculate the size increase for this insertion
        let mut is_new = false;
        let mut size_increase = 0;
        match self.postings.entry(field_value.clone()) {
            dashmap::Entry::Occupied(mut entry) => {
                // Check if duplicate keys are allowed
                if !self.config.allow_duplicates {
                    return Err(BTreeError::AlreadyExists {
                        name: self.name.clone(),
                        id: format!("{:?}", doc_id),
                        value: format!("{:?}", field_value),
                    });
                }

                let posting = entry.get_mut();
                // Add segment_id if it doesn't exist
                if !posting.2.contains(&doc_id) {
                    size_increase = CountingWriter::count_cbor(&doc_id) as u32 + 2;
                    posting.2.push(doc_id);
                    posting.1 += 1; // increment version
                }
            }
            dashmap::Entry::Vacant(entry) => {
                // Create a new posting for this field value
                let posting = (bucket, 1, vec![doc_id]);
                size_increase = CountingWriter::count_cbor(&posting) as u32 + 2;
                entry.insert(posting);
                is_new = true;
            }
        };

        if is_new {
            // Add the field value to the B-tree for range queries
            self.btree.write().insert(field_value.clone());
        }

        // If the index was modified, update bucket state
        let mut new_bucket = 0;
        if size_increase > 0 {
            // Update bucket state
            let mut b = self
                .buckets
                .get_mut(&bucket)
                .ok_or_else(|| BTreeError::Generic {
                    name: self.name.clone(),
                    source: format!("bucket {bucket} not found").into(),
                })?;

            // Check if the bucket has enough space
            if b.2.is_empty() || b.0 + size_increase < self.bucket_overload_size {
                b.0 += size_increase;
                // Mark as dirty, needs to be persisted
                b.1 = true;
                // Add field value to bucket if not already present
                if !b.2.contains(&field_value) {
                    b.2.push(field_value.clone());
                }
            } else {
                // If the current bucket is full, create a new one
                let mut size_decrease = 0;
                new_bucket = self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;
                {
                    if let Some(mut posting) = self.postings.get_mut(&field_value) {
                        // Update the posting's bucket ID
                        posting.0 = new_bucket;
                        size_decrease = CountingWriter::count_cbor(&posting) as u32 + 2;
                    }
                }
                // Remove the current field value from the current bucket
                // The freed space can still accommodate small growth in other field values
                if let Some(pos) = b.2.iter().position(|k| &field_value == k) {
                    b.0 = b.0.saturating_sub(size_decrease);
                    // b.1 = true; // do not need to set dirty
                    b.2.swap_remove(pos);
                }
            }
        }

        if new_bucket > 0 {
            // Create a new bucket and migrate this data to it
            match self.buckets.entry(new_bucket) {
                dashmap::Entry::Vacant(entry) => {
                    // Create a new bucket with the initial size
                    entry.insert((size_increase, true, vec![field_value]));
                }
                dashmap::Entry::Occupied(mut entry) => {
                    let bucket_entry = entry.get_mut();
                    bucket_entry.0 += size_increase;
                    bucket_entry.1 = true; // Mark as dirty
                    if !bucket_entry.2.contains(&field_value) {
                        bucket_entry.2.push(field_value);
                    }
                }
            }
        }

        self.update_metadata(|m| {
            m.stats.version += 1;
            m.stats.last_inserted = now_ms;
            m.stats.insert_count += 1;
        });

        Ok(size_increase > 0)
    }

    /// Batch inserts multiple document_id-field_value pairs into the index
    ///
    /// This method is optimized for inserting multiple items at once by grouping
    /// updates by field value and bucket, reducing lock contention and improving performance.
    /// Duplicate keys will be skipped if `allow_duplicates` is set to `false`.
    ///
    /// # Arguments
    ///
    /// * `items` - Iterator of (document_id, field_value) pairs to insert
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<usize, BTreeError>` - Success or error
    pub fn batch_insert<I>(&self, items: I, now_ms: u64) -> Result<usize, BTreeError>
    where
        I: IntoIterator<Item = (PK, FV)>,
    {
        // Group batch data by different field values
        let mut grouped_items: BTreeMap<FV, Vec<PK>> = BTreeMap::new();
        for (doc_id, field_value) in items {
            grouped_items.entry(field_value).or_default().push(doc_id);
        }

        if grouped_items.is_empty() {
            return Ok(0);
        }
        let insert_count = grouped_items.len() as u64;

        // Batch processing work, optimized by grouping writes by bucket
        // Phase 1: Update the postings collection
        let mut bucket_updates: HashMap<u32, (u32, Vec<FV>)> = HashMap::new();
        let mut new_btree_values: BTreeSet<FV> = BTreeSet::new();

        for (field_value, doc_ids) in grouped_items {
            // Consider concurrent scenarios where buckets might be modified by other threads
            let bucket = self.max_bucket_id.load(AtomicOrdering::Relaxed);
            let mut size_increase = 0;
            let mut inserted_to_bucket = bucket;

            match self.postings.entry(field_value.clone()) {
                dashmap::Entry::Occupied(mut entry) => {
                    // Check if duplicate keys are allowed
                    if !self.config.allow_duplicates && !doc_ids.is_empty() {
                        // Skip duplicate keys
                        continue;
                    }

                    let posting = entry.get_mut();
                    // Use HashSet to avoid adding duplicate doc_ids
                    let mut new_ids = Vec::new();
                    for doc_id in doc_ids {
                        if !posting.2.contains(&doc_id) {
                            new_ids.push(doc_id);
                        }
                    }

                    if !new_ids.is_empty() {
                        size_increase = new_ids
                            .iter()
                            .fold(0, |acc, id| acc + CountingWriter::count_cbor(id) as u32 + 2);
                        posting.2.extend(new_ids);
                        posting.1 += 1; // increment version
                    }
                    inserted_to_bucket = posting.0;
                }
                dashmap::Entry::Vacant(entry) => {
                    let posting = (bucket, 1, doc_ids);
                    size_increase = CountingWriter::count_cbor(&posting) as u32 + 2;
                    entry.insert(posting);
                    new_btree_values.insert(field_value.clone());
                }
            };

            // If new content was added
            if size_increase > 0 {
                // Collect updates by bucket for batch processing later
                let entry = bucket_updates
                    .entry(inserted_to_bucket)
                    .or_insert((0, Vec::new()));

                entry.0 += size_increase;
                if !entry.1.contains(&field_value) {
                    entry.1.push(field_value.clone());
                }
            }
        }

        if !new_btree_values.is_empty() {
            self.btree.write().append(&mut new_btree_values);
        }

        // Phase 2: Update bucket states
        // field_values_to_migrate: (old_bucket_id, field_value, size)
        let mut field_values_to_migrate: Vec<(u32, FV, u32)> = Vec::new();
        for (bucket_id, (size_increase, field_values)) in bucket_updates {
            if let Some(mut bucket_entry) = self.buckets.get_mut(&bucket_id) {
                // Check if the bucket would overflow
                if bucket_entry.2.is_empty()
                    || bucket_entry.0 + size_increase < self.bucket_overload_size
                {
                    // Bucket has enough space, update directly
                    bucket_entry.0 += size_increase;
                    bucket_entry.1 = true; // Mark as dirty

                    // Update field values contained in the bucket
                    for fv in field_values {
                        if !bucket_entry.2.contains(&fv) {
                            bucket_entry.2.push(fv);
                        }
                    }
                } else {
                    // Bucket doesn't have enough space, need to migrate these values to a new bucket
                    for fv in field_values {
                        // 获取需要迁移的 field_value 大小
                        let size = if let Some(posting) = self.postings.get(&fv) {
                            CountingWriter::count_cbor(&posting) as u32 + 2
                        } else {
                            0
                        };

                        if size > 0 {
                            field_values_to_migrate.push((bucket_id, fv, size));
                        }
                    }
                }
            }
        }

        // Phase 3: Create new buckets if needed
        if !field_values_to_migrate.is_empty() {
            let mut next_bucket_id = self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;

            {
                self.buckets
                    .entry(next_bucket_id)
                    .or_insert_with(|| (0, true, Vec::new()));
                // release the lock on the entry
            }

            for (old_bucket_id, field_value, size) in field_values_to_migrate {
                if let Some(mut posting) = self.postings.get_mut(&field_value) {
                    posting.0 = next_bucket_id;
                }

                if let Some(mut ob) = self.buckets.get_mut(&old_bucket_id) {
                    if let Some(pos) = ob.2.iter().position(|k| &field_value == k) {
                        ob.0 = ob.0.saturating_sub(size);
                        // ob.1 = true; // do not need to set dirty
                        ob.2.swap_remove(pos);
                    }
                }

                let mut new_bucket = false;
                if let Some(mut nb) = self.buckets.get_mut(&next_bucket_id) {
                    if nb.2.is_empty() || nb.0 + size < self.bucket_overload_size {
                        // Bucket has enough space, update directly
                        nb.0 += size;
                        if !nb.2.contains(&field_value) {
                            nb.2.push(field_value.clone());
                        }
                    } else {
                        // Bucket doesn't have enough space, need to migrate to the next bucket
                        new_bucket = true;
                    }
                }

                if new_bucket {
                    next_bucket_id = self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;
                    // update the posting's bucket_id again
                    if let Some(mut posting) = self.postings.get_mut(&field_value) {
                        posting.0 = next_bucket_id;
                    }

                    match self.buckets.entry(next_bucket_id) {
                        dashmap::Entry::Vacant(entry) => {
                            // Create a new bucket with the initial size
                            entry.insert((size, true, vec![field_value]));
                        }
                        dashmap::Entry::Occupied(mut entry) => {
                            let bucket_entry = entry.get_mut();
                            bucket_entry.0 += size;
                            bucket_entry.1 = true; // Mark as dirty
                            if !bucket_entry.2.contains(&field_value) {
                                bucket_entry.2.push(field_value);
                            }
                        }
                    }
                }
            }
        }

        // Update index metadata
        self.update_metadata(|m| {
            m.stats.version += 1;
            m.stats.last_inserted = now_ms;
            m.stats.insert_count += insert_count;
        });

        Ok(insert_count as usize)
    }

    /// Removes a document_id-field_value pair from the index with hook function
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Document identifier
    /// * `field_value` - field to remove
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the document_id-field_value pair was successfully removed, `false` otherwise
    pub fn remove(&self, doc_id: PK, field_value: FV, now_ms: u64) -> bool {
        let mut removed = false;
        let mut size_decrease = 0;
        let mut posting_empty = false;
        let mut bucket_id = 0;

        {
            if let Some(mut posting) = self.postings.get_mut(&field_value) {
                removed = true;
                bucket_id = posting.0;
                if let Some(pos) = posting.2.iter().position(|id| id == &doc_id) {
                    size_decrease = if posting.2.len() > 1 {
                        CountingWriter::count_cbor(&doc_id) as u32 + 2
                    } else {
                        CountingWriter::count_cbor(&posting) as u32 + 2
                    };
                    posting.1 += 1; // increment version
                    posting.2.swap_remove(pos);
                    posting_empty = posting.2.is_empty();
                }
            }
        }

        if removed {
            // Update the bucket state
            if let Some(mut b) = self.buckets.get_mut(&bucket_id) {
                b.0 = b.0.saturating_sub(size_decrease);
                b.1 = true;

                if posting_empty {
                    // remove FV from the bucket
                    if let Some(pos) = b.2.iter().position(|k| &field_value == k) {
                        b.2.swap_remove(pos);
                    }
                }
            }

            if posting_empty {
                self.btree.write().remove(&field_value);
                self.postings.remove(&field_value);
            }

            self.update_metadata(|m| {
                m.stats.version += 1;
                m.stats.last_deleted = now_ms;
                m.stats.delete_count += 1;
            });
        }

        removed
    }

    /// Searches the index for an exact key match
    ///
    /// # Arguments
    ///
    /// * `field_value` - Key to search for
    /// * `f` - Function to apply to the posting value
    ///
    /// # Returns
    ///
    /// * `Option<R>` - Result of the function applied to the posting value
    pub fn search_with<F, R>(&self, field_value: &FV, f: F) -> Option<R>
    where
        F: FnOnce(&Vec<PK>) -> Option<R>,
    {
        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);

        self.postings
            .get(field_value)
            .and_then(|posting| f(&posting.2))
    }

    /// Searches the index using a range query
    ///
    /// # Arguments
    ///
    /// * `query` - Range query specification
    /// * `f` - Function to apply to the posting value. The function should return a tuple
    ///   containing a boolean indicating if the search should continue and an optional result.
    ///
    /// # Returns
    ///
    /// * `Vec<R>` - Vector of results from the function applied to the posting values
    pub fn search_range_with<F, R>(&self, query: RangeQuery<FV>, mut f: F) -> Vec<R>
    where
        F: FnMut(&FV, &Vec<PK>) -> (bool, Vec<R>),
    {
        let mut results = Vec::new();
        if self.postings.is_empty() {
            return results;
        }

        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);

        match query {
            RangeQuery::Eq(key) => {
                if let Some(posting) = self.postings.get(&key) {
                    let (conti, rt) = f(&key, &posting.2);
                    results.extend(rt);
                    if !conti {
                        return results;
                    }
                }
            }
            RangeQuery::Gt(start_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeFrom {
                        start: start_key.clone(),
                    })
                    .filter(|&k| k > &start_key)
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Ge(start_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Lt(end_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeTo { end: end_key })
                    .rev()
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Le(end_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeToInclusive { end: end_key })
                    .rev()
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Between(start_key, end_key) => {
                for k in self.btree.read().range(start_key..=end_key) {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Include(keys) => {
                for k in keys.into_iter() {
                    if let Some(posting) = self.postings.get(&k) {
                        let (conti, rt) = f(&k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::And(queries) => {
                // 先找出最小结果集的子查询，减少交集计算量
                let keys = self.range_keys(RangeQuery::And(queries));
                for k in keys {
                    if let Some(posting) = self.postings.get(&k) {
                        let (conti, rt) = f(&k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Or(queries) => {
                let keys = self.range_keys(RangeQuery::Or(queries));
                for k in keys {
                    if let Some(posting) = self.postings.get(&k) {
                        let (conti, rt) = f(&k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Not(query) => {
                // 先收集要排除的 key，再遍历全集差集
                let exclude: BTreeSet<FV> = self.range_keys(*query).into_iter().collect();

                for k in self.btree.read().iter() {
                    if exclude.contains(k) {
                        continue;
                    }
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        results.extend(rt);
                        if !conti {
                            return results;
                        }
                    }
                }
            }
        }

        results
    }

    fn range_keys(&self, query: RangeQuery<FV>) -> Vec<FV> {
        let mut results: Vec<FV> = Vec::new();

        match query {
            RangeQuery::Eq(key) => {
                if self.btree.read().contains(&key) {
                    results.push(key);
                }
            }
            RangeQuery::Gt(start_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeFrom {
                        start: start_key.clone(),
                    })
                    .filter(|&k| k > &start_key)
                {
                    results.push(k.clone());
                }
            }
            RangeQuery::Ge(start_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeFrom { start: start_key })
                {
                    results.push(k.clone());
                }
            }
            RangeQuery::Lt(end_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeTo { end: end_key })
                    .rev()
                {
                    results.push(k.clone());
                }
            }
            RangeQuery::Le(end_key) => {
                for k in self
                    .btree
                    .read()
                    .range(std::ops::RangeToInclusive { end: end_key })
                    .rev()
                {
                    results.push(k.clone());
                }
            }
            RangeQuery::Between(start_key, end_key) => {
                for k in self.btree.read().range(start_key..=end_key) {
                    results.push(k.clone());
                }
            }
            RangeQuery::Include(keys) => {
                let btree = self.btree.read();
                for k in keys.into_iter() {
                    if btree.contains(&k) {
                        results.push(k.clone());
                    }
                }
            }
            RangeQuery::And(queries) => {
                let mut iter = queries.into_iter();
                if let Some(query) = iter.next() {
                    let mut intersection: BTreeSet<FV> =
                        self.range_keys(*query).into_iter().collect();

                    for query in iter {
                        let keys: BTreeSet<FV> = self.range_keys(*query).into_iter().collect();
                        intersection = intersection
                            .intersection(&keys)
                            .cloned()
                            .collect::<BTreeSet<_>>();
                        if intersection.is_empty() {
                            return vec![];
                        }
                    }

                    results.extend(intersection);
                }
            }
            RangeQuery::Or(queries) => {
                let mut seen = HashSet::new();
                for query in queries {
                    let keys = self.range_keys(*query);
                    for k in keys {
                        if seen.insert(k.clone()) {
                            results.push(k);
                        }
                    }
                }
            }
            RangeQuery::Not(query) => {
                let exclude: Vec<FV> = self.range_keys(*query);
                for k in self.btree.read().iter() {
                    if !exclude.contains(k) {
                        results.push(k.clone());
                    }
                }
            }
        }

        results
    }

    /// Stores the index metadata and dirty buckets to persistent storage.
    pub async fn flush<W: Write, F>(&self, metadata: W, now_ms: u64, f: F) -> Result<(), BTreeError>
    where
        F: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
    {
        self.store_metadata(metadata, now_ms)?;
        self.store_dirty_postings(f).await?;
        Ok(())
    }

    /// Stores the index metadata to a writer
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`Write`] trait
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - Success or error
    pub fn store_metadata<W: Write>(&self, w: W, now_ms: u64) -> Result<(), BTreeError> {
        let mut meta = self.metadata();
        meta.stats.last_saved = now_ms.max(meta.stats.last_saved);
        let postings: DashMap<FV, PostingValue<PK>> = DashMap::new();
        ciborium::into_writer(
            &BTreeIndexRef {
                postings: &postings,
                metadata: &meta,
            },
            w,
        )
        .map_err(|err| BTreeError::Serialization {
            name: self.name.clone(),
            source: err.into(),
        })?;

        self.update_metadata(|m| {
            m.stats.last_saved = meta.stats.last_saved.max(m.stats.last_saved);
        });

        Ok(())
    }

    /// Stores dirty postings to persistent storage using the provided async function
    ///
    /// This method iterates through all buckets and persists those that have been modified
    /// since the last save operation.
    ///
    /// # Arguments
    ///
    /// * `f` - Async function that writes bucket data to persistent storage
    ///   The function takes a bucket ID and serialized data, and returns whether to continue
    ///
    /// # Returns
    ///
    /// * `Result<(), BTreeError>` - Success or error
    pub async fn store_dirty_postings<F>(&self, mut f: F) -> Result<(), BTreeError>
    where
        F: AsyncFnMut(u32, &[u8]) -> Result<bool, BoxError>,
    {
        let mut buf = Vec::with_capacity(8192);
        for mut bucket in self.buckets.iter_mut() {
            if bucket.1 {
                // If the bucket is dirty, it needs to be persisted
                let mut postings: HashMap<&FV, ciborium::Value> =
                    HashMap::with_capacity(bucket.2.len());
                for k in bucket.2.iter() {
                    if let Some(posting) = self.postings.get(k) {
                        postings.insert(
                            k,
                            ciborium::cbor!(posting).map_err(|err| BTreeError::Serialization {
                                name: self.name.clone(),
                                source: err.into(),
                            })?,
                        );
                    }
                }

                buf.clear();
                ciborium::into_writer(&postings, &mut buf).map_err(|err| {
                    BTreeError::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    }
                })?;

                if let Ok(conti) = f(*bucket.key(), &buf).await {
                    // Only mark as clean if persistence was successful, otherwise wait for next round
                    bucket.1 = false;
                    if !conti {
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Updates the index metadata
    ///
    /// # Arguments
    ///
    /// * `f` - Function that modifies the metadata
    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut BTreeMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

impl<PK> BTreeIndex<PK, String>
where
    PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
{
    /// Specialized version of prefix search for String type
    /// Searches the index using a prefix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Prefix to search for
    /// * `f` - Function to apply to the posting value. The function should return a tuple
    ///   containing a boolean indicating if the search should continue and an optional result.
    ///
    /// # Returns
    /// * `Vec<R>` - Vector of results from the function applied to the posting values
    pub fn search_prefix_with<F, R>(&self, prefix: &str, mut f: F) -> Vec<R>
    where
        F: FnMut(&str, &Vec<PK>) -> (bool, Option<R>),
    {
        let mut results = Vec::new();
        if self.postings.is_empty() {
            return results;
        }

        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);
        // Use prefix search
        for k in self
            .btree
            .read()
            .range(prefix.to_string()..)
            .take_while(|k| k.starts_with(prefix))
        {
            if let Some(posting) = self.postings.get(k) {
                let (con, rt) = f(k, &posting.2);
                if let Some(r) = rt {
                    results.push(r);
                }
                if !con {
                    return results;
                }
            }
        }

        results
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
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::sync::{Barrier, Mutex};

    // 获取当前时间戳（毫秒）
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    // 辅助函数：创建一个测试用的 B-tree 索引
    fn create_test_index() -> BTreeIndex<u64, String> {
        let config = BTreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: true,
        };
        BTreeIndex::new("test_index".to_string(), Some(config))
    }

    // 辅助函数：创建一个测试用的 B-tree 索引并插入一些数据
    fn create_populated_index() -> BTreeIndex<u64, String> {
        let index = create_test_index();

        // 插入一些测试数据
        let _ = index.insert(1, "apple".to_string(), now_ms());
        let _ = index.insert(2, "banana".to_string(), now_ms());
        let _ = index.insert(3, "cherry".to_string(), now_ms());
        let _ = index.insert(4, "date".to_string(), now_ms());
        let _ = index.insert(5, "eggplant".to_string(), now_ms());

        // 测试重复键
        let _ = index.insert(6, "apple".to_string(), now_ms());
        let _ = index.insert(7, "banana".to_string(), now_ms());

        index
    }

    #[test]
    fn test_create_index() {
        let index = create_test_index();

        assert_eq!(index.name(), "test_index");
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let metadata = index.metadata();
        assert_eq!(metadata.name, "test_index");
        assert_eq!(metadata.stats.num_elements, 0);
    }

    #[test]
    fn test_insert() {
        let index = create_test_index();

        // 测试插入
        let result = index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());
        assert!(result.unwrap());

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());

        // 测试重复插入相同的文档ID和字段值
        let result = index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());
        assert!(!result.unwrap()); // 应该返回 false，因为没有实际插入新数据

        // 测试插入相同字段值但不同文档ID
        let result = index.insert(2, "apple".to_string(), now_ms());
        assert!(result.is_ok());
        assert!(result.unwrap());

        // 测试不允许重复键的情况
        let config = BTreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: false,
        };
        let unique_index = BTreeIndex::new("unique_index".to_string(), Some(config));

        let result = unique_index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());

        let result = unique_index.insert(2, "apple".to_string(), now_ms());
        assert!(result.is_err());
        match result {
            Err(BTreeError::AlreadyExists { .. }) => (),
            _ => panic!("Expected AlreadyExists error"),
        }
    }

    #[test]
    fn test_batch_insert() {
        let index = create_test_index();

        // 准备批量插入的数据
        let items = vec![
            (1, "apple".to_string()),
            (2, "banana".to_string()),
            (3, "cherry".to_string()),
            (4, "date".to_string()),
            (5, "eggplant".to_string()),
        ];

        // 测试批量插入
        let result = index.batch_insert(items, now_ms());
        assert!(result.is_ok());

        assert_eq!(index.len(), 5);

        // 测试重复插入
        let items = vec![(6, "apple".to_string()), (7, "banana".to_string())];

        let result = index.batch_insert(items, now_ms());
        assert!(result.is_ok());

        assert_eq!(index.len(), 5); // 字段值数量仍然是5

        // 测试重复键的情况
        let config = BTreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: false,
        };
        let unique_index = BTreeIndex::new("unique_index".to_string(), Some(config));

        let result = unique_index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());

        let items = vec![(2, "apple".to_string()), (3, "cherry".to_string())];
        let result = unique_index.batch_insert(items, now_ms());
        assert!(result.is_ok());
        assert_eq!(unique_index.len(), 2); // 字段值数量是2
    }

    #[test]
    fn test_remove() {
        let index = create_populated_index();

        // 测试删除存在的条目
        let result = index.remove(1, "apple".to_string(), now_ms());
        assert!(result);

        // 测试删除不存在的条目
        let result = index.remove(100, "nonexistent".to_string(), now_ms());
        assert!(!result);

        // 测试删除后的搜索
        let result = index.search_with(&"apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(!ids.contains(&1)); // ID 1 已被删除
        assert!(ids.contains(&6)); // ID 6 仍然存在

        // 测试删除所有相关文档后，键应该被完全移除
        let result = index.remove(6, "apple".to_string(), now_ms());
        assert!(result);

        let result = index.search_with(&"apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_none()); // 键应该已经被完全移除
    }

    #[test]
    fn test_search() {
        let index = create_populated_index();

        // 测试精确搜索
        let result = index.search_with(&"apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(ids.contains(&1));
        assert!(ids.contains(&6));

        // 测试搜索不存在的键
        let result = index.search_with(&"nonexistent".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_none());
    }

    #[test]
    fn test_range_search() {
        let index = create_populated_index();
        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let cherry = "cherry".to_string();
        let date = "date".to_string();
        let eggplant = "eggplant".to_string();

        // 测试等于查询
        let query = RangeQuery::Eq(apple.clone());
        let results =
            index.search_range_with(query, |k, ids| (true, vec![(k.clone(), ids.clone())]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");

        // 测试大于查询
        let query = RangeQuery::Gt(cherry.clone());
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"date".to_string()));
        assert!(results.contains(&"eggplant".to_string()));

        // 测试大于等于查询
        let query = RangeQuery::Ge(cherry.clone());
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"cherry".to_string()));

        // 测试小于查询
        let query = RangeQuery::Lt(cherry.clone());
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&apple));
        assert!(results.contains(&banana));

        // 测试小于等于查询
        let query = RangeQuery::Le(cherry.clone());
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&cherry));

        // 测试范围查询
        let query = RangeQuery::Between(banana.clone(), date.clone());
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&banana));
        assert!(results.contains(&cherry));
        assert!(results.contains(&date));

        // 测试包含查询
        let keys = vec![apple.clone(), eggplant.clone()];
        let query = RangeQuery::Include(keys);
        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&apple));
        assert!(results.contains(&eggplant));

        // 测试提前终止搜索
        let query = RangeQuery::Ge(apple.clone());
        let results = index.search_range_with(query, |k, _| (k != "banana", vec![k.clone()]));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "apple");
        assert_eq!(results[1], "banana");
    }

    #[test]
    fn test_logical_queries() {
        let index = create_populated_index();

        // 额外插入一些测试数据以丰富测试用例
        let _ = index.insert(8, "grape".to_string(), now_ms());
        let _ = index.insert(9, "fig".to_string(), now_ms());
        let _ = index.insert(10, "berry".to_string(), now_ms());
        let _ = index.insert(11, "berry".to_string(), now_ms());

        // 准备常用的查询键
        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let berry = "berry".to_string();
        let cherry = "cherry".to_string();
        let date = "date".to_string();
        let eggplant = "eggplant".to_string();
        let fig = "fig".to_string();
        let grape = "grape".to_string();

        // ===== 测试 AND 操作 =====
        // 测试两个有交集的范围的 AND 操作
        let query = RangeQuery::And(vec![
            Box::new(RangeQuery::Le(date.clone())), // <= date (apple, banana, cherry, date)
            Box::new(RangeQuery::Ge(cherry.clone())), // >= cherry (cherry, date, eggplant, fig, grape)
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&cherry));
        assert!(results.contains(&date));

        // 测试空交集的 AND 操作
        let query = RangeQuery::And(vec![
            Box::new(RangeQuery::Lt(cherry.clone())), // < cherry (apple, banana)
            Box::new(RangeQuery::Gt(date.clone())),   // > date (eggplant, fig, grape)
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 0); // 应该为空集

        // 测试精确匹配和范围查询的 AND 操作
        let query = RangeQuery::And(vec![
            Box::new(RangeQuery::Ge(banana.clone())),   // >= banana
            Box::new(RangeQuery::Lt(eggplant.clone())), // < eggplant
            Box::new(RangeQuery::Eq(cherry.clone())),   // == cherry
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 1);
        assert!(results.contains(&cherry));

        // ===== 测试 OR 操作 =====
        // 测试两个不相交范围的 OR 操作
        let query = RangeQuery::Or(vec![
            Box::new(RangeQuery::Le(banana.clone())), // <= banana (apple, banana)
            Box::new(RangeQuery::Ge(fig.clone())),    // >= fig (fig, grape)
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 4);
        assert!(results.contains(&apple));
        assert!(results.contains(&banana));
        assert!(results.contains(&fig));
        assert!(results.contains(&grape));

        // 测试有重叠的 OR 操作
        let query = RangeQuery::Or(vec![
            Box::new(RangeQuery::Between(banana.clone(), date.clone())), // banana到date
            Box::new(RangeQuery::Between(cherry.clone(), fig.clone())),  // cherry到fig
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert_eq!(results.len(), 6);
        assert!(results.contains(&banana));
        assert!(results.contains(&berry));
        assert!(results.contains(&cherry));
        assert!(results.contains(&date));
        assert!(results.contains(&eggplant));
        assert!(results.contains(&fig));

        // ===== 测试 NOT 操作 =====
        // 测试基本的 NOT 操作
        let query = RangeQuery::Not(Box::new(RangeQuery::Between(
            cherry.clone(),
            eggplant.clone(),
        )));

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert!(results.contains(&apple));
        assert!(results.contains(&banana));
        assert!(results.contains(&fig));
        assert!(results.contains(&grape));
        assert!(!results.contains(&cherry));
        assert!(!results.contains(&date));
        assert!(!results.contains(&eggplant));

        // 测试 NOT + Eq 操作
        let query = RangeQuery::Not(Box::new(RangeQuery::Eq(apple.clone())));

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert!(!results.contains(&apple));
        assert!(results.contains(&banana));
        assert!(results.contains(&cherry));
        // ...验证其它键

        // ===== 测试复合逻辑查询 =====
        // 测试 AND(OR, OR) 复杂嵌套
        let query = RangeQuery::And(vec![
            Box::new(RangeQuery::Or(vec![
                Box::new(RangeQuery::Le(cherry.clone())), // <= cherry
                Box::new(RangeQuery::Ge(fig.clone())),    // >= fig
            ])),
            Box::new(RangeQuery::Or(vec![
                Box::new(RangeQuery::Le(banana.clone())),   // <= banana
                Box::new(RangeQuery::Ge(eggplant.clone())), // >= eggplant
            ])),
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert!(results.contains(&apple));
        assert!(results.contains(&banana));
        assert!(results.contains(&fig));
        assert!(results.contains(&grape));
        assert!(!results.contains(&cherry));
        assert!(!results.contains(&date));

        // 测试 OR(NOT, NOT) 复杂嵌套
        let query = RangeQuery::Or(vec![
            Box::new(RangeQuery::Not(Box::new(RangeQuery::Ge(date.clone())))), // NOT >= date
            Box::new(RangeQuery::Not(Box::new(RangeQuery::Le(cherry.clone())))), // NOT <= cherry
        ]);

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        // 这应该返回所有键，因为每个键要么 < date 要么 > cherry
        assert_eq!(results.len(), index.len());

        // 测试 NOT(AND) 复合操作
        let query = RangeQuery::Not(Box::new(RangeQuery::And(vec![
            Box::new(RangeQuery::Ge(cherry.clone())),   // >= cherry
            Box::new(RangeQuery::Le(eggplant.clone())), // <= eggplant
        ])));

        let results = index.search_range_with(query, |k, _| (true, vec![k.clone()]));
        assert!(results.contains(&apple));
        assert!(results.contains(&banana));
        assert!(results.contains(&fig));
        assert!(results.contains(&grape));
        assert!(!results.contains(&cherry));
        assert!(!results.contains(&date));
        assert!(!results.contains(&eggplant));

        // 测试提前终止功能
        let query = RangeQuery::Or(vec![
            Box::new(RangeQuery::Ge(apple.clone())),
            Box::new(RangeQuery::Le(grape.clone())),
        ]);

        let mut count = 0;
        let results = index.search_range_with(query, |_, _| {
            count += 1;
            (count < 3, vec![count.to_string()])
        });

        assert_eq!(results.len(), 3);
        assert_eq!(count, 3); // 确认查询在第三项后停止
    }

    #[test]
    fn test_range_keys() {
        let index = create_populated_index();

        // 测试 search_range_keys 方法处理 And 逻辑
        let apple = "apple".to_string();
        let banana = "banana".to_string();
        let cherry = "cherry".to_string();
        let eggplant = "eggplant".to_string();

        let query = RangeQuery::And(vec![
            Box::new(RangeQuery::Ge(banana.clone())),
            Box::new(RangeQuery::Le(cherry.clone())),
        ]);

        let keys = index.range_keys(query);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&banana));
        assert!(keys.contains(&cherry));

        // 测试 search_range_keys 方法处理 Or 逻辑
        let query = RangeQuery::Or(vec![
            Box::new(RangeQuery::Eq(apple.clone())),
            Box::new(RangeQuery::Eq(eggplant.clone())),
        ]);

        let keys = index.range_keys(query);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&apple));
        assert!(keys.contains(&eggplant));

        // 测试 search_range_keys 方法处理 Not 逻辑
        let query = RangeQuery::Not(Box::new(RangeQuery::Eq(apple.clone())));

        let keys = index.range_keys(query);
        assert!(!keys.contains(&apple));
        assert!(keys.contains(&banana));
        assert!(keys.contains(&cherry));
    }

    #[test]
    fn test_prefix_search() {
        let index = create_populated_index();

        // 插入一些带前缀的数据
        let _ = index.insert(10, "app".to_string(), now_ms());
        let _ = index.insert(11, "application".to_string(), now_ms());

        // 测试前缀搜索
        let results = index.search_prefix_with("app", |k, _| (true, Some(k.to_string())));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"app".to_string()));
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"application".to_string()));

        // 测试提前终止搜索
        let results = index.search_prefix_with("app", |k, _| (k != "apple", Some(k.to_string())));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "app");
        assert_eq!(results[1], "apple");
    }

    #[tokio::test]
    async fn test_serialization() {
        let index = create_populated_index();

        // 序列化元数据
        let mut buf = Vec::new();
        let result = index.store_metadata(&mut buf, now_ms());
        assert!(result.is_ok());

        println!("Serialized metadata: {:?}", const_hex::encode(&buf));

        // 反序列化元数据
        let result = BTreeIndex::<u64, String>::load_metadata(&buf[..]);
        let mut loaded_index = result.unwrap();

        // 验证元数据
        assert_eq!(loaded_index.name(), "test_index");
        assert_eq!(loaded_index.len(), 0); // 注意：load_metadata 只加载元数据，不加载 postings

        // 模拟 load_buckets 函数
        let bucket_data = Arc::new(Mutex::new(Vec::new()));

        // 存储 bucket 数据
        {
            let bucket_data_clone = bucket_data.clone();
            let result = index
                .store_dirty_postings(async |bucket_id, data| {
                    let mut guard = bucket_data_clone.lock().await;
                    while guard.len() <= bucket_id as usize {
                        guard.push(Vec::new());
                    }
                    guard[bucket_id as usize] = data.to_vec();
                    Ok(true)
                })
                .await;
            assert!(result.is_ok());
        }

        // 加载 bucket 数据
        {
            let bucket_data_clone = bucket_data.clone();
            let result = loaded_index
                .load_postings(async |bucket_id| {
                    let guard = bucket_data_clone.lock().await;
                    if bucket_id as usize >= guard.len() {
                        return Err(BTreeError::Generic {
                            name: "test".to_string(),
                            source: "Bucket not found".into(),
                        }
                        .into());
                    }
                    Ok(Some(guard[bucket_id as usize].clone()))
                })
                .await;
            assert!(result.is_ok());
        }

        // 验证加载后的索引
        assert_eq!(loaded_index.len(), index.len());

        // 测试搜索
        let result = loaded_index.search_with(&"apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(ids.contains(&1));
        assert!(ids.contains(&6));
    }

    #[test]
    fn test_bucket_overflow() {
        // 创建一个非常小的 bucket 大小的索引，以便测试 bucket 溢出
        let config = BTreeConfig {
            bucket_overload_size: 100, // 非常小的 bucket 大小
            allow_duplicates: true,
        };
        let index = BTreeIndex::new("overflow_test".to_string(), Some(config));

        // 插入足够多的数据以触发 bucket 溢出
        for i in 0..100 {
            let key = format!("key_{}", i);
            let _ = index.insert(i, key, now_ms());
        }

        // 验证创建了多个 bucket
        println!("index.stats(): {:?}", index.stats());
        assert!(index.stats().max_bucket_id > 1);

        // 验证所有数据都可以被搜索到
        for i in 0..100 {
            let key = format!("key_{}", i);
            let result = index.search_with(&key, |ids| Some(ids.clone()));
            assert!(result.is_some());
            let ids = result.unwrap();
            assert!(ids.contains(&i));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_batch_insert_chaos() {
        let index = Arc::new(BTreeIndex::<u64, String>::new(
            "chaos_index".to_string(),
            Some(BTreeConfig {
                bucket_overload_size: 256,
                allow_duplicates: true,
            }),
        ));

        let n_threads = 10;
        let n_keys_per_thread = 100;
        let barrier = Arc::new(Barrier::new(n_threads));
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let index = index.clone();
            let b = barrier.clone();
            handles.push(tokio::spawn(async move {
                // 等待所有线程准备好
                b.wait().await;

                let base = t * n_keys_per_thread;
                let items: Vec<_> = (0..n_keys_per_thread)
                    .map(|i| ((base + i) as u64, format!("key_{}", base + i)))
                    .collect();
                // 多次调用 batch_insert，模拟混乱
                for _ in 0..3 {
                    let _ = index.batch_insert(items.clone(), now_ms());
                }
            }));
        }

        // 等待所有任务完成
        futures::future::try_join_all(handles).await.unwrap();

        // 检查所有数据都能被检索到
        for t in 0..n_threads {
            let base = t * n_keys_per_thread;
            for i in 0..n_keys_per_thread {
                let key = format!("key_{}", base + i);
                let result = index.search_with(&key, |ids| Some(ids.clone()));
                assert!(result.is_some(), "key {} not found", key);
                assert!(
                    result.unwrap().contains(&((base + i) as u64)),
                    "id {} not found for key {}",
                    base + i,
                    key
                );
            }
        }
    }

    #[test]
    fn test_stats() {
        let index = create_test_index();

        // 初始状态
        let stats = index.stats();
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.search_count, 0);
        assert_eq!(stats.insert_count, 0);
        assert_eq!(stats.delete_count, 0);

        // 插入一些数据
        let _ = index.insert(1, "apple".to_string(), now_ms());
        let _ = index.insert(2, "banana".to_string(), now_ms());

        // 检查插入后的统计信息
        let stats = index.stats();
        assert_eq!(stats.num_elements, 2);
        assert_eq!(stats.insert_count, 2);

        // 执行一些搜索
        let _ = index.search_with(&"apple".to_string(), |_| Some(()));
        let _: Vec<()> =
            index.search_range_with(RangeQuery::Ge("a".to_string()), |_, _| (true, vec![]));

        // 检查搜索后的统计信息
        let stats = index.stats();
        assert_eq!(stats.search_count, 2);

        // 删除一些数据
        let _ = index.remove(1, "apple".to_string(), now_ms());

        // 检查删除后的统计信息
        let stats = index.stats();
        assert_eq!(stats.num_elements, 1);
        assert_eq!(stats.delete_count, 1);
    }

    #[test]
    fn test_counting_writer() {
        // 测试 CountingWriter
        let mut writer = CountingWriter::new();
        assert_eq!(writer.size(), 0);

        // 写入一些数据
        let data = b"Hello, world!";
        let result = std::io::Write::write(&mut writer, data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), data.len());
        assert_eq!(writer.size(), data.len());

        // 测试 count_cbor 函数
        let size = CountingWriter::count_cbor(&"test string".to_string());
        assert!(size > 0);

        let size_complex = CountingWriter::count_cbor(&vec![1, 2, 3, 4, 5]);
        assert!(size_complex > 0);
    }
}
