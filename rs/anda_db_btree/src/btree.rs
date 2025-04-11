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
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Debug,
    hash::Hash,
    sync::atomic::{AtomicU32, AtomicU64, Ordering as AtomicOrdering},
};

use crate::BtreeError;

/// B-tree index for efficient key-value lookups
///
/// This structure provides a thread-safe B-tree index implementation
/// that supports concurrent reads and writes, as well as efficient range queries.
pub struct BtreeIndex<FV, PK>
where
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
    PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
{
    /// Index name
    name: String,

    /// Index configuration
    config: BtreeConfig,

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
    metadata: RwLock<BtreeIndexMetadata>,

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
pub struct BtreeConfig {
    /// Maximum size of a bucket before creating a new one (in bytes)
    pub bucket_overload_size: u32,

    /// Whether to allow duplicate keys
    /// If false, attempting to insert a duplicate key will result in an error
    pub allow_duplicates: bool,
}

impl Default for BtreeConfig {
    fn default() -> Self {
        BtreeConfig {
            bucket_overload_size: 1024 * 512,
            allow_duplicates: true,
        }
    }
}

/// Index metadata containing configuration and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtreeIndexMetadata {
    /// Index name
    pub name: String,

    /// Index configuration
    pub config: BtreeConfig,

    /// Index statistics
    pub stats: BtreeIndexStats,
}

/// Index statistics for monitoring and diagnostics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BtreeIndexStats {
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
// struct BtreeIndexOwned<FV, PK>
// where
//     FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
//     PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
// {
//     // #[serde(skip)]
//     postings: DashMap<String, PostingValue<PK>>,
//     metadata: BtreeIndexMetadata,
// }

// Helper structure for serialization and deserialization of index metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BtreeIndexOwnedHelper {
    // Serialized postings map (not used for actual data, just for structure)
    postings: HashMap<String, String>,

    // Index metadata
    metadata: BtreeIndexMetadata,
}

// Reference structure for serializing the index
#[derive(Serialize)]
struct BtreeIndexRef<'a, FV, PK>
where
    FV: Eq + Hash + Debug + Clone + Serialize,
    PK: Ord + Debug + Clone + Serialize,
{
    postings: &'a DashMap<FV, PostingValue<PK>>,
    metadata: &'a BtreeIndexMetadata,
}

/// Range query specification for flexible querying
#[derive(Debug, Clone)]
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
    Include(BTreeSet<FV>),
}

impl<FV, PK> BtreeIndex<FV, PK>
where
    FV: Eq + Ord + Hash + Debug + Clone + Serialize + DeserializeOwned,
    PK: Ord + Debug + Clone + Serialize + DeserializeOwned,
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
    /// * `BtreeIndex` - A new instance of the B-tree index
    pub fn new(name: String, config: Option<BtreeConfig>) -> Self {
        let config = config.unwrap_or_default();
        let bucket_overload_size = config.bucket_overload_size;
        BtreeIndex {
            name: name.clone(),
            config: config.clone(),
            postings: DashMap::new(),
            buckets: DashMap::from_iter(vec![(0, (0, true, Vec::new()))]),
            btree: RwLock::new(BTreeSet::new()),
            metadata: RwLock::new(BtreeIndexMetadata {
                name,
                config,
                stats: BtreeIndexStats::default(),
            }),
            bucket_overload_size,
            max_bucket_id: AtomicU32::new(0),
            search_count: AtomicU64::new(0),
        }
    }

    /// Loads an index from a reader
    /// This only loads metadata, you need to call [`Self::load_buckets`] to load the actual posting data
    ///
    /// # Arguments
    ///
    /// * `r` - Any type implementing the [`futures::io::AsyncRead`] trait
    ///
    /// # Returns
    ///
    /// * `Result<Self, Error>` - Loaded index or error
    pub async fn load_metadata<R: AsyncRead + Unpin>(mut r: R) -> Result<Self, BtreeError> {
        // Read all data from the reader into a buffer
        let data = {
            let mut buf = Vec::new();
            AsyncReadExt::read_to_end(&mut r, &mut buf)
                .await
                .map_err(|err| BtreeError::Generic {
                    name: "unknown".to_string(),
                    source: err.into(),
                })?;
            buf
        };

        // Deserialize the index metadata
        let index: BtreeIndexOwnedHelper =
            ciborium::from_reader(&data[..]).map_err(|err| BtreeError::Serialization {
                name: "unknown".to_string(),
                source: err.into(),
            })?;

        // Extract configuration values
        let bucket_overload_size = index.metadata.config.bucket_overload_size;
        let search_count = AtomicU64::new(index.metadata.stats.search_count);
        let max_bucket_id = AtomicU32::new(index.metadata.stats.max_bucket_id);
        Ok(BtreeIndex {
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
    ///   `F: AsyncFn(u32) -> Result<Vec<u8>, BtreeError>`
    ///   The function should take a bucket ID as input and return a vector of bytes
    ///   containing the serialized posting data.
    ///
    /// # Returns
    ///
    /// * `Result<(), BtreeError>` - Success or error
    pub async fn load_buckets<F>(&mut self, f: F) -> Result<(), BtreeError>
    where
        F: AsyncFn(u32) -> Result<Vec<u8>, BtreeError>,
    {
        for i in 0..=self.max_bucket_id.load(AtomicOrdering::Relaxed) {
            let data = f(i).await?;
            let postings: HashMap<FV, PostingValue<PK>> = ciborium::from_reader(&data[..])
                .map_err(|err| BtreeError::Serialization {
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
    pub fn metadata(&self) -> BtreeIndexMetadata {
        let mut metadata = self.metadata.read().clone();
        metadata.stats.num_elements = self.postings.len() as u64;
        metadata.stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        metadata.stats.max_bucket_id = self.max_bucket_id.load(AtomicOrdering::Relaxed);
        metadata
    }

    /// Gets current statistics about the index
    pub fn stats(&self) -> BtreeIndexStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.num_elements = self.postings.len() as u64;
        stats.search_count = self.search_count.load(AtomicOrdering::Relaxed);
        stats.max_bucket_id = self.max_bucket_id.load(AtomicOrdering::Relaxed);
        stats
    }

    /// Gets a posting by key and applies a function to it
    ///
    /// # Arguments
    ///
    /// * `field_value` - The key to look up
    /// * `f` - Function to apply to the posting value if found.
    ///   where `F: FnOnce(&FV, &Vec<PK>) -> Option<R>`
    ///
    ///
    /// # Returns
    ///
    /// * `Result<Option<R>, BtreeError>` - Result of the function or error if key not found
    pub fn get_posting_with<R, F>(&self, field_value: &FV, f: F) -> Result<Option<R>, BtreeError>
    where
        F: FnOnce(&FV, &Vec<PK>) -> Option<R>,
    {
        self.postings
            .get(field_value)
            .map(|v| f(field_value, &v.2))
            .ok_or_else(|| BtreeError::NotFound {
                name: self.name.clone(),
                value: format!("{:?}", field_value),
            })
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
    /// * `Err(BtreeError)` if failed
    pub fn insert(&self, doc_id: PK, field_value: FV, now_ms: u64) -> Result<bool, BtreeError> {
        let bucket = self.max_bucket_id.load(AtomicOrdering::Acquire);

        // Calculate the size increase for this insertion
        let size_increase = {
            match self.postings.entry(field_value.clone()) {
                dashmap::Entry::Occupied(mut entry) => {
                    // Check if duplicate keys are allowed
                    if !self.config.allow_duplicates {
                        return Err(BtreeError::AlreadyExists {
                            name: self.name.clone(),
                            id: format!("{:?}", doc_id),
                            value: format!("{:?}", field_value),
                        });
                    }

                    let posting = entry.get_mut();
                    // Add segment_id if it doesn't exist
                    if !posting.2.contains(&doc_id) {
                        let size_increase = CountingWriter::count_cbor(&doc_id) as u32 + 2;
                        posting.2.push(doc_id);
                        posting.1 += 1; // increment version
                        size_increase
                    } else {
                        0 // No change if document_id already exists
                    }
                }
                dashmap::Entry::Vacant(entry) => {
                    // Create a new posting for this field value
                    let posting = (bucket, 1, vec![doc_id]);
                    let size_increase = CountingWriter::count_cbor(&posting) as u32 + 2;
                    entry.insert(posting);

                    // Add the field value to the B-tree for range queries
                    self.btree.write().insert(field_value.clone());
                    size_increase
                }
            }
            // release the lock on the entry
        };

        // If the index was modified, update bucket state
        if size_increase > 0 {
            // Update bucket state
            let mut b = self
                .buckets
                .get_mut(&bucket)
                .ok_or_else(|| BtreeError::Generic {
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
                    b.2.push(field_value);
                }
            } else {
                // If the current bucket is full, create a new one
                let new_bucket = self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;
                if let Some(mut posting) = self.postings.get_mut(&field_value) {
                    // Update the posting's bucket ID
                    posting.0 = new_bucket;
                    let size_decrease = CountingWriter::count_cbor(&posting) as u32 + 2;
                    // Remove the current field value from the current bucket
                    // The freed space can still accommodate small growth in other field values
                    if let Some(pos) = b.2.iter().position(|k| &field_value == k) {
                        b.0 = b.0.saturating_sub(size_decrease);
                        b.1 = true;
                        b.2.swap_remove(pos);
                    }
                }

                // Create a new bucket and migrate this data to it
                self.buckets
                    .insert(new_bucket, (size_increase, true, vec![field_value]));
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
    /// * `Result<(), BtreeError>` - Success or error
    pub fn batch_insert<I>(&self, items: I, now_ms: u64) -> Result<(), BtreeError>
    where
        I: IntoIterator<Item = (PK, FV)>,
    {
        // Group batch data by different field values
        let mut grouped_items: BTreeMap<FV, Vec<PK>> = BTreeMap::new();
        for (doc_id, field_value) in items {
            grouped_items.entry(field_value).or_default().push(doc_id);
        }

        if grouped_items.is_empty() {
            return Ok(());
        }
        let insert_count = grouped_items.len() as u64;

        // Batch processing work, optimized by grouping writes by bucket
        // Phase 1: Update the postings collection
        let mut bucket_updates: HashMap<u32, (u32, Vec<FV>)> = HashMap::new();
        for (field_value, doc_ids) in grouped_items {
            // Consider concurrent scenarios where buckets might be modified by other threads
            let bucket = self.max_bucket_id.load(AtomicOrdering::Relaxed);
            let (size_increase, inserted_to_bucket) = match self.postings.entry(field_value.clone())
            {
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

                    if new_ids.is_empty() {
                        (0, posting.0)
                    } else {
                        let size_increase = new_ids
                            .iter()
                            .fold(0, |acc, id| acc + CountingWriter::count_cbor(id) as u32 + 2);
                        posting.2.extend(new_ids);
                        posting.1 += 1; // increment version
                        (size_increase, posting.0)
                    }
                }
                dashmap::Entry::Vacant(entry) => {
                    let posting = (bucket, 1, doc_ids);
                    let size_increase = CountingWriter::count_cbor(&posting) as u32 + 2;
                    entry.insert(posting);
                    self.btree.write().insert(field_value.clone());
                    (size_increase, bucket)
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

        // Phase 2: Update bucket states
        let mut field_values_to_create: Vec<(u32, Vec<FV>)> = Vec::new();
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
                    field_values_to_create.push((bucket_id, field_values));
                }
            }
        }

        // Phase 3: Create new buckets if needed
        if !field_values_to_create.is_empty() {
            let mut next_bucket_id = self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;
            let mut nb = self
                .buckets
                .entry(next_bucket_id)
                .or_insert_with(|| (0, true, Vec::new()));
            for (old_bucket_id, field_values) in field_values_to_create {
                let mut ob =
                    self.buckets
                        .get_mut(&old_bucket_id)
                        .ok_or_else(|| BtreeError::Generic {
                            name: self.name.clone(),
                            source: format!("bucket {old_bucket_id} not found").into(),
                        })?;

                for field_value in field_values {
                    if let Some(mut posting) = self.postings.get_mut(&field_value) {
                        posting.0 = next_bucket_id;
                        let size_decrease = CountingWriter::count_cbor(&posting) as u32 + 2;
                        // Remove current FV from the current bucket
                        // The freed space can still accommodate small growth in other FV postings
                        if let Some(pos) = ob.2.iter().position(|k| &field_value == k) {
                            ob.0 = ob.0.saturating_sub(size_decrease);
                            ob.1 = true;
                            ob.2.swap_remove(pos);
                        }

                        if nb.2.is_empty() || nb.0 + size_decrease < self.bucket_overload_size {
                            // Bucket has enough space, update directly
                            nb.0 += size_decrease;
                            if !nb.2.contains(&field_value) {
                                nb.2.push(field_value);
                            }
                        } else {
                            // Bucket doesn't have enough space, need to migrate to the next bucket
                            next_bucket_id =
                                self.max_bucket_id.fetch_add(1, AtomicOrdering::Release) + 1;
                            nb = self
                                .buckets
                                .entry(next_bucket_id)
                                .or_insert_with(|| (0, true, Vec::new()));
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

        Ok(())
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
        if let Some(mut posting) = self.postings.get_mut(&field_value) {
            if let Some(pos) = posting.2.iter().position(|id| id == &doc_id) {
                let size_decrease = if posting.2.len() > 1 {
                    CountingWriter::count_cbor(&doc_id) as u32 + 2
                } else {
                    CountingWriter::count_cbor(&posting) as u32 + 2
                };
                posting.1 += 1; // increment version
                posting.2.swap_remove(pos);

                // Update the bucket state
                if let Some(mut b) = self.buckets.get_mut(&posting.0) {
                    b.0 = b.0.saturating_sub(size_decrease);
                    b.1 = true;

                    if posting.2.is_empty() {
                        // remove FV from the bucket
                        if let Some(pos) = b.2.iter().position(|k| &field_value == k) {
                            b.2.swap_remove(pos);
                        }
                    }
                }
            }

            if posting.2.is_empty() {
                self.btree.write().remove(&field_value);

                // If no documents left, remove the key entirely
                drop(posting);
                self.postings.remove(&field_value);
            }

            self.update_metadata(|m| {
                m.stats.version += 1;
                m.stats.last_deleted = now_ms;
                m.stats.delete_count += 1;
            });
            return true;
        }

        false
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
    pub fn search_with<F, R>(&self, field_value: FV, f: F) -> Option<R>
    where
        F: FnOnce(&Vec<PK>) -> Option<R>,
    {
        self.search_count.fetch_add(1, AtomicOrdering::Relaxed);

        self.postings
            .get(&field_value)
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
        F: FnMut(&FV, &Vec<PK>) -> (bool, Option<R>),
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
                    if let Some(r) = rt {
                        results.push(r);
                    }
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
                        if let Some(r) = rt {
                            results.push(r);
                        }
                        if !conti {
                            return results;
                        }
                    }
                }
            }
            RangeQuery::Ge(start_key) => {
                for k in self.btree.read().range(std::ops::RangeFrom {
                    start: start_key.clone(),
                }) {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        if let Some(r) = rt {
                            results.push(r);
                        }
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
                    .range(std::ops::RangeTo {
                        end: end_key.clone(),
                    })
                    .rev()
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        if let Some(r) = rt {
                            results.push(r);
                        }
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
                    .range(std::ops::RangeToInclusive {
                        end: end_key.clone(),
                    })
                    .rev()
                {
                    if let Some(posting) = self.postings.get(k) {
                        let (conti, rt) = f(k, &posting.2);
                        if let Some(r) = rt {
                            results.push(r);
                        }
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
                        if let Some(r) = rt {
                            results.push(r);
                        }
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
                        if let Some(r) = rt {
                            results.push(r);
                        }
                        if !conti {
                            return results;
                        }
                    }
                }
            }
        }

        results
    }

    /// Stores the index metadata to a writer
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`futures::io::AsyncWrite`] trait
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - Success or error
    pub async fn store_metadata<W: AsyncWrite + Unpin>(
        &self,
        mut w: W,
        now_ms: u64,
    ) -> Result<(), BtreeError> {
        let serialized_data = {
            let mut buf = Vec::with_capacity(8192);
            self.update_metadata(|m| {
                m.stats.last_saved = now_ms.max(m.stats.last_saved);
            });
            let postings: DashMap<FV, PostingValue<PK>> = DashMap::new();
            ciborium::into_writer(
                &BtreeIndexRef {
                    postings: &postings,
                    metadata: &self.metadata(),
                },
                &mut buf,
            )
            .map_err(|err| BtreeError::Serialization {
                name: self.name.clone(),
                source: err.into(),
            })?;
            buf
        };

        AsyncWriteExt::write_all(&mut w, &serialized_data)
            .await
            .map_err(|err| BtreeError::Generic {
                name: self.name.clone(),
                source: err.into(),
            })?;

        Ok(())
    }

    /// Stores dirty buckets to persistent storage using the provided async function
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
    /// * `Result<(), BtreeError>` - Success or error
    pub async fn store_dirty_buckets<F>(&self, f: F) -> Result<(), BtreeError>
    where
        F: AsyncFn(u32, Vec<u8>) -> Result<bool, BtreeError>,
    {
        for mut bucket in self.buckets.iter_mut() {
            if bucket.1 {
                // If the bucket is dirty, it needs to be persisted
                let mut postings: HashMap<&FV, ciborium::Value> =
                    HashMap::with_capacity(bucket.2.len());
                for k in bucket.2.iter() {
                    if let Some(posting) = self.postings.get(k) {
                        postings.insert(
                            k,
                            ciborium::cbor!(posting).map_err(|err| BtreeError::Serialization {
                                name: self.name.clone(),
                                source: err.into(),
                            })?,
                        );
                    }
                }

                let mut data = Vec::new();
                ciborium::into_writer(&postings, &mut data).map_err(|err| {
                    BtreeError::Serialization {
                        name: self.name.clone(),
                        source: err.into(),
                    }
                })?;

                if let Ok(conti) = f(*bucket.key(), data).await {
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
        F: FnOnce(&mut BtreeIndexMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

impl<PK> BtreeIndex<String, PK>
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
    use tokio::sync::Mutex;

    // 获取当前时间戳（毫秒）
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    // 辅助函数：创建一个测试用的 B-tree 索引
    fn create_test_index() -> BtreeIndex<String, u64> {
        let config = BtreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: true,
        };
        BtreeIndex::new("test_index".to_string(), Some(config))
    }

    // 辅助函数：创建一个测试用的 B-tree 索引并插入一些数据
    fn create_populated_index() -> BtreeIndex<String, u64> {
        let index = create_test_index();

        // 插入一些测试数据
        let _ = index.insert(1, "apple".to_string(), now_ms());
        let _ = index.insert(2, "banana".to_string(), now_ms());
        let _ = index.insert(3, "cherry".to_string(), now_ms());
        let _ = index.insert(4, "date".to_string(), now_ms());
        let _ = index.insert(5, "elderberry".to_string(), now_ms());

        // 测试重复键
        let _ = index.insert(6, "apple".to_string(), now_ms());
        let _ = index.insert(7, "banana".to_string(), now_ms());

        index
    }

    #[tokio::test]
    async fn test_create_index() {
        let index = create_test_index();

        assert_eq!(index.name(), "test_index");
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let metadata = index.metadata();
        assert_eq!(metadata.name, "test_index");
        assert_eq!(metadata.stats.num_elements, 0);
    }

    #[tokio::test]
    async fn test_insert() {
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
        let config = BtreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: false,
        };
        let unique_index = BtreeIndex::new("unique_index".to_string(), Some(config));

        let result = unique_index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());

        let result = unique_index.insert(2, "apple".to_string(), now_ms());
        assert!(result.is_err());
        match result {
            Err(BtreeError::AlreadyExists { .. }) => (),
            _ => panic!("Expected AlreadyExists error"),
        }
    }

    #[tokio::test]
    async fn test_batch_insert() {
        let index = create_test_index();

        // 准备批量插入的数据
        let items = vec![
            (1, "apple".to_string()),
            (2, "banana".to_string()),
            (3, "cherry".to_string()),
            (4, "date".to_string()),
            (5, "elderberry".to_string()),
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
        let config = BtreeConfig {
            bucket_overload_size: 1024,
            allow_duplicates: false,
        };
        let unique_index = BtreeIndex::new("unique_index".to_string(), Some(config));

        let result = unique_index.insert(1, "apple".to_string(), now_ms());
        assert!(result.is_ok());

        let items = vec![(2, "apple".to_string()), (3, "cherry".to_string())];
        let result = unique_index.batch_insert(items, now_ms());
        assert!(result.is_ok());
        assert_eq!(unique_index.len(), 2); // 字段值数量是2
    }

    #[tokio::test]
    async fn test_remove() {
        let index = create_populated_index();

        // 测试删除存在的条目
        let result = index.remove(1, "apple".to_string(), now_ms());
        assert!(result);

        // 测试删除不存在的条目
        let result = index.remove(100, "nonexistent".to_string(), now_ms());
        assert!(!result);

        // 测试删除后的搜索
        let result = index.search_with("apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(!ids.contains(&1)); // ID 1 已被删除
        assert!(ids.contains(&6)); // ID 6 仍然存在

        // 测试删除所有相关文档后，键应该被完全移除
        let result = index.remove(6, "apple".to_string(), now_ms());
        assert!(result);

        let result = index.search_with("apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_none()); // 键应该已经被完全移除
    }

    #[tokio::test]
    async fn test_search() {
        let index = create_populated_index();

        // 测试精确搜索
        let result = index.search_with("apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(ids.contains(&1));
        assert!(ids.contains(&6));

        // 测试搜索不存在的键
        let result = index.search_with("nonexistent".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_range_search() {
        let index = create_populated_index();

        // 测试等于查询
        let query = RangeQuery::Eq("apple".to_string());
        let results =
            index.search_range_with(query, |k, ids| (true, Some((k.clone(), ids.clone()))));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "apple");

        // 测试大于查询
        let query = RangeQuery::Gt("cherry".to_string());
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"date".to_string()));
        assert!(results.contains(&"elderberry".to_string()));

        // 测试大于等于查询
        let query = RangeQuery::Ge("cherry".to_string());
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"cherry".to_string()));

        // 测试小于查询
        let query = RangeQuery::Lt("cherry".to_string());
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"banana".to_string()));

        // 测试小于等于查询
        let query = RangeQuery::Le("cherry".to_string());
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"cherry".to_string()));

        // 测试范围查询
        let query = RangeQuery::Between("banana".to_string(), "date".to_string());
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 3);
        assert!(results.contains(&"banana".to_string()));
        assert!(results.contains(&"cherry".to_string()));
        assert!(results.contains(&"date".to_string()));

        // 测试包含查询
        let mut keys = BTreeSet::new();
        keys.insert("apple".to_string());
        keys.insert("elderberry".to_string());
        let query = RangeQuery::Include(keys);
        let results = index.search_range_with(query, |k, _| (true, Some(k.clone())));
        assert_eq!(results.len(), 2);
        assert!(results.contains(&"apple".to_string()));
        assert!(results.contains(&"elderberry".to_string()));

        // 测试提前终止搜索
        let query = RangeQuery::Ge("apple".to_string());
        let results = index.search_range_with(query, |k, _| (k != "banana", Some(k.clone())));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "apple");
        assert_eq!(results[1], "banana");
    }

    #[tokio::test]
    async fn test_prefix_search() {
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
        let result = index.store_metadata(&mut buf, now_ms()).await;
        assert!(result.is_ok());

        println!("Serialized metadata: {:?}", const_hex::encode(&buf));

        // 反序列化元数据
        let result = BtreeIndex::<String, u64>::load_metadata(&buf[..]).await;
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
                .store_dirty_buckets(async |bucket_id, data| {
                    let mut guard = bucket_data_clone.lock().await;
                    while guard.len() <= bucket_id as usize {
                        guard.push(Vec::new());
                    }
                    guard[bucket_id as usize] = data;
                    Ok(true)
                })
                .await;
            assert!(result.is_ok());
        }

        // 加载 bucket 数据
        {
            let bucket_data_clone = bucket_data.clone();
            let result = loaded_index
                .load_buckets(async |bucket_id| {
                    let guard = bucket_data_clone.lock().await;
                    if bucket_id as usize >= guard.len() {
                        return Err(BtreeError::Generic {
                            name: "test".to_string(),
                            source: "Bucket not found".into(),
                        });
                    }
                    Ok(guard[bucket_id as usize].clone())
                })
                .await;
            assert!(result.is_ok());
        }

        // 验证加载后的索引
        assert_eq!(loaded_index.len(), index.len());

        // 测试搜索
        let result = loaded_index.search_with("apple".to_string(), |ids| Some(ids.clone()));
        assert!(result.is_some());
        let ids = result.unwrap();
        assert!(ids.contains(&1));
        assert!(ids.contains(&6));
    }

    #[tokio::test]
    async fn test_bucket_overflow() {
        // 创建一个非常小的 bucket 大小的索引，以便测试 bucket 溢出
        let config = BtreeConfig {
            bucket_overload_size: 100, // 非常小的 bucket 大小
            allow_duplicates: true,
        };
        let index = BtreeIndex::new("overflow_test".to_string(), Some(config));

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
            let result = index.search_with(key, |ids| Some(ids.clone()));
            assert!(result.is_some());
            let ids = result.unwrap();
            assert!(ids.contains(&i));
        }
    }

    #[tokio::test]
    async fn test_get_posting_with() {
        let index = create_populated_index();

        // 测试获取存在的 posting
        let result = index.get_posting_with(&"apple".to_string(), |k, posting| {
            Some((k.clone(), posting.clone()))
        });
        assert!(result.is_ok());
        let (key, ids) = result.unwrap().unwrap();
        assert_eq!(key, "apple");
        assert!(ids.contains(&1));
        assert!(ids.contains(&6));

        // 测试获取不存在的 posting
        let result = index.get_posting_with(&"nonexistent".to_string(), |k, posting| {
            Some((k.clone(), posting.clone()))
        });
        assert!(result.is_err());
        match result {
            Err(BtreeError::NotFound { .. }) => (),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[tokio::test]
    async fn test_stats() {
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
        let _ = index.search_with("apple".to_string(), |_| Some(()));
        let _ = index.search_range_with(RangeQuery::Ge("a".to_string()), |_, _| (true, Some(())));

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

    #[tokio::test]
    async fn test_counting_writer() {
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
