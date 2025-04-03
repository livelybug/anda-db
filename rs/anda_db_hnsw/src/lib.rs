//! # Anda-DB HNSW Vector Search Library
//!
//! A high-performance implementation of Hierarchical Navigable Small World (HNSW) algorithm
//! for approximate nearest neighbor search in high-dimensional spaces.
//!
//! HNSW is a graph-based indexing algorithm that creates a multi-layered structure
//! to enable fast and accurate nearest neighbor search in high-dimensional spaces.
//!
//! ## Features
//!
//! - Fast approximate nearest neighbor search;
//! - Multiple distance metrics (Euclidean, Cosine, Inner Product, Manhattan);
//! - Configurable index parameters;
//! - Thread-safe implementation with concurrent read/write operations;
//! - Serialization and deserialization support;
//! - Support for bf16 (brain floating point) vector storage for memory efficiency.
//!

use dashmap::{DashMap, DashSet};
use half::bf16;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::{distr::Uniform, prelude::*, rng};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub use half;

/// Returns the current unix timestamp in milliseconds.
#[inline]
pub fn unix_ms() -> u64 {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    ts.as_millis() as u64
}

/// Errors that can occur when working with HNSW index.
#[derive(Error, Debug, Clone)]
pub enum HnswError {
    /// Database-related errors.
    #[error("DB error: {0}")]
    Db(String),

    /// CBOR serialization/deserialization errors.
    #[error("CBOR serialization error: {0}")]
    Cbor(String),

    /// Error when vector dimensions don't match the index dimension.
    #[error("Vector dimension mismatch")]
    DimensionMismatch,

    /// Error when trying to search an empty index.
    #[error("Index is empty")]
    EmptyIndex,

    /// Error when index has reached its maximum capacity.
    #[error("Index is full")]
    IndexFull,

    /// Error when a vector with the specified ID is not found.
    #[error("Not found {0}")]
    NotFound(u64),

    /// Error when trying to add a vector with an ID that already exists.
    #[error("Vector {0} already exists")]
    AlreadyExists(u64),

    /// Error related to distance metric calculations.
    #[error("Distance metric error: {0}")]
    DistanceMetric(String),
}

/// Distance metric types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm).
    Euclidean,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Negative inner product (dot product).
    InnerProduct,
    /// Manhattan distance (L1 norm).
    Manhattan,
}

impl DistanceMetric {
    /// Compute the distance between two vectors using the selected metric.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    ///
    /// * `Result<f32, HnswError>` - The computed distance or an error if the dimensions don't match.
    pub fn compute(&self, a: &[bf16], b: &[bf16]) -> Result<f32, HnswError> {
        if a.len() != b.len() {
            return Err(HnswError::DimensionMismatch);
        }

        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
            DistanceMetric::Cosine => Ok(cosine_distance(a, b)),
            DistanceMetric::InnerProduct => Ok(inner_product(a, b)),
            DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
        }
    }
}

/// HNSW configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of layers in the graph. Default is 16.
    pub max_layers: u8,

    /// Maximum number of connections per node in each layer. Default is 32.
    pub max_connections: u8,

    /// Expansion factor during index construction. Default is 200.
    pub ef_construction: usize,

    /// Number of candidates to consider during search. Default is 50.
    pub ef_search: usize,

    /// Distance metric to use for similarity calculations. Default is Euclidean.
    pub distance_metric: DistanceMetric,

    /// Maximum number of elements in the index (None for unlimited). Default is None.
    pub max_elements: Option<u64>,

    /// Scale factor for adjusting layer distribution. Default is 1.0.
    pub scale_factor: Option<f64>,

    /// Strategy for selecting neighbors. Default is Heuristic.
    pub select_neighbors_strategy: SelectNeighborsStrategy,
}

impl HnswConfig {
    /// Creates a layer generator based on the configuration.
    ///
    /// # Returns
    ///
    /// * `LayerGen` - A layer generator with the configured parameters.
    pub fn layer_gen(&self) -> LayerGen {
        LayerGen::new_with_scale(
            self.max_connections,
            self.scale_factor.unwrap_or(1.0),
            self.max_layers,
        )
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_layers: 16,
            max_connections: 32,
            ef_construction: 200,
            ef_search: 50,
            distance_metric: DistanceMetric::Euclidean,
            max_elements: None,
            scale_factor: None,
            select_neighbors_strategy: SelectNeighborsStrategy::Heuristic,
        }
    }
}

/// Neighbor selection strategies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SelectNeighborsStrategy {
    /// Simple greedy strategy that selects the closest nodes.
    Simple,

    /// Heuristic neighbor selection algorithm (NN-descent) that considers diversity.
    /// It will comsume more time to build the index, but will improve search performance.
    Heuristic,
}

/// HNSW graph node.
#[derive(Clone, Serialize, Deserialize)]
struct HnswNode {
    /// Unique identifier for the node.
    id: u64,

    /// The highest layer this node appears in.
    layer: u8,

    /// Vector data stored in bf16 format.
    vector: Vec<bf16>,

    /// Neighbors at each layer (layer -> [(id, distance)]).
    neighbors: Vec<SmallVec<[(u64, bf16); 64]>>,
}

/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMetadata {
    /// Index format version.
    version: u16,

    /// Creation timestamp (unix ms).
    created_at: u64,

    /// Last modification timestamp (unix ms).
    last_modified: u64,

    /// Index statistics.
    stats: IndexStats,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStats {
    /// Maximum layer in the index.
    pub max_layer: u8,

    /// Number of elements in the index.
    pub num_elements: u64,

    /// Number of deleted elements.
    pub num_deleted: u64,

    /// Average number of connections per node.
    pub avg_connections: f32,

    /// Number of search operations performed.
    pub search_count: u64,

    /// Number of insert operations performed.
    pub insert_count: u64,

    /// Number of delete operations performed.
    pub delete_count: u64,
}

/// Serializable HNSW index structure (owned version).
#[derive(Clone, Serialize, Deserialize)]
struct HnswIndexSerdeOwn {
    config: HnswConfig,
    nodes: DashMap<u64, HnswNode>,
    entry_point: (u64, u8),
    dimension: usize,
    metadata: IndexMetadata,
    deleted_ids: DashSet<u64>,
}

/// Serializable HNSW index structure (reference version).
#[derive(Clone, Serialize)]
struct HnswIndexSerdeRef<'a> {
    config: &'a HnswConfig,
    nodes: &'a DashMap<u64, HnswNode>,
    entry_point: (u64, u8),
    dimension: usize,
    metadata: &'a IndexMetadata,
    deleted_ids: &'a DashSet<u64>,
}

impl From<HnswIndexSerdeOwn> for HnswIndex {
    fn from(val: HnswIndexSerdeOwn) -> Self {
        let layer_gen = val.config.layer_gen();

        HnswIndex {
            dimension: val.dimension,
            config: val.config,
            layer_gen,
            nodes: val.nodes,
            entry_point: RwLock::new(val.entry_point),
            metadata: RwLock::new(val.metadata),
            deleted_ids: val.deleted_ids,
            is_dirty: RwLock::new(false),
        }
    }
}

impl From<HnswIndex> for HnswIndexSerdeOwn {
    fn from(val: HnswIndex) -> Self {
        HnswIndexSerdeOwn {
            config: val.config,
            nodes: val.nodes,
            entry_point: *val.entry_point.read(),
            dimension: val.dimension,
            metadata: val.metadata.read().clone(),
            deleted_ids: val.deleted_ids,
        }
    }
}

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    /// Dimensionality of vectors in the index.
    dimension: usize,

    /// Index configuration.
    config: HnswConfig,

    /// Layer generator for assigning layers to new nodes.
    layer_gen: LayerGen,

    /// Map of node IDs to nodes.
    nodes: DashMap<u64, HnswNode>,

    /// Entry point for search (node_id, layer)
    entry_point: RwLock<(u64, u8)>,

    /// Index metadata.
    metadata: RwLock<IndexMetadata>,

    /// Set of deleted node IDs.
    deleted_ids: DashSet<u64>,

    /// Flag indicating whether the index has been modified since last save.
    is_dirty: RwLock<bool>,
}

impl HnswIndex {
    /// Creates a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimensionality of vectors to be indexed.
    /// * `config` - Index configuration.
    ///
    /// # Returns
    ///
    /// * `HnswIndex` - A new HNSW index.
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        let layer_gen = config.layer_gen();

        let now_ms = unix_ms();
        Self {
            dimension,
            config,
            layer_gen,
            nodes: DashMap::new(),
            entry_point: RwLock::new((0, 0)),
            metadata: RwLock::new(IndexMetadata {
                version: 1,
                created_at: now_ms,
                last_modified: now_ms,
                stats: IndexStats::default(),
            }),
            deleted_ids: DashSet::new(),
            is_dirty: RwLock::new(false),
        }
    }

    /// Inserts a vector into the index
    ///
    /// This is the core insertion method that implements the HNSW algorithm:
    /// 1. Randomly assigns a layer to the new node
    /// 2. Finds the nearest neighbors in each layer
    /// 3. Creates connections to selected neighbors
    /// 4. Updates the entry point if necessary
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data as bf16 values
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error
    pub fn insert(&self, id: u64, vector: Vec<bf16>) -> Result<(), HnswError> {
        if vector.len() != self.dimension {
            return Err(HnswError::DimensionMismatch);
        }

        // Check if ID already exists.
        if self.nodes.contains_key(&id) {
            return Err(HnswError::AlreadyExists(id));
        }

        // Check capacity
        if let Some(max) = self.config.max_elements {
            if self.nodes.len() as u64 >= max {
                return Err(HnswError::IndexFull);
            }
        }

        let mut distance_cache = HashMap::with_capacity(self.config.ef_construction * 2);
        let mut entry_point_dist = f32::INFINITY;
        let (mut entry_point_node, current_max_layer) = { *self.entry_point.read() };

        // Randomly determine the node's layer
        let layer = self.layer_gen.generate(current_max_layer);

        // Create new node
        let mut node = HnswNode {
            id,
            layer,
            vector,
            neighbors: vec![
                SmallVec::with_capacity(self.config.max_connections as usize * 2);
                layer as usize + 1
            ],
        };

        // If this is the first node, set it as the entry point
        {
            if self.nodes.is_empty() {
                *self.entry_point.write() = (id, layer);
                self.nodes.insert(id, node);
                self.update_metadata(|m| {
                    m.stats.max_layer = layer;
                    m.stats.insert_count += 1;
                });
                *self.is_dirty.write() = true;
                return Ok(());
            }
        }

        // Search from top layer down to find the best entry point
        for current_layer in (current_max_layer.min(layer + 1)..=current_max_layer).rev() {
            // Find the nearest neighbor at the current layer
            let nearest = self.search_layer(
                &node.vector,
                entry_point_node,
                current_layer,
                1,
                &mut distance_cache,
            )?;
            if let Some(node) = nearest.first() {
                if node.1 < entry_point_dist {
                    entry_point_node = node.0;
                    entry_point_dist = node.1;
                }
            }
        }

        // Connect the new node to its nearest neighbors
        #[allow(clippy::type_complexity)]
        let mut neighbors_to_update: Vec<(u64, u8, SmallVec<[(u64, bf16); 64]>)> =
            Vec::with_capacity(64); // id, layer, connections
        // Use distance cache to reduce redundant calculations
        let mut multi_distance_cache: HashMap<(u64, u64), f32> = HashMap::new();
        // Build connections from bottom layer up
        for current_layer in 0..=layer {
            // Find nearest neighbors at the current layer
            let nearest = self.search_layer(
                &node.vector,
                entry_point_node,
                current_layer,
                self.config.ef_construction,
                &mut distance_cache,
            )?;

            let max_connections = if current_layer > 0 {
                self.config.max_connections as usize
            } else {
                self.config.max_connections as usize * 2
            };

            let selected_neighbors = self.select_neighbors(
                nearest,
                max_connections,
                self.config.select_neighbors_strategy,
                &mut multi_distance_cache,
            )?;

            let should_truncate = (max_connections as f64 * 1.5) as usize;
            // Connect new node to its nearest neighbors
            for &(neighbor, dist) in &selected_neighbors {
                if neighbor != id {
                    if let Some(mut neighbor_node) = self.nodes.get_mut(&neighbor) {
                        let dist = bf16::from_f32(dist);
                        node.neighbors[current_layer as usize].push((neighbor, dist));

                        if let Some(n_layer) =
                            neighbor_node.neighbors.get_mut(current_layer as usize)
                        {
                            n_layer.push((id, dist));
                            // If over threshold, collect nodes to update later rather than updating immediately
                            if n_layer.len() >= should_truncate {
                                // Collect information for later batch processing
                                neighbors_to_update.push((
                                    neighbor,
                                    current_layer,
                                    n_layer.clone(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Add the new node to the index
        match self.nodes.try_entry(id) {
            None => {
                return Err(HnswError::Db(
                    "Failed to insert node into index".to_string(),
                ));
            }
            Some(entry) => match entry {
                dashmap::Entry::Occupied(_) => {
                    return Err(HnswError::AlreadyExists(id));
                }
                dashmap::Entry::Vacant(v) => {
                    v.insert(node);
                    *self.is_dirty.write() = true;
                    // Update entry point if new node is at a higher layer
                    if layer > current_max_layer {
                        *self.entry_point.write() = (id, layer);
                    }

                    self.update_metadata(|m| {
                        if layer > m.stats.max_layer {
                            m.stats.max_layer = layer;
                        }
                        m.stats.insert_count += 1;
                    });
                }
            },
        }

        // Update collected nodes after releasing all locks
        for (node_id, layer, connections) in neighbors_to_update {
            let max_connections = if layer > 0 {
                self.config.max_connections as usize
            } else {
                self.config.max_connections as usize * 2
            };
            let candidates: Vec<(u64, f32)> = connections
                .iter()
                .map(|&(id, dist)| (id, dist.to_f32()))
                .collect();

            if let Ok(selected) = self.select_neighbors(
                candidates,
                max_connections,
                self.config.select_neighbors_strategy,
                &mut multi_distance_cache,
            ) {
                if let Some(mut node) = self.nodes.get_mut(&node_id) {
                    if let Some(n_layer) = node.neighbors.get_mut(layer as usize) {
                        // Update neighbor connections
                        *n_layer = selected
                            .into_iter()
                            .map(|(id, dist)| (id, bf16::from_f32(dist)))
                            .collect();
                    }
                }
            }
        }

        Ok(())
    }

    /// Inserts a vector with f32 values into the index
    ///
    /// Automatically converts f32 values to bf16 for storage efficiency
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data as f32 values
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error
    pub fn insert_f32(&self, id: u64, vector: Vec<f32>) -> Result<(), HnswError> {
        self.insert(id, vector.into_iter().map(bf16::from_f32).collect())
    }

    /// Searches for the k nearest neighbors to the query vector
    ///
    /// Results are sorted by ascending distance (closest first)
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Vector of (id, distance) pairs
    pub fn search(&self, query: &[bf16], k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        if query.len() != self.dimension {
            return Err(HnswError::DimensionMismatch);
        }

        if self.nodes.is_empty() {
            return Err(HnswError::EmptyIndex);
        }

        let mut distance_cache = HashMap::new();
        let mut current_dist = f32::INFINITY;
        let (mut current_node, current_max_layer) = { *self.entry_point.read() };
        // 从最高层向下搜索入口点
        for current_layer in (1..=current_max_layer).rev() {
            let nearest =
                self.search_layer(query, current_node, current_layer, 1, &mut distance_cache)?;
            if let Some(node) = nearest.first() {
                if node.1 < current_dist {
                    current_dist = node.1;
                    current_node = node.0;
                }
            }
        }

        // 在底层搜索最近的邻居
        let ef = self.config.ef_search.max(k);
        let mut results = self.search_layer(query, current_node, 0, ef, &mut distance_cache)?;
        results.truncate(k);

        self.update_metadata(|m| {
            m.stats.search_count += 1;
        });

        Ok(results)
    }

    /// Searches for nearest neighbors using f32 query vector
    ///
    /// Automatically converts f32 values to bf16 for distance calculations
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector as f32 values
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Vector of (id, distance) pairs sorted by ascending distance
    pub fn search_f32(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        self.search(
            &query.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
            k,
        )
    }

    /// Searches for nearest neighbors within a specific layer
    ///
    /// This is an internal method used by both insert and search operations
    /// to find nearest neighbors at a specific layer of the graph.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `entry_point` - Starting node ID for the search
    /// * `layer` - Layer to search in
    /// * `ef` - Expansion factor (number of candidates to consider)
    /// * `distance_cache` - Cache of previously computed distances
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Vector of (id, distance) pairs sorted by ascending distance
    fn search_layer(
        &self,
        query: &[bf16],
        entry_point: u64,
        layer: u8,
        ef: usize,
        distance_cache: &mut HashMap<u64, f32>,
    ) -> Result<Vec<(u64, f32)>, HnswError> {
        let mut visited: HashSet<u64> = HashSet::with_capacity(ef * 2);
        let mut candidates: BinaryHeap<(Reverse<OrderedFloat<f32>>, u64)> =
            BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::with_capacity(ef * 2);

        // Calculate distance to entry point
        let entry_dist = match self.nodes.get(&entry_point) {
            Some(node) => self.get_distance_with_cache(distance_cache, query, &node)?,
            None => return Err(HnswError::NotFound(entry_point)),
        };

        // Initialize candidate list
        visited.insert(entry_point);
        candidates.push((Reverse(OrderedFloat(entry_dist)), entry_point));
        results.push((OrderedFloat(entry_dist), entry_point));

        // Get nearest candidates
        while let Some((Reverse(OrderedFloat(dist)), point)) = candidates.pop() {
            if let Some((OrderedFloat(max_dist), _)) = results.peek() {
                if &dist > max_dist && results.len() >= ef {
                    break;
                };
            }

            // Check neighbors of current node
            if let Some(node) = self.nodes.get(&point) {
                if let Some(neighbors) = node.neighbors.get(layer as usize) {
                    for &(neighbor, _) in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            if let Some(neighbor_node) = self.nodes.get(&neighbor) {
                                match self.get_distance_with_cache(
                                    distance_cache,
                                    query,
                                    &neighbor_node,
                                ) {
                                    Ok(dist) => {
                                        if let Some((OrderedFloat(max_dist), _)) = results.peek() {
                                            if &dist < max_dist || results.len() < ef {
                                                candidates
                                                    .push((Reverse(OrderedFloat(dist)), neighbor));
                                                results.push((OrderedFloat(dist), neighbor));

                                                // Prune distant results
                                                if results.len() > ef {
                                                    results.pop();
                                                }
                                            }
                                        } else {
                                            candidates
                                                .push((Reverse(OrderedFloat(dist)), neighbor));
                                            results.push((OrderedFloat(dist), neighbor));
                                        }
                                    }
                                    Err(e) => {
                                        log::warn!("Distance calculation error: {:?}", e);
                                        distance_cache.insert(neighbor, f32::INFINITY);
                                    }
                                };
                            }
                        }
                    }
                }
            }
        }

        Ok(results
            .into_sorted_vec()
            .into_iter()
            .map(|(d, id)| (id, d.0))
            .collect())
    }

    /// Selects the best neighbors for a node based on the configured strategy
    ///
    /// # Arguments
    ///
    /// * `candidates` - List of candidate nodes with their distances
    /// * `m` - Maximum number of neighbors to select
    /// * `strategy` - Strategy to use for selection (Simple or Heuristic)
    /// * `distance_cache` - Cache of previously computed distances between nodes
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Selected neighbors with their distances
    fn select_neighbors(
        &self,
        candidates: Vec<(u64, f32)>,
        m: usize,
        strategy: SelectNeighborsStrategy,
        distance_cache: &mut HashMap<(u64, u64), f32>,
    ) -> Result<Vec<(u64, f32)>, HnswError> {
        if candidates.len() <= m {
            return Ok(candidates);
        }

        match strategy {
            SelectNeighborsStrategy::Simple => {
                // Simple strategy: select m closest neighbors
                let mut selected = candidates;
                selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                selected.truncate(m);
                Ok(selected)
            }
            SelectNeighborsStrategy::Heuristic => {
                // Heuristic strategy: balance distance and connection diversity
                // Create candidate and result sets
                let mut selected: Vec<(u64, f32)> = Vec::with_capacity(m);
                let mut remaining = candidates;
                remaining.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                // Add the first nearest neighbor
                if !remaining.is_empty() {
                    selected.push(remaining.remove(0));
                }

                // Greedily add remaining nodes while considering diversity
                while selected.len() < m && !remaining.is_empty() {
                    let mut best_candidate_idx = 0;
                    let mut best_distance_improvement = f32::NEG_INFINITY;

                    for (i, &(cand_id, cand_dist)) in remaining.iter().enumerate() {
                        let mut min_dist_to_selected = f32::INFINITY;
                        for &(sel_id, _) in &selected {
                            let cache_key = if cand_id < sel_id {
                                (cand_id, sel_id)
                            } else {
                                (sel_id, cand_id)
                            };

                            let dist = if let Some(&cached_dist) = distance_cache.get(&cache_key) {
                                cached_dist
                            } else if let (Some(cand_node), Some(sel_node)) =
                                (self.nodes.get(&cand_id), self.nodes.get(&sel_id))
                            {
                                let new_dist = self
                                    .config
                                    .distance_metric
                                    .compute(&cand_node.vector, &sel_node.vector)?;
                                distance_cache.insert(cache_key, new_dist);
                                new_dist
                            } else {
                                continue;
                            };

                            min_dist_to_selected = min_dist_to_selected.min(dist);
                        }

                        // 平衡因子 = 距离查询点的近似程度 + 与已选集合的多样性
                        let improvement = min_dist_to_selected - cand_dist;
                        if improvement > best_distance_improvement {
                            best_distance_improvement = improvement;
                            best_candidate_idx = i;
                        }
                    }

                    // 添加最佳候选点
                    // 添加最佳候选点（防止索引越界）
                    if best_candidate_idx < remaining.len() {
                        selected.push(remaining.swap_remove(best_candidate_idx));
                    } else if !remaining.is_empty() {
                        // 退化为简单策略，避免可能的死循环
                        selected.push(remaining.remove(0));
                    } else {
                        break;
                    }
                }

                Ok(selected)
            }
        }
    }

    /// Gets the distance between a query vector and a node, using cache when available
    ///
    /// # Arguments
    ///
    /// * `cache` - Cache of previously computed distances
    /// * `query` - Query vector
    /// * `neighbor` - Node to compute distance to
    ///
    /// # Returns
    ///
    /// * `Result<f32, HnswError>` - Computed distance
    fn get_distance_with_cache(
        &self,
        cache: &mut HashMap<u64, f32>,
        query: &[bf16],
        neighbor: &HnswNode,
    ) -> Result<f32, HnswError> {
        match cache.get(&neighbor.id) {
            Some(&dist) => Ok(dist),
            None => {
                let dist = self
                    .config
                    .distance_metric
                    .compute(query, &neighbor.vector)?;
                cache.insert(neighbor.id, dist);
                Ok(dist)
            }
        }
    }

    /// Saves the index to a writer.
    ///
    /// Serializes the index using CBOR format.
    ///
    /// # Arguments
    ///
    /// * `w` - Writer to save the index to.
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error.
    pub async fn save<W: AsyncWrite + Unpin>(&self, mut w: W) -> Result<(), HnswError> {
        {
            let mut metadata = self.metadata.write();
            metadata.last_modified = unix_ms();
        }

        let mut buf = Vec::new();
        ciborium::into_writer(
            &HnswIndexSerdeRef {
                config: &self.config,
                nodes: &self.nodes,
                entry_point: *self.entry_point.read(),
                dimension: self.dimension,
                metadata: &self.metadata.read(),
                deleted_ids: &self.deleted_ids,
            },
            &mut buf,
        )
        .map_err(|e| HnswError::Cbor(e.to_string()))?;

        *self.is_dirty.write() = false;

        AsyncWriteExt::write_all(&mut w, &buf)
            .await
            .map_err(|e| HnswError::Db(e.to_string()))?;
        Ok(())
    }

    /// Loads an index from a reader.
    ///
    /// Deserializes the index from CBOR format.
    ///
    /// # Arguments
    ///
    /// * `r` - Reader to load the index from.
    ///
    /// # Returns
    ///
    /// * `Result<Self, HnswError>` - Loaded index or error.
    pub async fn load<R: AsyncRead + Unpin>(mut r: R) -> Result<Self, HnswError> {
        let data = {
            let mut buf = Vec::new();
            AsyncReadExt::read_to_end(&mut r, &mut buf)
                .await
                .map_err(|e| HnswError::Db(e.to_string()))?;
            buf
        };

        let index: HnswIndexSerdeOwn =
            ciborium::from_reader(&data[..]).map_err(|e| HnswError::Cbor(e.to_string()))?;
        let index: HnswIndex = index.into();
        *index.is_dirty.write() = false;
        Ok(index)
    }

    /// Returns the number of vectors in the index.
    ///
    /// # Returns
    ///
    /// * `usize` - Number of vectors
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Checks if the index is empty
    ///
    /// # Returns
    ///
    /// * `bool` - True if the index contains no vectors
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the dimensionality of vectors in the index
    ///
    /// # Returns
    ///
    /// * `usize` - Vector dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Removes a vector from the index
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to remove
    ///
    /// # Returns
    ///
    /// * `Result<bool, HnswError>` - True if the vector was removed, false if it wasn't found
    pub fn remove(&self, id: u64) -> Result<bool, HnswError> {
        let mut deleted = false;

        if let Some((_, node)) = self.nodes.remove(&id) {
            deleted = true;
            self.deleted_ids.insert(id);
            self.try_update_entry_point(&node)?;
            self.update_metadata(|m| {
                m.stats.delete_count += 1;
            });

            // 遍历所有节点，删除与已删除节点的连接
            self.nodes.iter_mut().for_each(|mut n| {
                for layer in 0..=(n.layer as usize) {
                    if let Some(pos) = n.neighbors[layer].iter().position(|&(idx, _)| idx == id) {
                        n.neighbors[layer].swap_remove(pos);
                    }
                }
            });

            *self.is_dirty.write() = true;
        }

        Ok(deleted)
    }

    /// Updates the entry point after a node deletion
    ///
    /// # Arguments
    ///
    /// * `deleted_node` - The node that was deleted
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error
    fn try_update_entry_point(&self, deleted_node: &HnswNode) -> Result<(), HnswError> {
        let (_, mut max_layer) = {
            let point = self.entry_point.read();
            if point.0 != deleted_node.id {
                return Ok(());
            }
            *point
        };

        loop {
            if let Some(neighbors) = deleted_node.neighbors.get(max_layer as usize) {
                for &(neighbor, _) in neighbors {
                    if let Some(neighbor_node) = self.nodes.get(&neighbor) {
                        *self.entry_point.write() = (neighbor, neighbor_node.layer);
                        return Ok(());
                    }
                }
            }

            if max_layer == 0 {
                break;
            }
            max_layer -= 1;
        }

        if let Some(node) = self.nodes.iter().next() {
            *self.entry_point.write() = (node.id, node.layer);
        } else {
            *self.entry_point.write() = (0, 0);
        }

        if log::log_enabled!(log::Level::Debug) {
            let entry_point = self.entry_point.read();
            log::debug!(
                "Updated entry point to {} at layer {}",
                entry_point.0,
                entry_point.1
            );
        }
        Ok(())
    }

    /// Gets current statistics about the index
    ///
    /// # Returns
    ///
    /// * `IndexStats` - Current statistics
    pub fn stats(&self) -> IndexStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.num_elements = self.nodes.len() as u64;

        // 计算平均连接数
        let total_connections: u64 = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|v| v.len() as u64).sum::<u64>())
            .sum();

        let active_nodes = self.nodes.len();
        stats.avg_connections = if active_nodes > 0 {
            total_connections as f32 / active_nodes as f32
        } else {
            0.0
        };

        stats
    }

    /// Updates the index metadata
    ///
    /// # Arguments
    ///
    /// * `f` - Function that modifies the metadata
    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

/// Random layer generator for HNSW
///
/// Provides functionality for generating random layers for nodes in the HNSW graph.
/// Uses an exponential distribution to ensure upper layers are sparse and lower layers are dense.
#[derive(Debug)]
pub struct LayerGen {
    /// Uniform distribution sampler
    uniform: Uniform<f64>,
    /// Scaling factor for the exponential distribution
    scale: f64,
    /// Maximum layer (exclusive)
    max_level: u8,
}

impl LayerGen {
    /// Creates a new layer generator
    ///
    /// # Arguments
    ///
    /// * `max_connections` - Maximum connections per node
    /// * `max_level` - Maximum layer (exclusive)
    ///
    /// # Returns
    ///
    /// * `LayerGen` - New layer generator
    pub fn new(max_connections: u8, max_level: u8) -> Self {
        Self::new_with_scale(max_connections, 1.0, max_level)
    }

    /// Creates a new layer generator with a custom scale factor
    ///
    /// # Arguments
    ///
    /// * `max_connections` - Maximum connections per node
    /// * `scale_factor` - Custom scale factor for the distribution
    /// * `max_level` - Maximum layer (exclusive)
    ///
    /// # Returns
    ///
    /// * `LayerGen` - New layer generator
    pub fn new_with_scale(max_connections: u8, scale_factor: f64, max_level: u8) -> Self {
        let base_scale = 1.0 / (max_connections as f64).ln();
        LayerGen {
            uniform: Uniform::<f64>::new(0.0, 1.0).unwrap(),
            scale: base_scale * scale_factor,
            max_level,
        }
    }

    /// Generates a random layer for a new node
    ///
    /// Uses an exponential distribution to determine the layer,
    /// ensuring that higher layers have fewer nodes.
    ///
    /// # Arguments
    ///
    /// * `current_max_layer` - Current maximum layer in the index
    ///
    /// # Returns
    ///
    /// * `u8` - Generated layer
    pub fn generate(&self, current_max_layer: u8) -> u8 {
        let mut r = rng();
        let val = r.sample(self.uniform);

        // 使用指数分布计算层级
        let level = (-val.ln() * self.scale).floor() as u8;

        // 确保层级在有效范围内
        level.min(current_max_layer + 1).min(self.max_level - 1)
    }
}

#[inline]
fn euclidean_distance(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn cosine_distance(a: &[bf16], b: &[bf16]) -> f32 {
    let dot_product: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() * y.to_f32())
        .sum();
    let norm_a: f32 = a.iter().map(|x| x.to_f32().powi(2)).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x.to_f32().powi(2)).sum::<f32>().sqrt();
    1.0 - (dot_product / (norm_a * norm_b))
}

#[inline]
fn inner_product(a: &[bf16], b: &[bf16]) -> f32 {
    -a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() * y.to_f32())
        .sum::<f32>()
}

#[inline]
fn manhattan_distance(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_distribution() {
        let lg = LayerGen::new(10, 16);
        let mut counts = [0; 16];

        // 生成大量样本以验证分布
        const SAMPLES: usize = 100_000;
        for _ in 0..SAMPLES {
            let level = lg.generate(15) as usize;
            counts[level] += 1;
        }

        // 验证层级分布是递减的
        for i in 1..16 {
            assert!(counts[i] <= counts[i - 1]);
        }

        // 验证最底层占比合理
        let bottom_ratio = counts[0] as f64 / SAMPLES as f64;
        assert!(bottom_ratio > 0.5);
    }

    #[tokio::test]
    async fn test_hnsw_basic() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(2, config);

        // 添加一些二维向量
        index.insert_f32(1, vec![1.0, 1.0]).unwrap();

        index.insert_f32(2, vec![1.0, 2.0]).unwrap();
        index.insert_f32(3, vec![2.0, 1.0]).unwrap();
        index.insert_f32(4, vec![2.0, 2.0]).unwrap();
        index.insert_f32(5, vec![3.0, 3.0]).unwrap();
        println!("Added vectors to index.");

        // 搜索最近的邻居
        let results = index.search_f32(&[1.1, 1.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 < results[1].1);

        println!("Search results: {:?}", results);

        // 测试持久化
        let mut data = Vec::new();
        index.save(&mut data).await.unwrap();
        println!("Serialized data size: {}", data.len());

        let loaded_index = HnswIndex::load(&data[..]).await.unwrap();

        println!("Loaded index stats: {:?}", loaded_index.stats());
        let loaded_results = loaded_index.search_f32(&[1.1, 1.1], 2).unwrap();
        assert_eq!(results, loaded_results);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_distance_metrics() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];

        // 欧氏距离
        let config = HnswConfig {
            distance_metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let index = HnswIndex::new(2, config);
        index.insert_f32(1, v1.clone()).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 1.4142135).abs() < 1e-6);

        // 余弦距离
        let config = HnswConfig {
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let index = HnswIndex::new(2, config);
        index.insert_f32(1, v1.clone()).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 1.0).abs() < 1e-6);

        // 内积
        let config = HnswConfig {
            distance_metric: DistanceMetric::InnerProduct,
            ..Default::default()
        };
        let index = HnswIndex::new(2, config);
        index.insert_f32(1, v1.clone()).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];

        // 曼哈顿距离
        let config = HnswConfig {
            distance_metric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let index = HnswIndex::new(2, config);
        index.insert_f32(1, v1.clone()).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_index() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(3, config);

        // 空索引搜索应该返回错误
        let result = index.search_f32(&[1.0, 2.0, 3.0], 5);
        assert!(matches!(result, Err(HnswError::EmptyIndex)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(3, config);

        // 插入维度不匹配的向量
        let result = index.insert_f32(1, vec![1.0, 2.0]);
        assert!(matches!(result, Err(HnswError::DimensionMismatch)));

        // 插入正确维度的向量
        index.insert_f32(1, vec![1.0, 2.0, 3.0]).unwrap();

        // 搜索维度不匹配的向量
        let result = index.search_f32(&[1.0, 2.0], 5);
        assert!(matches!(result, Err(HnswError::DimensionMismatch)));
    }

    #[test]
    fn test_duplicate_insert() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(2, config);

        // 首次插入成功
        index.insert_f32(1, vec![1.0, 2.0]).unwrap();

        // 重复插入同一ID应该失败
        let result = index.insert_f32(1, vec![3.0, 4.0]);
        assert!(matches!(result, Err(HnswError::AlreadyExists(1))));
    }

    #[test]
    fn test_remove() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(2, config);

        // 添加向量
        index.insert_f32(1, vec![1.0, 1.0]).unwrap();
        index.insert_f32(2, vec![2.0, 2.0]).unwrap();
        index.insert_f32(3, vec![3.0, 3.0]).unwrap();

        assert_eq!(index.len(), 3);

        // 删除存在的向量
        let result = index.remove(2).unwrap();
        assert!(result);
        assert_eq!(index.len(), 2);

        // 删除不存在的向量
        let result = index.remove(4).unwrap();
        assert!(!result);

        // 搜索应该只返回剩余的向量
        let results = index.search_f32(&[1.5, 1.5], 5).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(id, _)| *id == 1 || *id == 3));
    }

    #[test]
    fn test_max_elements() {
        let config = HnswConfig {
            max_elements: Some(3),
            ..Default::default()
        };
        let index = HnswIndex::new(2, config);

        // 添加向量直到达到最大容量
        index.insert_f32(1, vec![1.0, 1.0]).unwrap();
        index.insert_f32(2, vec![2.0, 2.0]).unwrap();
        index.insert_f32(3, vec![3.0, 3.0]).unwrap();

        // 超出容量限制应该失败
        let result = index.insert_f32(4, vec![4.0, 4.0]);
        assert!(matches!(result, Err(HnswError::IndexFull)));
    }

    #[test]
    fn test_select_neighbors_strategies() {
        // 测试简单策略
        let config = HnswConfig {
            select_neighbors_strategy: SelectNeighborsStrategy::Simple,
            ..Default::default()
        };
        let simple_index = HnswIndex::new(2, config);

        // 测试启发式策略
        let config = HnswConfig {
            select_neighbors_strategy: SelectNeighborsStrategy::Heuristic,
            ..Default::default()
        };
        let heuristic_index = HnswIndex::new(2, config);

        // 添加相同的向量到两个索引
        for i in 0..20 {
            let x = (i % 5) as f32;
            let y = (i / 5) as f32;
            simple_index.insert_f32(i, vec![x, y]).unwrap();
            heuristic_index.insert_f32(i, vec![x, y]).unwrap();
        }

        // 两种策略都应该能找到最近邻
        let simple_results = simple_index.search_f32(&[2.5, 2.5], 5).unwrap();
        let heuristic_results = heuristic_index.search_f32(&[2.5, 2.5], 5).unwrap();

        // 两种策略都应该返回5个结果
        assert_eq!(simple_results.len(), 5);
        assert_eq!(heuristic_results.len(), 5);
    }

    #[tokio::test]
    async fn test_file_persistence() {
        // 创建临时目录
        let mut data: Vec<u8> = Vec::new();

        // 创建并填充索引
        {
            let config = HnswConfig::default();
            let index = HnswIndex::new(3, config);

            for i in 0..100 {
                let x = (i % 10) as f32;
                let y = ((i / 10) % 10) as f32;
                let z = (i / 100) as f32;
                index.insert_f32(i, vec![x, y, z]).unwrap();
            }

            index.save(&mut data).await.unwrap();
        }

        // 从文件加载索引
        {
            let loaded_index = HnswIndex::load(&data[..]).await.unwrap();

            // 验证索引大小
            assert_eq!(loaded_index.len(), 100);

            // 验证搜索功能
            let results = loaded_index.search_f32(&[5.0, 5.0, 0.0], 10).unwrap();
            assert_eq!(results.len(), 10);
        }
    }

    #[test]
    fn test_stats() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(2, config);

        // 初始状态
        let stats = index.stats();
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.insert_count, 0);
        assert_eq!(stats.search_count, 0);
        assert_eq!(stats.delete_count, 0);

        // 添加向量
        for i in 0..10 {
            index.insert_f32(i, vec![i as f32, i as f32]).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.num_elements, 10);
        assert_eq!(stats.insert_count, 10);

        // 执行搜索
        for _ in 0..5 {
            index.search_f32(&[5.0, 5.0], 3).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.search_count, 5);

        // 删除向量
        index.remove(5).unwrap();
        index.remove(6).unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_elements, 8);
        assert_eq!(stats.delete_count, 2);
    }

    #[test]
    fn test_bf16_conversion() {
        // 测试f32到bf16的转换精度
        let original = [1.234f32, 5.678f32, 9.012f32];
        let bf16_vec: Vec<bf16> = original.iter().map(|&x| bf16::from_f32(x)).collect();
        let back_to_f32: Vec<f32> = bf16_vec.iter().map(|x| x.to_f32()).collect();

        // bf16有限的精度会导致一些舍入误差
        for (i, (orig, converted)) in original.iter().zip(back_to_f32.iter()).enumerate() {
            println!(
                "Original: {}, Converted: {}, Diff: {}",
                orig,
                converted,
                (orig - converted).abs()
            );
            // 允许一定的误差范围
            assert!(
                (orig - converted).abs() < 0.1,
                "Too much precision loss at index {}",
                i
            );
        }
    }

    #[test]
    fn test_large_dimension() {
        // 测试高维向量
        let dim = 128;
        let config = HnswConfig::default();
        let index = HnswIndex::new(dim, config);

        // 创建一些高维向量
        for i in 0..10 {
            let vec = vec![i as f32 / 10.0; dim];
            index.insert_f32(i, vec).unwrap();
        }

        // 搜索
        let query = vec![0.35; dim];
        let results = index.search_f32(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // 最近的应该是0.3或0.4
        assert!(results[0].0 == 3 || results[0].0 == 4);
    }

    #[test]
    fn test_entry_point_update() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(2, config);

        // 添加向量
        index.insert_f32(1, vec![1.0, 1.0]).unwrap();

        // 获取当前入口点
        let (entry_id, _) = *index.entry_point.read();
        assert_eq!(entry_id, 1);

        // 删除入口点
        index.remove(entry_id).unwrap();

        // 添加新向量，应该成为新的入口点
        index.insert_f32(2, vec![2.0, 2.0]).unwrap();

        let (new_entry_id, _) = *index.entry_point.read();
        assert_eq!(new_entry_id, 2);
    }

    #[test]
    #[ignore]
    fn test_concurrent_operations() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let config = HnswConfig::default();
        let index = Arc::new(HnswIndex::new(3, config));

        // 添加一些初始数据
        for i in 0..10 {
            index
                .insert_f32(i, vec![i as f32, i as f32, i as f32])
                .unwrap();
        }

        let num_threads = 10;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for t in 0..num_threads {
            let index_clone = Arc::clone(&index);
            let barrier_clone = Arc::clone(&barrier);

            let handle = thread::spawn(move || {
                // 等待所有线程准备好
                barrier_clone.wait();

                // 每个线程执行不同的操作
                let base_id = 100 + t * 100;

                // 插入操作
                for i in 0..10 {
                    let id = base_id + i;
                    let _ =
                        index_clone.insert_f32(id as u64, vec![id as f32, id as f32, id as f32]);
                }

                // 搜索操作
                for _ in 0..5 {
                    let _ = index_clone.search_f32(&[t as f32, t as f32, t as f32], 5);
                }

                // 删除操作
                for i in 0..5 {
                    let id = base_id + i;
                    let _ = index_clone.remove(id as u64);
                }
            });

            handles.push(handle);
        }

        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }

        // 验证索引状态
        let stats = index.stats();
        println!("Final stats: {:?}", stats);

        // 索引应该仍然可用
        let results = index.search_f32(&[5.0, 5.0, 5.0], 5);
        assert!(results.is_ok());
    }
}
