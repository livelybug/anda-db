//! # Anda-DB HNSW Vector Search Library

use dashmap::DashMap;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use half::bf16;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeSet, BinaryHeap, HashMap, HashSet},
    sync::atomic::{self, AtomicU64},
};

pub use half;

use crate::{DistanceMetric, LayerGen, error::HnswError};

/// HNSW index for approximate nearest neighbor search.
pub struct HnswIndex {
    /// Index name
    name: String,

    /// Index configuration.
    config: HnswConfig,

    /// Layer generator for assigning layers to new nodes.
    layer_gen: LayerGen,

    /// Map of node IDs to nodes.
    nodes: DashMap<u64, HnswNode>,

    /// Entry point for search (node_id, layer)
    entry_point: RwLock<(u64, u8)>,

    /// Index metadata.
    metadata: RwLock<HnswMetadata>,

    /// Number of search operations performed.
    search_count: AtomicU64,
}

/// HNSW configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Dimensionality of vectors in the index.
    pub dimension: usize,

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
    pub max_nodes: Option<u64>,

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
            dimension: 512,
            max_layers: 16,
            max_connections: 32,
            ef_construction: 200,
            ef_search: 50,
            distance_metric: DistanceMetric::Euclidean,
            max_nodes: None,
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HnswNode {
    /// Unique identifier for the node.
    #[serde(rename = "i")]
    pub id: u64,

    /// The highest layer this node appears in.
    #[serde(rename = "l")]
    pub layer: u8,

    /// Vector data stored in bf16 format.
    #[serde(rename = "vec")]
    pub vector: Vec<bf16>,

    /// Neighbors at each layer (layer -> [(id, distance)]).
    #[serde(rename = "n")]
    pub neighbors: Vec<SmallVec<[(u64, bf16); 64]>>,

    /// Updated version for the node. It will be incremented when the node is updated.
    #[serde(rename = "v")]
    pub version: u64,
}

/// Mapping function for a node.
pub type NodeMapFn<R> = fn(&HnswNode) -> Option<R>;

/// No-op function for node mapping.
pub const NODE_NOOP_FN: NodeMapFn<()> = |_: &HnswNode| None;

/// Function to serialize a node into binary in CBOR format.
pub const NODE_SERIALIZE_FN: NodeMapFn<Vec<u8>> = |node: &HnswNode| {
    let mut buf = Vec::new();
    ciborium::into_writer(node, &mut buf).expect("Failed to serialize node");
    Some(buf)
};

/// Function to retrieve the version of a node.
pub const NODE_VERSION_FN: NodeMapFn<u64> = |node: &HnswNode| Some(node.version);

/// Index metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswMetadata {
    /// Index name
    pub name: String,

    /// Index configuration.
    pub config: HnswConfig,

    /// Index statistics.
    pub stats: HnswStats,
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HnswStats {
    /// Last insertion timestamp (unix ms).
    pub last_inserted: u64,

    /// Last deletion timestamp (unix ms).
    pub last_deleted: u64,

    /// Last saved timestamp (unix ms).
    pub last_saved: u64,

    /// Updated version for the index. It will be incremented when the index is updated.
    pub version: u64,

    /// Maximum layer in the index.
    pub max_layer: u8,

    /// Number of nodes in the index.
    pub num_nodes: u64,

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
struct HnswIndexOwned {
    pub nodes: DashMap<u64, HnswNode>,
    pub entry_point: (u64, u8),
    pub metadata: HnswMetadata,
}

/// Serializable HNSW index structure (reference version).
#[derive(Clone, Serialize)]
struct HnswIndexRef<'a> {
    nodes: &'a DashMap<u64, HnswNode>,
    entry_point: (u64, u8),
    metadata: &'a HnswMetadata,
}

impl HnswIndex {
    /// Creates a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the index
    /// * `config` - Optional HNSW configuration parameters
    ///
    /// # Returns
    ///
    /// * `HnswIndex` - New HNSW index instance
    pub fn new(name: String, config: Option<HnswConfig>) -> Self {
        let config = config.unwrap_or_default();
        let layer_gen = config.layer_gen();

        Self {
            name: name.clone(),
            config: config.clone(),
            layer_gen,
            nodes: DashMap::new(),
            entry_point: RwLock::new((0, 0)),
            metadata: RwLock::new(HnswMetadata {
                name,
                config,
                stats: HnswStats::default(),
            }),
            search_count: AtomicU64::new(0),
        }
    }

    /// Loads an index from a reader.
    ///
    /// Deserializes the index from CBOR format.
    ///
    /// # Arguments
    ///
    /// * `r` - Any type implementing the [`futures::io::AsyncRead`] trait
    ///
    /// # Returns
    ///
    /// * `Result<Self, HnswError>` - Loaded index or error.
    pub async fn load<R: AsyncRead + Unpin>(mut r: R) -> Result<Self, HnswError> {
        let data = {
            let mut buf = Vec::new();
            AsyncReadExt::read_to_end(&mut r, &mut buf)
                .await
                .map_err(|err| HnswError::Generic {
                    name: "unknown".to_string(),
                    source: err.into(),
                })?;
            buf
        };

        let index: HnswIndexOwned =
            ciborium::from_reader(&data[..]).map_err(|err| HnswError::Serialization {
                name: "unknown".to_string(),
                source: err.into(),
            })?;
        let layer_gen = index.metadata.config.layer_gen();
        let search_count = AtomicU64::new(index.metadata.stats.search_count);

        Ok(HnswIndex {
            name: index.metadata.name.clone(),
            config: index.metadata.config.clone(),
            layer_gen,
            nodes: index.nodes,
            entry_point: RwLock::new(index.entry_point),
            metadata: RwLock::new(index.metadata),
            search_count,
        })
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

    /// Returns the index name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the dimensionality of vectors in the index
    ///
    /// # Returns
    ///
    /// * `usize` - Vector dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// Returns the index metadata
    pub fn metadata(&self) -> HnswMetadata {
        let mut metadata = { self.metadata.read().clone() };
        metadata.stats.num_nodes = self.nodes.len() as u64;
        metadata.stats.search_count = self.search_count.load(atomic::Ordering::Relaxed);

        let total_connections: u64 = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|v| v.len() as u64).sum::<u64>())
            .sum();

        metadata.stats.avg_connections = if metadata.stats.num_nodes > 0 {
            total_connections as f32 / metadata.stats.num_nodes as f32
        } else {
            0.0
        };
        metadata
    }

    /// Gets current statistics about the index
    ///
    /// # Returns
    ///
    /// * `IndexStats` - Current statistics
    pub fn stats(&self) -> HnswStats {
        let mut stats = { self.metadata.read().stats.clone() };
        stats.num_nodes = self.nodes.len() as u64;
        stats.search_count = self.search_count.load(atomic::Ordering::Relaxed);

        let total_connections: u64 = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|v| v.len() as u64).sum::<u64>())
            .sum();

        stats.avg_connections = if stats.num_nodes > 0 {
            total_connections as f32 / stats.num_nodes as f32
        } else {
            0.0
        };

        stats
    }

    /// Gets all node IDs in the index.
    pub fn node_ids(&self) -> BTreeSet<u64> {
        self.nodes.iter().map(|n| n.id).collect()
    }

    /// Gets a node by ID and applies a function to it.
    pub fn get_node_with<R, F>(&self, id: u64, f: F) -> Result<Option<R>, HnswError>
    where
        F: FnOnce(&HnswNode) -> Option<R>,
    {
        self.nodes
            .get(&id)
            .map(|node| f(&node))
            .ok_or_else(|| HnswError::NotFound {
                name: self.name.clone(),
                id,
            })
    }

    /// Sets the node if it is not already present or if the version is newer.
    /// This method is only used to bootstrap the index from persistent storage.
    pub fn set_node(&self, node: HnswNode) -> bool {
        match self.nodes.entry(node.id) {
            dashmap::Entry::Occupied(mut v) => {
                let n = v.get_mut();
                if n.version < node.version {
                    *n = node;
                    return true;
                }
                false
            }
            dashmap::Entry::Vacant(v) => {
                v.insert(node);
                true
            }
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
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Ok(()) if successful, or an error.
    pub fn insert(&self, id: u64, vector: Vec<bf16>, now_ms: u64) -> Result<(), HnswError> {
        self.insert_with(id, vector, now_ms, NODE_NOOP_FN)
            .map(|_| ())
    }

    /// Inserts a vector into the index with hook function.
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
    /// * `now_ms` - Current timestamp in milliseconds
    /// * `hook` - A function that is called with the updated node after insertion. It can be used for incremental persistence.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, R)>, HnswError>` - Vector of updated node's (ID, hook result) pairs or an error.
    pub fn insert_with<R, F>(
        &self,
        id: u64,
        vector: Vec<bf16>,
        now_ms: u64,
        hook: F,
    ) -> Result<Vec<(u64, R)>, HnswError>
    where
        F: Fn(&HnswNode) -> Option<R>,
    {
        if vector.len() != self.config.dimension {
            return Err(HnswError::DimensionMismatch {
                name: self.name.clone(),
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        // Check if ID already exists.
        if self.nodes.contains_key(&id) {
            return Err(HnswError::AlreadyExists {
                name: self.name.clone(),
                id,
            });
        }

        // Check capacity
        if let Some(max) = self.config.max_nodes {
            if self.nodes.len() as u64 >= max {
                return Err(HnswError::IndexFull {
                    name: self.name.clone(),
                });
            }
        }

        let mut updated_node_hook_result: Vec<(u64, R)> = Vec::new();
        let mut distance_cache = HashMap::with_capacity(self.config.ef_construction * 2);
        let mut entry_point_dist = f32::MAX;
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
            version: 0,
        };

        // If this is the first node, set it as the entry point
        if self.nodes.is_empty() {
            *self.entry_point.write() = (id, layer);
            node.version = 1;
            if let Some(rt) = hook(&node) {
                updated_node_hook_result.push((id, rt));
            }

            self.nodes.insert(id, node);
            self.update_metadata(|m| {
                m.stats.version = 1;
                m.stats.last_inserted = now_ms;
                m.stats.max_layer = layer;
                m.stats.insert_count += 1;
            });

            return Ok(updated_node_hook_result);
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
        let mut updated_neighbors: HashSet<u64> = HashSet::new();

        #[allow(clippy::type_complexity)]
        let mut neighbors_to_truncate: Vec<(u64, u8, SmallVec<[(u64, bf16); 64]>)> =
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
                        updated_neighbors.insert(neighbor);

                        if let Some(n_layer) =
                            neighbor_node.neighbors.get_mut(current_layer as usize)
                        {
                            n_layer.push((id, dist));
                            // If over threshold, collect nodes to update later rather than updating immediately
                            if n_layer.len() >= should_truncate {
                                // Collect information for later batch processing
                                neighbors_to_truncate.push((
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
        match self.nodes.entry(id) {
            dashmap::Entry::Occupied(_) => {
                return Err(HnswError::AlreadyExists {
                    name: self.name.clone(),
                    id,
                });
            }
            dashmap::Entry::Vacant(v) => {
                if let Some(rt) = hook(&node) {
                    updated_node_hook_result.push((id, rt));
                }
                v.insert(node);
                // Update entry point if new node is at a higher layer
                if layer > current_max_layer {
                    *self.entry_point.write() = (id, layer);
                }

                self.update_metadata(|m| {
                    m.stats.version += 1;
                    m.stats.last_inserted = now_ms;
                    if layer > m.stats.max_layer {
                        m.stats.max_layer = layer;
                    }
                    m.stats.insert_count += 1;
                });
            }
        }

        // Update collected nodes after releasing all locks
        for (node_id, layer, connections) in neighbors_to_truncate {
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

        for id in updated_neighbors {
            if let Some(mut node) = self.nodes.get_mut(&id) {
                node.version += 1;
                if let Some(rt) = hook(&node) {
                    updated_node_hook_result.push((id, rt));
                }
            }
        }

        Ok(updated_node_hook_result)
    }

    /// Inserts a vector with f32 values into the index
    ///
    /// Automatically converts f32 values to bf16 for storage efficiency
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data as f32 values
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Ok(()) if successful, or an error.
    pub fn insert_f32(&self, id: u64, vector: Vec<f32>, now_ms: u64) -> Result<(), HnswError> {
        self.insert_with(
            id,
            vector.into_iter().map(bf16::from_f32).collect(),
            now_ms,
            NODE_NOOP_FN,
        )
        .map(|_| ())
    }

    /// Inserts a vector into the index with hook function.
    ///
    /// Automatically converts f32 values to bf16 for storage efficiency
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - Vector data as bf16 values
    /// * `now_ms` - Current timestamp in milliseconds
    /// * `hook` - A function that is called with the updated node after insertion. It can be used for incremental persistence.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, R)>, HnswError>` - Vector of updated node's (ID, hook result) pairs or an error.
    pub fn insert_f32_with<R, F>(
        &self,
        id: u64,
        vector: Vec<f32>,
        now_ms: u64,
        hook: F,
    ) -> Result<Vec<(u64, R)>, HnswError>
    where
        F: Fn(&HnswNode) -> Option<R>,
    {
        self.insert_with(
            id,
            vector.into_iter().map(bf16::from_f32).collect(),
            now_ms,
            hook,
        )
    }

    /// Removes a vector from the index
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to remove
    /// * `now_ms` - Current timestamp in milliseconds
    ///
    /// # Returns
    ///
    /// * `bool` - True if the node was deleted, false otherwise
    pub fn remove(&self, id: u64, now_ms: u64) -> bool {
        let (deleted, _) = self.remove_with(id, now_ms, NODE_NOOP_FN);
        deleted
    }

    /// Removes a vector from the index with hook function.
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to remove
    /// * `now_ms` - Current timestamp in milliseconds
    /// * `hook` - A function that is called with the updated neighbors after removal. It can be used for incremental persistence.
    ///
    /// # Returns
    ///
    /// * `(bool, Vec<(u64, R)>)` - Tuple containing a boolean indicating if the node was deleted and a vector of updated node's (ID, hook result) pairs.
    pub fn remove_with<R, F>(&self, id: u64, now_ms: u64, hook: F) -> (bool, Vec<(u64, R)>)
    where
        F: Fn(&HnswNode) -> Option<R>,
    {
        let mut deleted = false;

        let mut updated_node_hook_result: Vec<(u64, R)> = Vec::new();
        if let Some((_, node)) = self.nodes.remove(&id) {
            deleted = true;
            self.try_update_entry_point(&node);
            self.update_metadata(|m| {
                m.stats.version += 1;
                m.stats.last_deleted = now_ms;
                m.stats.delete_count += 1;
            });

            // 遍历所有节点，删除与已删除节点的连接
            self.nodes.iter_mut().for_each(|mut n| {
                let mut updated = false;
                for layer in 0..=(n.layer as usize) {
                    if let Some(pos) = n.neighbors[layer].iter().position(|&(idx, _)| idx == id) {
                        n.neighbors[layer].swap_remove(pos);
                        updated = true;
                    }
                }
                if updated {
                    n.version += 1;
                    if let Some(rt) = hook(&n) {
                        updated_node_hook_result.push((n.id, rt));
                    }
                }
            });
        }

        (deleted, updated_node_hook_result)
    }

    /// Searches for the k nearest neighbors to the query vector
    ///
    /// Results are sorted by ascending distance (closest first)
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `top_k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Vector of (id, distance) pairs
    pub fn search(&self, query: &[bf16], top_k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        if query.len() != self.config.dimension {
            return Err(HnswError::DimensionMismatch {
                name: self.name.clone(),
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        if self.nodes.is_empty() {
            return Err(HnswError::EmptyIndex {
                name: self.name.clone(),
            });
        }

        let mut distance_cache = HashMap::new();
        let mut current_dist = f32::MAX;
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
        let ef = self.config.ef_search.max(top_k);
        let mut results = self.search_layer(query, current_node, 0, ef, &mut distance_cache)?;
        results.truncate(top_k);

        self.search_count.fetch_add(1, atomic::Ordering::Relaxed);

        Ok(results)
    }

    /// Searches for nearest neighbors using f32 query vector
    ///
    /// Automatically converts f32 values to bf16 for distance calculations
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector as f32 values
    /// * `top_k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// * `Result<Vec<(u64, f32)>, HnswError>` - Vector of (id, distance) pairs sorted by ascending distance
    pub fn search_f32(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        self.search(
            &query.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
            top_k,
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
            None => {
                return Err(HnswError::NotFound {
                    name: self.name.clone(),
                    id: entry_point,
                });
            }
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
                                        distance_cache.insert(neighbor, f32::MAX);
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
                    let mut best_distance_improvement = f32::MIN;

                    for (i, &(cand_id, cand_dist)) in remaining.iter().enumerate() {
                        let mut min_dist_to_selected = f32::MAX;
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

    /// Stores the index without nodes to a writer.
    ///
    /// Serializes the index using CBOR format.
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`futures::io::AsyncWrite`] trait
    /// * `now_ms` - Current timestamp in milliseconds.
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error.
    pub async fn store<W: AsyncWrite + Unpin>(
        &self,
        mut w: W,
        now_ms: u64,
    ) -> Result<(), HnswError> {
        let mut buf = Vec::with_capacity(8192);
        // avoid holding the lock for a long time
        {
            self.update_metadata(|m| {
                m.stats.last_saved = now_ms.max(m.stats.last_saved);
            });

            ciborium::into_writer(
                &HnswIndexRef {
                    nodes: &DashMap::new(),
                    entry_point: *self.entry_point.read(),
                    metadata: &self.metadata.read(),
                },
                &mut buf,
            )
            .map_err(|err| HnswError::Serialization {
                name: self.name.clone(),
                source: err.into(),
            })?;
        }

        AsyncWriteExt::write_all(&mut w, &buf)
            .await
            .map_err(|err| HnswError::Generic {
                name: self.name.clone(),
                source: err.into(),
            })?;

        Ok(())
    }

    /// Stores the index with nodes to a writer.
    ///
    /// Serializes the index using CBOR format.
    ///
    /// # Arguments
    ///
    /// * `w` - Any type implementing the [`futures::io::AsyncWrite`] trait
    /// * `now_ms` - Current timestamp in milliseconds.
    ///
    /// # Returns
    ///
    /// * `Result<(), HnswError>` - Success or error.
    pub async fn store_all<W: AsyncWrite + Unpin>(
        &self,
        mut w: W,
        now_ms: u64,
    ) -> Result<(), HnswError> {
        let mut buf = Vec::with_capacity(8192);
        // avoid holding the lock for a long time
        {
            self.update_metadata(|m| {
                m.stats.last_saved = now_ms.max(m.stats.last_saved);
            });

            ciborium::into_writer(
                &HnswIndexRef {
                    nodes: &self.nodes,
                    entry_point: *self.entry_point.read(),
                    metadata: &self.metadata.read(),
                },
                &mut buf,
            )
            .map_err(|err| HnswError::Serialization {
                name: self.name.clone(),
                source: err.into(),
            })?;
        }

        AsyncWriteExt::write_all(&mut w, &buf)
            .await
            .map_err(|err| HnswError::Generic {
                name: self.name.clone(),
                source: err.into(),
            })?;

        Ok(())
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
    fn try_update_entry_point(&self, deleted_node: &HnswNode) {
        let (_, mut max_layer) = {
            let point = self.entry_point.read();
            if point.0 != deleted_node.id {
                return;
            }
            *point
        };

        loop {
            if let Some(neighbors) = deleted_node.neighbors.get(max_layer as usize) {
                for &(neighbor, _) in neighbors {
                    if let Some(neighbor_node) = self.nodes.get(&neighbor) {
                        *self.entry_point.write() = (neighbor, neighbor_node.layer);
                        return;
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
    }

    /// Updates the index metadata
    ///
    /// # Arguments
    ///
    /// * `f` - Function that modifies the metadata
    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut HnswMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hnsw_basic() {
        let config = HnswConfig {
            dimension: 2,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 添加一些二维向量
        index.insert_f32(1, vec![1.0, 1.0], 0).unwrap();
        index.insert_f32(2, vec![1.0, 2.0], 0).unwrap();
        index.insert_f32(4, vec![2.0, 2.0], 0).unwrap();
        index.insert_f32(3, vec![2.0, 1.0], 0).unwrap();
        index.insert_f32(5, vec![3.0, 3.0], 0).unwrap();
        println!("Added vectors to index.");

        let ids = index.node_ids();
        assert_eq!(ids.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);

        let data = index.get_node_with(1, NODE_SERIALIZE_FN).unwrap().unwrap();
        let node: HnswNode = ciborium::from_reader(&data[..]).unwrap();
        println!("Node data: {:?}", node);
        assert_eq!(node.vector, vec![bf16::from_f32(1.0), bf16::from_f32(1.0)]);
        assert!(!node.neighbors[0].is_empty());

        // 搜索最近的邻居
        let results = index.search_f32(&[1.1, 1.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 < results[1].1);
        println!("Search results: {:?}", results);

        // 测试持久化
        let mut data = Vec::new();
        index.store_all(&mut data, 0).await.unwrap();
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
            dimension: 2,
            distance_metric: DistanceMetric::Euclidean,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));
        index.insert_f32(1, v1.clone(), 0).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 1.4142135).abs() < 1e-6);

        // 余弦距离
        let config = HnswConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));
        index.insert_f32(1, v1.clone(), 0).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 1.0).abs() < 1e-6);

        // 内积
        let config = HnswConfig {
            dimension: 2,
            distance_metric: DistanceMetric::InnerProduct,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));
        index.insert_f32(1, v1.clone(), 0).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];

        // 曼哈顿距离
        let config = HnswConfig {
            dimension: 2,
            distance_metric: DistanceMetric::Manhattan,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));
        index.insert_f32(1, v1.clone(), 0).unwrap();
        let results = index.search_f32(&v2, 1).unwrap();
        assert!((results[0].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_index() {
        let config = HnswConfig {
            dimension: 3,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 空索引搜索应该返回错误
        let result = index.search_f32(&[1.0, 2.0, 3.0], 5);
        assert!(matches!(result, Err(HnswError::EmptyIndex { .. })));
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = HnswConfig {
            dimension: 3,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 插入维度不匹配的向量
        let result = index.insert_f32(1, vec![1.0, 2.0], 0);
        assert!(matches!(
            result,
            Err(HnswError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            })
        ));

        // 插入正确维度的向量
        index.insert_f32(1, vec![1.0, 2.0, 3.0], 0).unwrap();

        // 搜索维度不匹配的向量
        let result = index.search_f32(&[1.0, 2.0], 5);
        assert!(matches!(
            result,
            Err(HnswError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            })
        ));
    }

    #[test]
    fn test_duplicate_insert() {
        let config = HnswConfig {
            dimension: 2,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 首次插入成功
        index.insert_f32(1, vec![1.0, 2.0], 0).unwrap();

        // 重复插入同一ID应该失败
        let result = index.insert_f32(1, vec![3.0, 4.0], 0);
        assert!(matches!(
            result,
            Err(HnswError::AlreadyExists { id: 1, .. })
        ));
    }

    #[test]
    fn test_remove() {
        let config = HnswConfig {
            dimension: 2,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 添加向量
        index.insert_f32(1, vec![1.0, 1.0], 0).unwrap();
        index.insert_f32(2, vec![2.0, 2.0], 0).unwrap();
        index.insert_f32(3, vec![3.0, 3.0], 0).unwrap();

        assert_eq!(index.len(), 3);

        // 删除存在的向量
        let deleted = index.remove(2, 0);
        assert!(deleted);
        assert_eq!(index.len(), 2);

        // 删除不存在的向量
        let deleted = index.remove(4, 0);
        assert!(!deleted);

        // 搜索应该只返回剩余的向量
        let results = index.search_f32(&[1.5, 1.5], 5).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(id, _)| *id == 1 || *id == 3));
    }

    #[test]
    fn test_max_nodes() {
        let config = HnswConfig {
            dimension: 2,
            max_nodes: Some(3),
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 添加向量直到达到最大容量
        index.insert_f32(1, vec![1.0, 1.0], 0).unwrap();
        index.insert_f32(2, vec![2.0, 2.0], 0).unwrap();
        index.insert_f32(3, vec![3.0, 3.0], 0).unwrap();

        // 超出容量限制应该失败
        let result = index.insert_f32(4, vec![4.0, 4.0], 0);
        assert!(matches!(result, Err(HnswError::IndexFull { .. })));
    }

    #[test]
    fn test_select_neighbors_strategies() {
        // 测试简单策略
        let config = HnswConfig {
            dimension: 2,
            select_neighbors_strategy: SelectNeighborsStrategy::Simple,
            ..Default::default()
        };
        let simple_index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 测试启发式策略
        let config = HnswConfig {
            dimension: 2,
            select_neighbors_strategy: SelectNeighborsStrategy::Heuristic,
            ..Default::default()
        };
        let heuristic_index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 添加相同的向量到两个索引
        for i in 0..20 {
            let x = (i % 5) as f32;
            let y = (i / 5) as f32;
            simple_index.insert_f32(i, vec![x, y], 0).unwrap();
            heuristic_index.insert_f32(i, vec![x, y], 0).unwrap();
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
            let config = HnswConfig {
                dimension: 3,
                ..Default::default()
            };
            let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

            for i in 0..100 {
                let x = (i % 10) as f32;
                let y = ((i / 10) % 10) as f32;
                let z = (i / 100) as f32;
                index.insert_f32(i, vec![x, y, z], 0).unwrap();
            }

            index.store_all(&mut data, 0).await.unwrap();
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
        let config = HnswConfig {
            dimension: 2,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 初始状态
        let stats = index.stats();
        assert_eq!(stats.num_nodes, 0);
        assert_eq!(stats.insert_count, 0);
        assert_eq!(stats.search_count, 0);
        assert_eq!(stats.delete_count, 0);

        // 添加向量
        for i in 0..10 {
            index.insert_f32(i, vec![i as f32, i as f32], 0).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.num_nodes, 10);
        assert_eq!(stats.insert_count, 10);

        // 执行搜索
        for _ in 0..5 {
            index.search_f32(&[5.0, 5.0], 3).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.search_count, 5);

        // 删除向量
        index.remove(5, 0);
        index.remove(6, 0);

        let stats = index.stats();
        assert_eq!(stats.num_nodes, 8);
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
        let config = HnswConfig {
            dimension: dim,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 创建一些高维向量
        for i in 0..10 {
            let vec = vec![i as f32 / 10.0; dim];
            index.insert_f32(i, vec, 0).unwrap();
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
        let config = HnswConfig {
            dimension: 2,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

        // 添加向量
        index.insert_f32(1, vec![1.0, 1.0], 0).unwrap();

        // 获取当前入口点
        let (entry_id, _) = *index.entry_point.read();
        assert_eq!(entry_id, 1);

        // 删除入口点
        index.remove(entry_id, 0);

        // 添加新向量，应该成为新的入口点
        index.insert_f32(2, vec![2.0, 2.0], 0).unwrap();

        let (new_entry_id, _) = *index.entry_point.read();
        assert_eq!(new_entry_id, 2);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        use std::sync::Arc;
        use tokio::sync::Barrier;

        let config = HnswConfig {
            dimension: 3,
            ..Default::default()
        };
        let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));
        let index = Arc::new(index);
        let mut handles: Vec<tokio::task::JoinHandle<Result<(), HnswError>>> =
            Vec::with_capacity(10);
        let barrier = Arc::new(Barrier::new(10));

        // 添加一些初始数据
        for i in 0..10 {
            index
                .insert_f32(i, vec![i as f32, i as f32, i as f32], 0)
                .unwrap();
        }

        for t in 0..10 {
            let b = barrier.clone();
            let index_clone = Arc::clone(&index);
            // The same messages will be printed together.
            // You will NOT see any interleaving.
            handles.push(tokio::spawn(async move {
                b.wait().await;

                // 每个线程执行不同的操作
                let base_id = 100 + t * 100;

                // 插入操作
                for i in 0..10 {
                    let id = base_id + i;
                    index_clone.insert_f32(id as u64, vec![id as f32, id as f32, id as f32], 0)?;
                }

                // 搜索操作
                for _ in 0..5 {
                    let _ = index_clone.search_f32(&[t as f32, t as f32, t as f32], 5)?;
                }

                // 删除操作
                for i in 0..5 {
                    let id = base_id + i;
                    let _ = index_clone.remove(id as u64, 0);
                }
                Ok(())
            }));
        }

        futures::future::try_join_all(handles).await.unwrap();
    }
}
