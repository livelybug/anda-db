use dashmap::{DashMap, DashSet};
use half::bf16;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::{distr::Uniform, prelude::*, rng};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub use half;

/// Returns the current unix timestamp in milliseconds.
#[inline]
pub fn unix_ms() -> u64 {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    ts.as_millis() as u64
}

#[derive(Error, Debug)]
pub enum HnswError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CBOR serialization error: {0}")]
    Cbor(String),
    #[error("Vector dimension mismatch")]
    DimensionMismatch,
    #[error("Index is empty")]
    EmptyIndex,
    #[error("Invalid ID {0}")]
    InvalidId(u64),
    #[error("Distance metric error: {0}")]
    DistanceMetric(String),
    #[error("Index is full")]
    IndexFull,
    #[error("Compression error: {0}")]
    Compression(String),
    #[error("Operation cancelled")]
    Cancelled,
    #[error("Concurrent modification detected")]
    ConcurrentModification,
}

/// 距离度量类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    InnerProduct,
    Manhattan,
}

impl DistanceMetric {
    pub fn compute(&self, a: &[bf16], b: &[bf16]) -> Result<f32, HnswError> {
        if a.len() != b.len() {
            return Err(HnswError::DimensionMismatch);
        }

        let result = match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
            DistanceMetric::Cosine => Ok(cosine_distance(a, b)),
            DistanceMetric::InnerProduct => Ok(inner_product(a, b)),
            DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
        };
        result
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

/// HNSW 配置参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// 最大层数
    pub max_layers: u8,
    /// 每层最大连接数
    pub max_connections: u8,
    /// 搜索时的扩展因子
    pub ef_construction: usize,
    /// 搜索时的候选数量
    pub ef_search: usize,
    /// 距离度量类型
    pub distance_metric: DistanceMetric,
    /// 最大容量 (None 表示无限制)
    pub max_elements: Option<u64>,
    /// 缩放因子，用于调整层级分布
    pub scale_factor: Option<f64>,
}

impl HnswConfig {
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
        }
    }
}

/// HNSW 图节点
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    id: u64,
    layer: u8,
    vector: Vec<bf16>,
    neighbors: Vec<Vec<(u64, bf16)>>, // 每层的邻居列表
}

/// 索引元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMetadata {
    version: u16,
    created_at: u64,
    last_modified: u64,
    stats: IndexStats,
}

/// 索引统计信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStats {
    pub max_layer: u8,
    pub num_elements: u64,
    pub num_deleted: u64,
    pub avg_connections: f32,
    pub search_count: u64,
    pub insert_count: u64,
    pub delete_count: u64,
}

/// 序列化的HNSW索引结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswIndexSerdeOwn {
    config: HnswConfig,
    nodes: Vec<HnswNode>,
    entry_point: (u64, u8),
    dimension: usize,
    metadata: IndexMetadata,
    deleted_ids: BTreeSet<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct HnswIndexSerdeRef<'a> {
    config: &'a HnswConfig,
    nodes: &'a Vec<HnswNode>,
    entry_point: (u64, u8),
    dimension: usize,
    metadata: &'a IndexMetadata,
    deleted_ids: &'a Vec<u64>,
}

impl From<HnswIndexSerdeOwn> for HnswIndex {
    fn from(val: HnswIndexSerdeOwn) -> Self {
        let layer_gen = val.config.layer_gen();

        HnswIndex {
            dimension: val.dimension,
            config: val.config,
            layer_gen,
            nodes: DashMap::from_iter(val.nodes.into_iter().map(|node| (node.id, node))),
            entry_point: RwLock::new(val.entry_point),
            metadata: RwLock::new(val.metadata),
            deleted_ids: DashSet::from_iter(val.deleted_ids.into_iter()),
            is_dirty: RwLock::new(false),
        }
    }
}

impl From<HnswIndex> for HnswIndexSerdeOwn {
    fn from(val: HnswIndex) -> Self {
        HnswIndexSerdeOwn {
            config: val.config,
            nodes: val.nodes.iter().map(|node| node.value().clone()).collect(),
            entry_point: val.entry_point.read().clone(),
            dimension: val.dimension,
            metadata: val.metadata.read().clone(),
            deleted_ids: val.deleted_ids.iter().map(|item| item.clone()).collect(),
        }
    }
}

/// HNSW 索引
#[derive(Debug)]
pub struct HnswIndex {
    dimension: usize,
    config: HnswConfig,
    layer_gen: LayerGen,
    nodes: DashMap<u64, HnswNode>,
    entry_point: RwLock<(u64, u8)>, // 入口点
    metadata: RwLock<IndexMetadata>,
    deleted_ids: DashSet<u64>,
    is_dirty: RwLock<bool>,
}

impl HnswIndex {
    /// 创建一个新的 HNSW 索引
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

    pub fn insert_f32(&self, id: u64, vector: Vec<f32>) -> Result<bool, HnswError> {
        self.insert(id, vector.into_iter().map(|v| bf16::from_f32(v)).collect())
    }

    /// 添加向量到索引
    pub fn insert(&self, id: u64, vector: Vec<bf16>) -> Result<bool, HnswError> {
        if vector.len() != self.dimension {
            return Err(HnswError::DimensionMismatch);
        }

        // 检查容量
        if let Some(max) = self.config.max_elements {
            let nodes = self.nodes.len();
            if (nodes - self.deleted_ids.len()) as u64 >= max {
                return Err(HnswError::IndexFull);
            }
        }

        let mut distance_cache = HashMap::new();
        let mut entry_point_dist = f32::INFINITY;
        let (mut entry_point_node, current_max_layer) = self.entry_point.read().clone();

        // 随机确定节点的层数
        let layer = self.layer_gen.generate();
        let layer = if layer > current_max_layer {
            (self.config.max_layers - 1).min(current_max_layer + 1)
        } else {
            layer
        };

        // 创建新节点
        let mut node = HnswNode {
            id,
            layer,
            vector,
            neighbors: vec![Vec::new(); layer as usize + 1],
        };

        // 如果是第一个节点，设为入口点
        {
            if self.nodes.is_empty() {
                *self.entry_point.write() = (id, layer);
                self.nodes.insert(id, node);
                self.update_metadata(|m| {
                    m.stats.max_layer = layer;
                    m.stats.insert_count += 1;
                });
                *self.is_dirty.write() = true;
                return Ok(true);
            }
        }

        // 从最高层向下搜索
        for current_layer in (current_max_layer.min(layer + 1)..=current_max_layer).rev() {
            // 在当前层找到最近的邻居
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

        // 从底层开始构建连接
        for current_layer in 0..=layer {
            // 在当前层找到最近的邻居
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
            let truncate_len = (max_connections as f64 * 1.5) as usize;

            // 连接新节点到最近的邻居
            for &(neighbor, _) in &nearest {
                if neighbor != id {
                    if let Some(mut neighbor) = self.nodes.get_mut(&neighbor) {
                        let dist = self.get_distance_with_cache(
                            &mut distance_cache,
                            &node.vector,
                            &neighbor,
                        )?;
                        let neighbor_id = neighbor.id;
                        let dist = bf16::from_f32(dist);
                        node.neighbors[current_layer as usize].push((neighbor_id, dist));
                        node.neighbors[current_layer as usize]
                            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        if let Some(n_layer) = neighbor.neighbors.get_mut(current_layer as usize) {
                            n_layer.push((id, dist));
                            n_layer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                            // 修剪过多的连接
                            if n_layer.len() > truncate_len {
                                // if log::log_enabled!(log::Level::Debug) {
                                //     log::debug!(
                                //         "Trimming {} neighbors for node {} at layer {}",
                                //         n_layer.len() - max_connections,
                                //         neighbor_id,
                                //         current_layer
                                //     );
                                // }
                                n_layer.truncate(max_connections);
                            }
                        }
                    }
                }
            }
        }

        // 添加节点到索引
        self.nodes.insert(id, node);

        // 更新入口点（如果新节点在更高层）
        if layer >= current_max_layer {
            *self.entry_point.write() = (id, layer);
        }

        self.update_metadata(|m| {
            if layer > m.stats.max_layer {
                m.stats.max_layer = layer;
            }
            m.stats.insert_count += 1;
        });

        *self.is_dirty.write() = true;
        Ok(true)
    }

    pub fn search_f32(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        self.search(
            &query.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
            k,
        )
    }

    /// 搜索最近的k个邻居
    /// 按照临近距离升序
    pub fn search(&self, query: &[bf16], k: usize) -> Result<Vec<(u64, f32)>, HnswError> {
        if query.len() != self.dimension {
            return Err(HnswError::DimensionMismatch);
        }

        if self.nodes.is_empty() {
            return Err(HnswError::EmptyIndex);
        }

        let mut distance_cache = HashMap::new();
        let mut current_dist = f32::INFINITY;
        let (mut current_node, current_max_layer) = self.entry_point.read().clone();
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

    /// 在特定层搜索最近的邻居
    /// 按照临近距离升序
    fn search_layer(
        &self,
        query: &[bf16],
        entry_point: u64,
        layer: u8,
        ef: usize,
        distance_cache: &mut HashMap<u64, f32>,
    ) -> Result<Vec<(u64, f32)>, HnswError> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<(Reverse<OrderedFloat<f32>>, u64)> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, u64)> = BinaryHeap::new();

        // 计算入口点距离

        let entry_dist = match self.nodes.get(&entry_point) {
            Some(node) => self.get_distance_with_cache(distance_cache, query, &node)?,
            None => return Err(HnswError::InvalidId(entry_point)),
        };

        // 初始化候选列表
        visited.insert(entry_point);
        candidates.push((Reverse(OrderedFloat(entry_dist)), entry_point));
        results.push((OrderedFloat(entry_dist), entry_point));

        // 获取最近的候选
        while let Some((Reverse(OrderedFloat(dist)), point)) = candidates.pop() {
            let (OrderedFloat(max_dist), _) = results.peek().unwrap();
            if max_dist < &dist && results.len() >= ef {
                return Ok(results
                    .into_sorted_vec()
                    .into_iter()
                    .map(|(d, id)| (id, d.0))
                    .collect());
            };

            // 检查当前节点的邻居
            if let Some(node) = self.nodes.get(&point) {
                if let Some(neighbors) = node.neighbors.get(layer as usize) {
                    for &(neighbor, _) in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            if let Some(neighbor) = self.nodes.get(&neighbor) {
                                match self.get_distance_with_cache(distance_cache, query, &neighbor)
                                {
                                    Ok(dist) => {
                                        let (OrderedFloat(max_dist), _) = results.peek().unwrap();
                                        if &dist < max_dist || results.len() < ef {
                                            candidates
                                                .push((Reverse(OrderedFloat(dist)), neighbor.id));
                                            results.push((OrderedFloat(dist), neighbor.id));

                                            // 修剪远距离的结果
                                            if results.len() > ef {
                                                results.pop();
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        distance_cache.insert(neighbor.id, f32::INFINITY);
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

    /// 获取缓存的距离
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

    /// 保存索引到文件
    pub fn save<W: Write>(&self, w: W) -> Result<(), HnswError> {
        {
            let mut metadata = self.metadata.write();
            metadata.last_modified = unix_ms();
        }

        let mut w = zstd::Encoder::new(w, 0)?;
        ciborium::into_writer(
            &HnswIndexSerdeRef {
                config: &self.config,
                nodes: &self.nodes.iter().map(|node| node.clone()).collect(),
                entry_point: self.entry_point.read().clone(),
                dimension: self.dimension,
                metadata: &self.metadata.read(),
                deleted_ids: &self.deleted_ids.iter().map(|item| item.clone()).collect(),
            },
            &mut w,
        )
        .map_err(|e| HnswError::Cbor(e.to_string()))?;
        let _ = w.finish()?;

        *self.is_dirty.write() = false;
        Ok(())
    }

    pub fn load<R: Read>(r: R) -> Result<Self, HnswError> {
        let zr = zstd::Decoder::new(r)?;
        let index: HnswIndexSerdeOwn =
            ciborium::from_reader(zr).map_err(|e| HnswError::Cbor(e.to_string()))?;
        let index: HnswIndex = index.into();
        *index.is_dirty.write() = false;
        Ok(index)
    }

    /// 获取索引中的向量数量
    pub fn len(&self) -> usize {
        let nodes = self.nodes.len();
        nodes - self.deleted_ids.len()
    }

    /// 检查索引是否为空
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 获取向量维度
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 删除指定ID的向量
    pub fn remove(&self, id: u64) -> Result<bool, HnswError> {
        let mut deleted = false;

        self.deleted_ids.insert(id);
        if let Some((_, node)) = self.nodes.remove(&id) {
            deleted = true;
            self.try_update_entry_point(&node)?;
            self.update_metadata(|m| {
                m.stats.num_deleted += 1;
                m.stats.delete_count += 1;
            });

            // 遍历所有节点，删除与已删除节点的连接
            self.nodes.iter_mut().for_each(|mut n| {
                for layer in 0..=(n.layer as usize) {
                    n.neighbors[layer].retain(|&(neighbor, _)| neighbor != id);
                }
            });

            *self.is_dirty.write() = true;
        }

        Ok(deleted)
    }

    fn try_update_entry_point(&self, deleted_node: &HnswNode) -> Result<(), HnswError> {
        let (_, mut max_layer) = {
            let point = self.entry_point.read();
            if point.0 != deleted_node.id {
                return Ok(());
            }
            point.clone()
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

    /// 获取索引统计信息
    pub fn stats(&self) -> IndexStats {
        let mut stats = self.metadata.read().stats.clone();
        stats.num_elements = self.nodes.len() as u64;
        stats.num_deleted = self.deleted_ids.len() as u64;

        // 计算平均连接数
        let total_connections: u64 = self
            .nodes
            .iter()
            .map(|n| n.neighbors.iter().map(|v| v.len() as u64).sum::<u64>())
            .sum();

        let active_nodes = self.nodes.len() - self.deleted_ids.len();
        stats.avg_connections = if active_nodes > 0 {
            total_connections as f32 / active_nodes as f32
        } else {
            0.0
        };

        stats
    }

    /// 更新元数据
    fn update_metadata<F>(&self, f: F)
    where
        F: FnOnce(&mut IndexMetadata),
    {
        let mut metadata = self.metadata.write();
        f(&mut metadata);
    }
}

// 随机层级生成器
//
// 提供 HNSW 算法中的随机层级生成功能
// 基于指数分布生成层级，确保上层节点稀疏，底层节点密集
// 根据指数分布生成层级，层级范围为 [0..max_level)
// 较高层级的生成概率呈指数衰减
#[derive(Debug)]
pub struct LayerGen {
    /// 均匀分布采样器
    uniform: Uniform<f64>,
    /// 指数分布的缩放因子，控制层级分布
    scale: f64,
    /// 最大层级（不包含）
    max_level: u8,
}

impl LayerGen {
    /// 创建新的层级生成器
    ///
    /// # 参数
    /// * `max_connections` - 每个节点的最大连接数
    /// * `max_level` - 最大层级数（不包含）
    pub fn new(max_connections: u8, max_level: u8) -> Self {
        Self::new_with_scale(max_connections, 1.0, max_level)
    }

    /// 使用自定义缩放因子创建层级生成器
    ///
    /// # 参数
    /// * `max_connections` - 每个节点的最大连接数
    /// * `scale_factor` - 缩放因子，用于调整层级分布
    /// * `max_level` - 最大层级数（不包含）
    pub fn new_with_scale(max_connections: u8, scale_factor: f64, max_level: u8) -> Self {
        let base_scale = 1.0 / (max_connections as f64).ln();
        LayerGen {
            uniform: Uniform::<f64>::new(0.0, 1.0).unwrap(),
            scale: base_scale * scale_factor,
            max_level,
        }
    }

    /// 生成随机层级
    ///
    /// 根据指数分布生成一个 [0..max_level) 范围内的层级
    pub fn generate(&self) -> u8 {
        let mut r = rng();
        let val = r.sample(self.uniform);

        // 使用指数分布计算层级
        let level = (-val.ln() * self.scale).floor() as usize;

        // 确保层级在有效范围内
        if level >= self.max_level as usize {
            // 超出范围时重新采样（概率极低）
            r.sample(Uniform::<usize>::new(1, self.max_level as usize).unwrap()) as u8
        } else {
            level as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_distribution() {
        let lg = LayerGen::new(10, 16);
        let mut counts = vec![0; 16];

        // 生成大量样本以验证分布
        const SAMPLES: usize = 100_000;
        for _ in 0..SAMPLES {
            let level = lg.generate() as usize;
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

    #[test]
    fn test_hnsw_basic() {
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
        let results = index.search_f32(&vec![1.1, 1.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].1 < results[1].1);

        println!("Search results: {:?}", results);

        // 测试持久化
        let mut data = Vec::new();
        index.save(&mut data).unwrap();
        println!("Serialized data size: {}", data.len());

        let loaded_index = HnswIndex::load(&data[..]).unwrap();

        println!("Loaded index stats: {:?}", loaded_index.stats());
        let loaded_results = loaded_index.search_f32(&vec![1.1, 1.1], 2).unwrap();
        assert_eq!(results, loaded_results);
    }

    #[test]
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
}
