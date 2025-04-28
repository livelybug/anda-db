use half::bf16;
use ndarray::ArrayView1;
use rand::{distr::Uniform, prelude::*, rng};
use serde::{Deserialize, Serialize};

use crate::error::HnswError;

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
        // 使用线程本地缓冲区
        A_BUFFER.with_borrow_mut(|a_buf| {
            B_BUFFER.with_borrow_mut(|b_buf| {
                // 确保缓冲区足够大
                if a_buf.len() < a.len() {
                    a_buf.resize(a.len(), 0.0);
                    b_buf.resize(b.len(), 0.0);
                }

                // 转换为 f32 数组
                let a_array = convert_to_f32_array(a, a_buf);
                let b_array = convert_to_f32_array(b, b_buf);

                self.compute_f32(a_array, b_array)
            })
        })
    }

    pub fn compute_f32(&self, a: &[f32], b: &[f32]) -> Result<f32, HnswError> {
        if a.len() != b.len() {
            return Err(HnswError::DimensionMismatch {
                name: "unknown".to_string(),
                expected: a.len(),
                got: b.len(),
            });
        }

        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance_simd(a, b)),
            DistanceMetric::Cosine => Ok(cosine_distance_simd(a, b)),
            DistanceMetric::InnerProduct => Ok(inner_product_simd(a, b)),
            DistanceMetric::Manhattan => Ok(manhattan_distance_simd(a, b)),
        }
    }
}

#[inline]
fn convert_to_f32_array<'a>(input: &[bf16], buffer: &'a mut [f32]) -> &'a [f32] {
    debug_assert!(buffer.len() >= input.len());
    for (i, &val) in input.iter().enumerate() {
        buffer[i] = val.to_f32();
    }
    &buffer[..input.len()]
}

// 使用线程本地存储的缓冲区，避免频繁分配内存
thread_local! {
    static A_BUFFER: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(1024));
    static B_BUFFER: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::with_capacity(1024));
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
fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    // 边界情况处理
    if a.is_empty() {
        return 0.0;
    } else if a.len() == 1 {
        let diff = a[0] - b[0];
        return diff.abs();
    }

    let a_array = ArrayView1::from(a);
    let b_array = ArrayView1::from(b);

    // 计算欧几里德距离
    let diff = &a_array - &b_array;
    diff.dot(&diff).sqrt()
}

#[inline]
fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    // 边界情况处理
    if a.is_empty() {
        return 0.0;
    } else if a.len() == 1 {
        if a[0] == 0.0 || b[0] == 0.0 {
            return 1.0; // 零向量的余弦距离为1
        }
        return 1.0 - (a[0] * b[0]) / (a[0].abs() * b[0].abs());
    }

    let a_array = ArrayView1::from(a);
    let b_array = ArrayView1::from(b);

    // 计算点积
    let dot_product = a_array.dot(&b_array);

    // 计算向量范数
    let norm_a = a_array.dot(&a_array).sqrt();
    let norm_b = b_array.dot(&b_array).sqrt();

    // 处理零向量情况
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 1.0;
    }

    // 计算余弦距离
    1.0 - (dot_product / (norm_a * norm_b))
}

#[inline]
fn inner_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // 边界情况处理
    if a.is_empty() {
        return 0.0;
    } else if a.len() == 1 {
        return -(a[0] * b[0]);
    }

    let a_array = ArrayView1::from(a);
    let b_array = ArrayView1::from(b);

    // 计算负内积
    -a_array.dot(&b_array)
}

#[inline]
fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    // 边界情况处理
    if a.is_empty() {
        return 0.0;
    } else if a.len() == 1 {
        return (a[0] - b[0]).abs();
    }

    let a_array = ArrayView1::from(a);
    let b_array = ArrayView1::from(b);

    // 计算曼哈顿距离
    (&a_array - &b_array).mapv(|x| x.abs()).sum()
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
        assert!(bottom_ratio >= 0.5);
    }

    #[test]
    fn test_simd_vs_scalar() {
        let mut rng = rand::rng();

        fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        }

        fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                return 1.0; // 零向量的余弦距离默认为1.0
            }
            1.0 - (dot_product / (norm_a * norm_b))
        }

        fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
            -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
        }

        fn manhattan_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }

        // 创建随机测试向量
        let dims = 128;
        let mut v1: Vec<f32> = Vec::with_capacity(dims);
        let mut v2: Vec<f32> = Vec::with_capacity(dims);

        for _ in 0..dims {
            v1.push(rng.random::<f32>());
            v2.push(rng.random::<f32>());
        }

        // 测试欧几里得距离
        let simd_euclidean = euclidean_distance_simd(&v1, &v2);
        let scalar_euclidean = euclidean_distance_scalar(&v1, &v2);
        assert!(
            (simd_euclidean - scalar_euclidean).abs() < 1e-4,
            "欧几里得距离: SIMD={}, 标量={}",
            simd_euclidean,
            scalar_euclidean
        );

        // 测试余弦距离
        let simd_cosine = cosine_distance_simd(&v1, &v2);
        let scalar_cosine = cosine_distance_scalar(&v1, &v2);
        assert!(
            (simd_cosine - scalar_cosine).abs() < 1e-4,
            "余弦距离: SIMD={}, 标量={}",
            simd_cosine,
            scalar_cosine
        );

        // 测试内积
        let simd_inner = inner_product_simd(&v1, &v2);
        let scalar_inner = inner_product_scalar(&v1, &v2);
        assert!(
            (simd_inner - scalar_inner).abs() < 1e-4,
            "内积: SIMD={}, 标量={}",
            simd_inner,
            scalar_inner
        );

        // 测试曼哈顿距离
        let simd_manhattan = manhattan_distance_simd(&v1, &v2);
        let scalar_manhattan = manhattan_distance_scalar(&v1, &v2);
        assert!(
            (simd_manhattan - scalar_manhattan).abs() < 1e-4,
            "曼哈顿距离: SIMD={}, 标量={}",
            simd_manhattan,
            scalar_manhattan
        );
    }
}
