use half::bf16;
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
        if a.len() != b.len() {
            return Err(HnswError::DimensionMismatch {
                name: "unknown".to_string(),
                expected: a.len(),
                got: b.len(),
            });
        }

        match self {
            DistanceMetric::Euclidean => Ok(euclidean_distance(a, b)),
            DistanceMetric::Cosine => Ok(cosine_distance(a, b)),
            DistanceMetric::InnerProduct => Ok(inner_product(a, b)),
            DistanceMetric::Manhattan => Ok(manhattan_distance(a, b)),
        }
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
}
