use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use anda_db_btree::RangeQuery;
pub use anda_db_schema::{Fv, bf16};
pub use anda_db_tfs::BM25Params;

/// A query for searching the database
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Query {
    // 在 field 中进行全文本搜索：(field name, search term, Option<BM25Params>)
    // field 需要建立 TFS 索引
    pub search: Option<Search>,

    /// 用 field 进行范围过滤
    /// field 需要建立 B-Tree 索引
    pub filter: Option<Filter>,

    // default to 10
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Search {
    pub field: String,
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub bm25_params: Option<BM25Params>,
    pub reranker: Option<RRFReranker>,
    pub logical_search: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// 单一字段过滤条件 (字段名, 范围查询)
    Field((String, RangeQuery<Fv>)),

    /// A logical OR filter that requires at least one subfilter to match
    Or(Vec<Box<Filter>>),

    /// A logical AND filter that requires all subfilters to match
    And(Vec<Box<Filter>>),

    /// A logical NOT filter that negates the result of its subfilter
    Not(Box<Filter>),
}

/// Reranks the results using Reciprocal Rank Fusion(RRF) algorithm based
/// on the scores of vector and FTS search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RRFReranker {
    pub k: f32,
}

impl Default for RRFReranker {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

impl RRFReranker {
    /// Rerank results using RRF.
    ///
    /// # Arguments
    /// * `ranked_lists` - 一个包含多个排序列表的切片，每个列表是 doc_id 的有序 Vec。
    ///
    /// # Returns
    /// 返回一个 Vec<(doc_id, score)>，按融合分数降序排列。
    pub fn rerank(&self, ranked_lists: &[Vec<u64>]) -> Vec<(u64, f32)> {
        let mut scores: HashMap<u64, f32> = HashMap::new();

        for ranked in ranked_lists {
            for (rank, &doc_id) in ranked.iter().enumerate() {
                let score = 1.0 / (self.k + rank as f32);
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        let mut results: Vec<(u64, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}
