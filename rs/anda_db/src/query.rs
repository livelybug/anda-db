use anda_db_btree::RangeQuery;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::schema::{Fv, Xid, bf16};

/// A query for searching the database
#[derive(Debug, Clone, Default)]
pub struct Query<'a> {
    // 在 segments 中进行全文本搜索
    // 需要启用 segments 索引
    pub text_search: Option<&'a str>,

    // 在 segments 中进行向量搜索
    // 需要启用 segments 索引
    pub vector_search: Option<&'a [bf16]>,

    // 在 field 中进行全文本搜索：(field name, search term)
    // field 需要建立 TFS 索引
    pub field_text_search: Option<(&'a str, &'a str)>,

    // 在 field 中进行向量搜索：(field name, search vector)
    // field 需要建立向量索引
    pub field_vector_search: Option<(&'a str, &'a [bf16])>,

    /// 用 field 进行过滤
    /// field 需要建立 B-Tree 索引
    pub field_filter: Option<(&'a str, &'a Fv)>,

    /// 用 field 进行范围过滤
    /// field 需要建立 B-Tree 索引
    pub field_range_filter: Option<(&'a str, RangeQuery<'a, Fv>)>,

    // default to 10
    pub limit: Option<usize>,
}
