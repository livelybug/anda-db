use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// 代表一个可以存储在节点属性或元数据中的值
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KipValue {
    String(String),
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl std::fmt::Display for KipValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KipValue::String(s) => write!(f, "\"{}\"", s),
            KipValue::Uint(i) => write!(f, "{}", i),
            KipValue::Int(i) => write!(f, "{}", i),
            KipValue::Float(fl) => write!(f, "{}", fl),
            KipValue::Bool(b) => write!(f, "{}", b),
            KipValue::Null => write!(f, "null"),
        }
    }
}

pub type Metadata = HashMap<String, KipValue>;
pub type Attributes = HashMap<String, KipValue>;

/// 概念节点：代表一个实体化的概念
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub id: String,
    pub node_type: String,
    pub attributes: Attributes,
    pub metadata: Metadata,
    pub outgoing_propositions: HashSet<String>,
    pub incoming_propositions: HashSet<String>,
}

/// 命题链接：代表一个 (主语, 谓词, 宾语) 的事实
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropositionLink {
    pub id: String,
    pub subject_id: String,
    pub predicate: String,
    pub object_id: String,
    pub metadata: Metadata,
}

/// 认知引信：为 LLM 设计的知识摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePrimer {
    pub universal_abstract: UniversalAbstract,
    pub domain_map: DomainMap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalAbstract {
    pub agent_role: String,
    pub nexus_description: String,
    pub core_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainMap {
    pub domains: Vec<String>,
    pub key_concepts: HashMap<String, Vec<String>>,
    pub key_propositions: Vec<String>,
}

/// 认知中枢：知识图谱的核心
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CognitiveNexus {
    pub nodes: HashMap<String, ConceptNode>,
    pub propositions: HashMap<String, PropositionLink>,
    // 索引以加速查找
    type_index: HashMap<String, HashSet<String>>,
    name_index: HashMap<String, HashSet<String>>, // 假设 "name" 是一个常用属性
}

impl CognitiveNexus {
    pub fn new() -> Self {
        Self::default()
    }

    /// 插入或更新一个概念节点
    pub fn upsert_node(&mut self, mut node: ConceptNode) -> String {
        // 如果节点已存在，保留其关系
        if let Some(existing_node) = self.nodes.get(&node.id) {
            node.incoming_propositions = existing_node.incoming_propositions.clone();
            node.outgoing_propositions = existing_node.outgoing_propositions.clone();
        }

        let id = node.id.clone();
        // 更新索引
        self.type_index
            .entry(node.node_type.clone())
            .or_default()
            .insert(id.clone());
        if let Some(KipValue::String(name)) = node.attributes.get("name") {
            self.name_index
                .entry(name.clone())
                .or_default()
                .insert(id.clone());
        }
        self.nodes.insert(id.clone(), node);
        id
    }

    // TODO: 实现删除节点、查询节点、更新属性等方法
}
