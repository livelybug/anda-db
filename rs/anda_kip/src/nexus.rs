use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{Command, Executor, KipError, Response, Value};

pub type Metadata = HashMap<String, Value>;
pub type Attributes = HashMap<String, Value>;

/// 认知中枢：仅用于实现参考和测试
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct CognitiveNexus {
    pub nodes: HashMap<String, ConceptNode>,
    pub propositions: HashMap<String, PropositionLink>,
    // 索引以加速查找
    type_index: HashMap<String, HashSet<String>>,
    name_index: HashMap<String, HashSet<String>>,
}

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

impl CognitiveNexus {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Executor for CognitiveNexus {
    async fn execute(&self, _command: Command) -> Result<Response, KipError> {
        unimplemented!("CognitiveNexus does not support execution yet");
    }
}
