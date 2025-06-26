use serde::{Deserialize, Serialize};

use crate::ast::{Json, Map};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "$type")]
pub enum Entity {
    ConceptNode(ConceptNode),
    PropositionLink(PropositionLink),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ConceptNode {
    pub id: String,
    pub r#type: String,
    pub name: String,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub attributes: Map<String, Json>,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Json>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PropositionLink {
    pub id: String,
    pub subject: String,
    pub object: String,
    pub predicate: String,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub attributes: Map<String, Json>,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Json>,
}
