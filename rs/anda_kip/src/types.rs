use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ast::{Json, Map};

#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct ConceptNode {
    pub id: String,
    pub r#type: String,
    pub name: String,
    pub attributes: Map<String, Json>,
    pub metadata: Map<String, Json>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema)]
pub struct PropositionLink {
    pub id: String,
    pub subject: String,
    pub object: String,
    pub predicate: String,
    pub attributes: Map<String, Json>,
    pub metadata: Map<String, Json>,
}
