use serde::{Deserialize, Serialize};

use crate::ast::{DotPathVar, Json, Map};

/// The absolute root type of all knowledge concepts.
pub static META_CONCEPT_TYPE: &str = "$ConceptType";

/// The absolute root type of all knowledge propositions.
pub static META_PROPOSITION_TYPE: &str = "$PropositionType";

/// The agent itself: {type: "Person", name: "$self"}
pub static META_SELF_NAME: &str = "$self";

/// The system itself: {type: "System", name: "$system"}
pub static META_SYSTEM_NAME: &str = "$system";

pub static DOMAIN_TYPE: &str = "Domain";

pub static PERSON_TYPE: &str = "Person";

pub static BELONGS_TO_DOMAIN_TYPE: &str = "belongs_to_domain";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "_type")]
pub enum Entity {
    ConceptNode(ConceptNode),
    PropositionLink(PropositionLink),
}

#[derive(Debug, Serialize)]
#[serde(tag = "_type")]
pub enum EntityRef<'a> {
    ConceptNode(ConceptNodeRef<'a>),
    PropositionLink(PropositionLinkRef<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    ConceptNode,
    PropositionLink,
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

#[derive(Debug, Serialize)]
pub struct ConceptNodeRef<'a> {
    pub id: &'a str,
    pub r#type: &'a str,
    pub name: &'a str,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub attributes: &'a Map<String, Json>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub metadata: &'a Map<String, Json>,
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

#[derive(Debug, Serialize)]
pub struct PropositionLinkRef<'a> {
    pub id: &'a str,
    pub subject: &'a str,
    pub object: &'a str,
    pub predicate: &'a str,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub attributes: &'a Map<String, Json>,

    #[serde(skip_serializing_if = "Map::is_empty")]
    pub metadata: &'a Map<String, Json>,
}

pub fn validate_dot_path_var(val: &DotPathVar, et: EntityType) -> Result<(), String> {
    match val.path.len() {
        0 => Ok(()),
        1 => match val.path[0].as_str() {
            "id" | "attributes" | "metadata" => Ok(()),
            "type" | "name" if et == EntityType::ConceptNode => Ok(()),
            "subject" | "predicate" | "object" if et == EntityType::PropositionLink => Ok(()),
            _ => Err(format!(
                "Dot notation path: invalid path component '{}'",
                val.path[0]
            )),
        },
        2 => match (val.path[0].as_str(), val.path[1].as_str()) {
            ("attributes", _) | ("metadata", _) => Ok(()),
            _ => Err(format!(
                "Dot notation path: invalid path components '{}.{}'",
                val.path[0], val.path[1]
            )),
        },
        _ => Err(format!(
            "Dot notation path: too many components in path '{}'",
            val.path.join(".")
        )),
    }
}
