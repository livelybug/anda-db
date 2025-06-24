use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, Json, Schema, SchemaError};
use anda_kip::Map;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum Entity {
    Concept(ConceptNode),
    Proposition(PropositionLink),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct ConceptNode {
    pub _id: u64,
    pub id: String,
    pub r#type: String,
    pub name: String,

    #[field_type = "Map"]
    pub attributes: Map<String, Json>,

    #[field_type = "Map"]
    pub metadata: Map<String, Json>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct PropositionLink {
    pub _id: u64,
    pub subject: String,
    pub object: String,

    #[field_type = "Map"]
    pub predicates: BTreeSet<String>,

    #[field_type = "Map"]
    pub properties: HashMap<String, Properties>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Properties {
    #[serde(rename = "a")]
    pub attributes: Map<String, Json>,
    #[serde(rename = "m")]
    pub metadata: Map<String, Json>,
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_concept_node_schema() {
//         let c = ConceptNode::default();
//         println!("{c:#?}");
//         let schema = ConceptNode::schema().unwrap();
//         println!("{schema:#?}");
//         assert_eq!(schema.len(), 5);
//         assert!(schema.get_field("id").is_some());
//         assert!(schema.get_field("type").is_some());
//         assert!(schema.get_field("name").is_some());
//         assert!(schema.get_field("attributes").is_some());
//         assert!(schema.get_field("metadata").is_some());
//     }

//     #[test]
//     fn test_proposition_link_schema() {
//         let schema = PropositionLink::schema().unwrap();
//         println!("{schema:#?}");
//         assert_eq!(schema.len(), 6);
//         assert!(schema.get_field("subject").is_some());
//         assert!(schema.get_field("object").is_some());
//         assert!(schema.get_field("predicate").is_some());
//         assert!(schema.get_field("attributes").is_some());
//         assert!(schema.get_field("metadata").is_some());
//     }
// }
