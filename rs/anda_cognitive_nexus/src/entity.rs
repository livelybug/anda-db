use anda_db_schema::{AndaDBSchema, FieldEntry, FieldType, FieldTyped, Json, Schema, SchemaError};
use anda_kip::Map;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "_type")]
pub enum Entity {
    Concept(Concept),
    Proposition(Proposition),
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct Concept {
    pub _id: u64,
    pub r#type: String,
    pub name: String,
    pub attributes: Map<String, Json>,
    pub metadata: Map<String, Json>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct Proposition {
    pub _id: u64,
    pub subject: String,
    pub object: String,
    pub predicates: BTreeSet<String>,
    pub properties: HashMap<String, Properties>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, FieldTyped)]
pub struct Properties {
    #[serde(rename = "a")]
    pub attributes: Map<String, Json>,
    #[serde(rename = "m")]
    pub metadata: Map<String, Json>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_node_schema() {
        let schema = Concept::schema().unwrap();
        assert_eq!(schema.len(), 5);
        assert_eq!(schema.get_field("_id").unwrap().r#type(), &FieldType::U64);
        assert_eq!(schema.get_field("type").unwrap().r#type(), &FieldType::Text);
        assert_eq!(schema.get_field("name").unwrap().r#type(), &FieldType::Text);
        assert_eq!(
            schema.get_field("attributes").unwrap().r#type(),
            &FieldType::Map(std::collections::BTreeMap::from([(
                "*".to_string(),
                FieldType::Json
            )]))
        );
        assert_eq!(
            schema.get_field("metadata").unwrap().r#type(),
            &FieldType::Map(std::collections::BTreeMap::from([(
                "*".to_string(),
                FieldType::Json
            )]))
        );
    }

    #[test]
    fn test_proposition_link_schema() {
        let schema = Proposition::schema().unwrap();
        assert_eq!(schema.len(), 5);
        assert_eq!(schema.get_field("_id").unwrap().r#type(), &FieldType::U64);
        assert_eq!(
            schema.get_field("subject").unwrap().r#type(),
            &FieldType::Text
        );
        assert_eq!(
            schema.get_field("object").unwrap().r#type(),
            &FieldType::Text
        );
        assert_eq!(
            schema.get_field("predicates").unwrap().r#type(),
            &FieldType::Array(vec![FieldType::Text])
        );
        assert_eq!(
            schema.get_field("properties").unwrap().r#type(),
            &FieldType::Map(std::collections::BTreeMap::from([(
                "*".to_string(),
                FieldType::Map(std::collections::BTreeMap::from([
                    (
                        "a".to_string(),
                        FieldType::Map(std::collections::BTreeMap::from([(
                            "*".to_string(),
                            FieldType::Json
                        )]))
                    ),
                    (
                        "m".to_string(),
                        FieldType::Map(std::collections::BTreeMap::from([(
                            "*".to_string(),
                            FieldType::Json
                        )]))
                    )
                ]))
            )]))
        );
    }
}
