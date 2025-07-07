use anda_db_schema::{
    AndaDBSchema, FieldEntry, FieldType, FieldTyped, FieldValue, Json, Map, Schema, SchemaError,
};
use anda_kip::{ConceptNode, ConceptNodeRef, EntityRef, PropositionLinkRef};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fmt,
    str::FromStr,
};

#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct Concept {
    pub _id: u64,
    pub r#type: String,
    pub name: String,
    pub attributes: Map<String, Json>,
    pub metadata: Map<String, Json>,
}

impl Concept {
    pub fn entity_id(&self) -> EntityID {
        EntityID::Concept(self._id)
    }

    pub fn to_concept_node(&self) -> Json {
        json!(EntityRef::ConceptNode(ConceptNodeRef {
            id: self.entity_id().to_string().as_str(),
            r#type: &self.r#type,
            name: &self.name,
            attributes: &self.attributes,
            metadata: &self.metadata,
        }))
    }

    pub fn into_concept_node(self) -> ConceptNode {
        ConceptNode {
            id: self.entity_id().to_string(),
            r#type: self.r#type,
            name: self.name,
            attributes: self.attributes,
            metadata: self.metadata,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Proposition {
    pub _id: u64,

    #[field_type = "Text"]
    pub subject: EntityID,

    #[field_type = "Text"]
    pub object: EntityID,

    pub predicates: BTreeSet<String>,

    pub properties: BTreeMap<String, Properties>,
}

impl Proposition {
    pub fn entity_id(&self, predicate: String) -> EntityID {
        EntityID::Proposition(self._id, predicate)
    }

    pub fn to_proposition_link(&self, predicate: &str) -> Option<Json> {
        match self.predicates.get(predicate) {
            Some(predicate) => {
                let prop = self
                    .properties
                    .get(predicate)
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| {
                        Cow::Owned(Properties {
                            attributes: Map::new(),
                            metadata: Map::new(),
                        })
                    });

                Some(json!(EntityRef::PropositionLink(PropositionLinkRef {
                    id: self.entity_id(predicate.to_string()).to_string().as_str(),
                    subject: self.subject.to_string().as_str(),
                    predicate,
                    object: self.object.to_string().as_str(),
                    attributes: &prop.attributes,
                    metadata: &prop.metadata,
                })))
            }
            None => None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, FieldTyped)]
pub struct Properties {
    #[serde(rename = "a")]
    pub attributes: Map<String, Json>,

    #[serde(rename = "m")]
    pub metadata: Map<String, Json>,
}

impl From<Properties> for FieldValue {
    fn from(value: Properties) -> Self {
        FieldValue::Map(BTreeMap::from([
            ("a".to_string(), value.attributes.into()),
            ("m".to_string(), value.metadata.into()),
        ]))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityID {
    Concept(u64),
    Proposition(u64, String),
}

impl fmt::Display for EntityID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityID::Concept(id) => write!(f, "C:{}", id),
            EntityID::Proposition(id, pred) => write!(f, "P:{}:{}", id, pred),
        }
    }
}

impl FromStr for EntityID {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(id) = s.strip_prefix("C:") {
            id.parse::<u64>()
                .map(EntityID::Concept)
                .map_err(|_| format!("Invalid Concept ID, expected format 'C:<u64>', got {s:?}"))
        } else if let Some(id) = s.strip_prefix("P:") {
            let parts: Vec<&str> = id.split(':').collect();
            if parts.len() != 2 {
                return Err(format!(
                    "Invalid Proposition ID, expected format 'P:<u64>:<predicate>', got {s:?}"
                ));
            }

            let id = parts[0].parse::<u64>().map_err(|_| {
                format!("Invalid Proposition ID, expected format 'P:<u64>:<predicate>', got {s:?}")
            })?;

            Ok(EntityID::Proposition(id, parts[1].to_string()))
        } else {
            Err(format!("EntityID must start with 'C:' or 'P:', got {s:?}"))
        }
    }
}

impl TryFrom<&str> for EntityID {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        EntityID::from_str(value)
    }
}

impl Serialize for EntityID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

struct EntityIDVisitor;

impl serde::de::Visitor<'_> for EntityIDVisitor {
    type Value = EntityID;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "a string")
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        EntityID::from_str(s).map_err(|err| E::custom(err))
    }
}

impl<'de> Deserialize<'de> for EntityID {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(EntityIDVisitor)
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DomainInfo {
    pub domain_name: String,
    pub description: String,
    pub key_concept_types: Vec<ConceptTypeInfo>,
    pub key_proposition_types: Vec<PropositionTypeInfo>,
}

impl DomainInfo {
    pub fn from(domain: &Concept) -> Self {
        Self {
            domain_name: domain.name.clone(),
            description: domain
                .attributes
                .get("description")
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .unwrap_or_default(),
            key_concept_types: Vec::new(),
            key_proposition_types: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConceptTypeInfo {
    pub type_name: String,
    pub description: String,
    pub key_instances: Vec<String>,
}

impl ConceptTypeInfo {
    pub fn from(concept: &Concept) -> Self {
        Self {
            type_name: concept.name.clone(),
            description: concept
                .attributes
                .get("description")
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .unwrap_or_default(),
            key_instances: concept
                .attributes
                .get("key_instances")
                .map(|v| {
                    v.as_array()
                        .map(|v| {
                            v.iter()
                                .map(|v| v.as_str().unwrap_or_default().to_string())
                                .collect()
                        })
                        .unwrap_or_default()
                })
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PropositionTypeInfo {
    pub predicate_name: String,
    pub description: String,
}

impl PropositionTypeInfo {
    pub fn from(concept: &Concept) -> Self {
        Self {
            predicate_name: concept.name.clone(),
            description: concept
                .attributes
                .get("description")
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity() {}

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
