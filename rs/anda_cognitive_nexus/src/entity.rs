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

/// Represents a concept entity in the cognitive nexus.
///
/// A concept is a fundamental unit of knowledge that represents an idea, object, or category.
/// It contains identifying information, type classification, and associated attributes and metadata.
#[derive(Debug, Clone, Default, Deserialize, Serialize, AndaDBSchema)]
pub struct Concept {
    /// Unique identifier for the concept
    pub _id: u64,
    /// Type classification of the concept
    pub r#type: String,
    /// Human-readable name of the concept
    pub name: String,
    /// Key-value attributes associated with the concept
    pub attributes: Map<String, Json>,
    /// Additional metadata for the concept
    pub metadata: Map<String, Json>,
}

impl Concept {
    /// Returns the entity ID for this concept.
    ///
    /// # Returns
    ///
    /// An `EntityID::Concept` variant containing the concept's ID.
    pub fn entity_id(&self) -> EntityID {
        EntityID::Concept(self._id)
    }

    /// Converts the concept to a JSON representation as a concept node reference.
    ///
    /// # Returns
    ///
    /// A JSON object representing the concept node with references to its fields.
    pub fn to_concept_node(&self) -> Json {
        json!(EntityRef::ConceptNode(ConceptNodeRef {
            id: self.entity_id().to_string().as_str(),
            r#type: &self.r#type,
            name: &self.name,
            attributes: &self.attributes,
            metadata: &self.metadata,
        }))
    }

    /// Consumes the concept and converts it into a concept node.
    ///
    /// # Returns
    ///
    /// A `ConceptNode` containing the owned data from this concept.
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

/// Represents a proposition entity that defines relationships between concepts.
///
/// A proposition establishes connections between a subject and object through predicates,
/// allowing for the representation of complex relationships in the knowledge graph.
#[derive(Debug, Clone, Deserialize, Serialize, AndaDBSchema)]
pub struct Proposition {
    /// Unique identifier for the proposition
    pub _id: u64,

    /// The subject ID in the proposition relationship
    #[field_type = "Text"]
    pub subject: EntityID,

    /// The object ID in the proposition relationship
    #[field_type = "Text"]
    pub object: EntityID,

    /// Set of predicates that define the relationship types
    pub predicates: BTreeSet<String>,

    /// Properties associated with each predicate
    pub properties: BTreeMap<String, Properties>,
}

impl Proposition {
    /// Returns the entity ID for this proposition with a specific predicate.
    ///
    /// # Arguments
    ///
    /// * `predicate` - The predicate string to include in the entity ID
    ///
    /// # Returns
    ///
    /// An `EntityID::Proposition` variant containing the proposition ID and predicate.
    pub fn entity_id(&self, predicate: String) -> EntityID {
        EntityID::Proposition(self._id, predicate)
    }

    /// Converts the proposition to a JSON representation as a proposition link for a specific predicate.
    ///
    /// # Arguments
    ///
    /// * `predicate` - The predicate to create the link for
    ///
    /// # Returns
    ///
    /// An optional JSON object representing the proposition link, or `None` if the predicate doesn't exist.
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

/// Properties container for storing attributes and metadata.
///
/// This structure provides a standardized way to store key-value pairs
/// for both attributes and metadata, with compact serialization.
#[derive(Debug, Clone, Default, Deserialize, Serialize, FieldTyped)]
pub struct Properties {
    #[serde(rename = "a")]
    pub attributes: Map<String, Json>,

    #[serde(rename = "m")]
    pub metadata: Map<String, Json>,
}

impl From<Properties> for FieldValue {
    /// Converts Properties into a FieldValue for database storage.
    ///
    /// # Arguments
    ///
    /// * `value` - The Properties instance to convert
    ///
    /// # Returns
    ///
    /// A `FieldValue::Map` containing the attributes and metadata.
    fn from(value: Properties) -> Self {
        FieldValue::Map(BTreeMap::from([
            ("a".to_string(), value.attributes.into()),
            ("m".to_string(), value.metadata.into()),
        ]))
    }
}

/// Unique identifier for entities in the cognitive nexus.
///
/// EntityID provides a type-safe way to reference different types of entities,
/// with distinct formats for concepts and propositions.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityID {
    /// Identifier for a concept entity
    Concept(u64),
    /// Identifier for a proposition entity with a specific predicate
    Proposition(u64, String),
}

impl fmt::Display for EntityID {
    /// Formats the EntityID as a string.
    ///
    /// # Format
    ///
    /// * Concepts: "C:{id}"
    /// * Propositions: "P:{id}:{predicate}"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityID::Concept(id) => write!(f, "C:{}", id),
            EntityID::Proposition(id, pred) => write!(f, "P:{}:{}", id, pred),
        }
    }
}

impl FromStr for EntityID {
    type Err = String;

    /// Parses a string into an EntityID.
    ///
    /// # Arguments
    ///
    /// * `s` - String to parse, expected format: "C:{id}" or "P:{id}:{predicate}"
    ///
    /// # Returns
    ///
    /// Result containing the parsed EntityID or an error message.
    ///
    /// # Errors
    ///
    /// Returns an error if the string doesn't match the expected format.
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

    /// Attempts to convert a string slice into an EntityID.
    ///
    /// # Arguments
    ///
    /// * `value` - String slice to convert
    ///
    /// # Returns
    ///
    /// Result containing the parsed EntityID or an error message.
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        EntityID::from_str(value)
    }
}

impl Serialize for EntityID {
    /// Serializes the EntityID as a string.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// Visitor for deserializing EntityID from strings.
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
    /// Deserializes an EntityID from a string.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(EntityIDVisitor)
    }
}

/// Information about a knowledge domain.
///
/// Contains metadata about a domain including its key concept types and proposition types.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DomainInfo {
    /// Name of the domain
    pub domain_name: String,
    /// Description of the domain
    pub description: String,
    /// Key concept types in this domain
    pub key_concept_types: Vec<ConceptTypeInfo>,
    /// Key proposition types in this domain
    pub key_proposition_types: Vec<PropositionTypeInfo>,
}

impl DomainInfo {
    /// Creates domain information from a domain concept.
    ///
    /// # Arguments
    ///
    /// * `domain` - The concept representing the domain
    ///
    /// # Returns
    ///
    /// A new `DomainInfo` instance with basic information extracted from the concept.
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

/// Information about a concept type.
///
/// Describes a category of concepts including its key instances.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ConceptTypeInfo {
    /// Name of the concept type
    pub type_name: String,
    /// Description of the concept type
    pub description: String,
    /// Key instances of this concept type
    pub key_instances: Vec<String>,
}

impl ConceptTypeInfo {
    /// Creates concept type information from a concept.
    ///
    /// # Arguments
    ///
    /// * `concept` - The concept representing the type
    ///
    /// # Returns
    ///
    /// A new `ConceptTypeInfo` instance with information extracted from the concept.
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

/// Information about a proposition type.
///
/// Describes a category of propositions defined by their predicate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PropositionTypeInfo {
    /// Name of the predicate
    pub predicate_name: String,
    /// Description of the proposition type
    pub description: String,
}

impl PropositionTypeInfo {
    /// Creates proposition type information from a concept.
    ///
    /// # Arguments
    ///
    /// * `concept` - The concept representing the proposition type
    ///
    /// # Returns
    ///
    /// A new `PropositionTypeInfo` instance with information extracted from the concept.
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
