use anda_db::error::DBError;
use anda_db_schema::{
    AndaDBSchema, BoxError, FieldEntry, FieldType, FieldTyped, Json, Schema, SchemaError,
};
use anda_kip::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeSet, HashMap},
    fmt,
    str::FromStr,
};

use crate::entity::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConceptPK {
    ID(u64),
    Object { r#type: String, name: String },
}

impl TryFrom<ConceptMatcher> for ConceptPK {
    type Error = KipError;

    fn try_from(value: ConceptMatcher) -> Result<Self, Self::Error> {
        match value {
            ConceptMatcher::ID(id) => {
                let id = EntityID::from_str(&id).map_err(KipError::Parse)?;
                match id {
                    EntityID::Concept(id) => Ok(ConceptPK::ID(id)),
                    _ => Err(KipError::InvalidCommand(format!(
                        "ConceptMatcher::ID must be a Concept ID, got: {id:?}"
                    ))),
                }
            }
            ConceptMatcher::Object { r#type, name } => Ok(ConceptPK::Object { r#type, name }),
            _ => Err(KipError::InvalidCommand(format!(
                "ConceptMatcher must be either ID or Object, got: {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropositionPK {
    ID(u64, String),
    Object {
        subject: Box<EntityPK>,
        predicate: String,
        object: Box<EntityPK>,
    },
}

impl TryFrom<PropositionMatcher> for PropositionPK {
    type Error = KipError;

    fn try_from(value: PropositionMatcher) -> Result<Self, Self::Error> {
        match value {
            PropositionMatcher::ID(id) => {
                let id = EntityID::from_str(&id).map_err(KipError::Parse)?;
                match id {
                    EntityID::Proposition(id, predicate) => Ok(PropositionPK::ID(id, predicate)),
                    _ => Err(KipError::InvalidCommand(format!(
                        "PropositionMatcher::ID must be a Proposition ID, got: {id:?}"
                    ))),
                }
            }
            PropositionMatcher::Object {
                subject,
                predicate,
                object,
            } => {
                let subject = Box::new(EntityPK::try_from(subject)?);
                let object = Box::new(EntityPK::try_from(object)?);
                let predicate = match predicate {
                    PredTerm::Literal(value) => value,
                    val => {
                        return Err(KipError::InvalidCommand(format!(
                            "PropositionMatcher::Object's predicate must be a literal string, got: {val:?}"
                        )));
                    }
                };

                Ok(PropositionPK::Object {
                    subject,
                    predicate,
                    object,
                })
            }
            _ => Err(KipError::InvalidCommand(format!(
                "PropositionMatcher must be either ID or Object, got: {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityPK {
    Concept(ConceptPK),
    Proposition(PropositionPK),
}

impl TryFrom<TargetTerm> for EntityPK {
    type Error = KipError;

    fn try_from(value: TargetTerm) -> Result<Self, Self::Error> {
        match value {
            TargetTerm::Concept(value) => Ok(EntityPK::Concept(ConceptPK::try_from(value)?)),
            TargetTerm::Proposition(value) => {
                Ok(EntityPK::Proposition(PropositionPK::try_from(*value)?))
            }
            _ => Err(KipError::InvalidCommand(format!(
                "TargetTerm must be either Concept or Proposition, got: {value:?}"
            ))),
        }
    }
}
