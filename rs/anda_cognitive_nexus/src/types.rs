use anda_kip::*;
use parking_lot::RwLock;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
    str::FromStr,
    sync::Arc,
};

use crate::entity::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConceptPK {
    ID(u64),
    Object { r#type: String, name: String },
}

impl fmt::Display for ConceptPK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // `{id: "<id>"}`
            ConceptPK::ID(id) => write!(f, "{{id: {:?}}}", EntityID::Concept(*id)),
            // `{type: "<type>", name: "<name>"}`
            ConceptPK::Object { r#type, name } => {
                write!(f, "{{type: {:?}, name: {:?}}}", r#type, name)
            }
        }
    }
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

impl fmt::Display for PropositionPK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // `(id: "<link_id>")`
            PropositionPK::ID(id, predicate) => write!(
                f,
                "(id: {:?})",
                EntityID::Proposition(*id, predicate.clone()),
            ),
            // `(?subject, "<predicate>", ?object)`
            PropositionPK::Object {
                subject,
                predicate,
                object,
            } => write!(f, "({}, {:?}, {})", subject, predicate, object),
        }
    }
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityPK {
    Concept(ConceptPK),
    Proposition(PropositionPK),
}

impl fmt::Display for EntityPK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityPK::Concept(pk) => write!(f, "{}", pk),
            EntityPK::Proposition(pk) => write!(f, "{}", pk),
        }
    }
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

impl From<EntityID> for EntityPK {
    fn from(value: EntityID) -> Self {
        match value {
            EntityID::Concept(id) => EntityPK::Concept(ConceptPK::ID(id)),
            EntityID::Proposition(id, pred) => EntityPK::Proposition(PropositionPK::ID(id, pred)),
        }
    }
}

// 查询执行上下文
#[derive(Clone, Debug, Default)]
pub struct QueryContext {
    // 变量名到实体ID的映射
    pub entities: HashMap<String, HashSet<EntityID>>,
    // 变量名到谓词的映射
    pub predicates: HashMap<String, HashSet<String>>,
    pub cache: Arc<QueryCache>,
}

#[derive(Debug, Default)]
pub struct QueryCache {
    pub concepts: RwLock<HashMap<u64, Concept>>,
    pub propositions: RwLock<HashMap<u64, Proposition>>,
}

#[derive(Debug)]
pub enum TargetEntities {
    Any(String),
    AnyPropositions,
    IDs(Vec<EntityID>),
}

pub struct PropositionsMatchResult {
    pub matched_propositions: HashSet<EntityID>,
    pub matched_subjects: HashSet<EntityID>,
    pub matched_objects: HashSet<EntityID>,
    pub matched_predicates: HashSet<String>,
}

impl Default for PropositionsMatchResult {
    fn default() -> Self {
        Self::new()
    }
}

impl PropositionsMatchResult {
    pub fn new() -> Self {
        Self {
            matched_propositions: HashSet::new(),
            matched_subjects: HashSet::new(),
            matched_objects: HashSet::new(),
            matched_predicates: HashSet::new(),
        }
    }

    pub fn add_match(
        &mut self,
        subject: EntityID,
        object: EntityID,
        predicates: Vec<String>,
        proposition_id: u64,
    ) {
        self.matched_subjects.insert(subject);
        self.matched_objects.insert(object);
        for pred in &predicates {
            self.matched_propositions
                .insert(EntityID::Proposition(proposition_id, pred.clone()));
        }
        self.matched_predicates.extend(predicates);
    }
}

#[derive(Clone, Debug)]
pub struct GraphPath {
    pub start: EntityID,
    pub end: EntityID,
    pub propositions: Vec<EntityID>,
    pub hops: u16,
}

pub trait Pipe<T> {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
        Self: Sized;
}

impl<T> Pipe<T> for T {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}
