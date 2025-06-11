use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    String(String),
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Uint(i) => write!(f, "{}", i),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Null => write!(f, "null"),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Command {
    Kql(KqlQuery),
    Kml(KmlStatement),
    Meta(MetaCommand),
}

// --- Common AST Nodes ---

#[derive(Debug, PartialEq, Clone)]
pub struct KeyValue {
    pub key: String,
    pub value: Value,
}

#[derive(Debug, PartialEq, Clone)]
pub struct OnClause {
    pub keys: Vec<KeyValue>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PropObject {
    Node(OnClause),
    LocalHandle(String),
}

// --- KQL AST ---

#[derive(Debug, PartialEq, Clone)]
pub struct KqlQuery {
    pub find_clause: FindClause,
    pub where_clauses: Vec<WhereClause>,
    pub order_by: Option<Vec<OrderByCondition>>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FindClause {
    pub expressions: Vec<FindExpression>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum FindExpression {
    Variable(String),
    Aggregation {
        func: AggregationFunction,
        var: String,
        distinct: bool,
        alias: String,
    },
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum AggregationFunction {
    Count,
    Collect,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, PartialEq, Clone)]
pub enum WhereClause {
    Grounding(Grounding),
    Proposition(PropositionPattern),
    Attribute(AttributePattern),
    Filter(FilterCondition),
    Not(Vec<WhereClause>),
    Optional(Vec<WhereClause>),
    Union {
        left: Vec<WhereClause>,
        right: Vec<WhereClause>,
    },
    // For simplicity, Subquery and BIND are treated as advanced features
    // that might be handled by a more complex executor or parser extension.
}

#[derive(Debug, PartialEq, Clone)]
pub struct Grounding {
    pub variable: String,
    pub constraints: Vec<KeyValue>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum PropTerm {
    Variable(String),
    Node(OnClause),
    NestedProp(Box<PropositionPattern>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct PropositionPattern {
    pub subject: PropTerm,
    pub predicate: String, // Simplified, path expressions can be added here
    pub object: PropTerm,
    pub metadata_constraints: Option<Vec<KeyValue>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AttributePattern {
    pub node_variable: String,
    pub attribute_name: String,
    pub value_variable: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FilterCondition {
    pub expression: String, // For simplicity, we parse the expression as a string
}

#[derive(Debug, PartialEq, Clone)]
pub struct OrderByCondition {
    pub variable: String,
    pub direction: OrderDirection,
}

#[derive(Debug, PartialEq, Clone)]
pub enum OrderDirection {
    Asc,
    Desc,
}

// --- KML AST ---

#[derive(Debug, PartialEq, Clone)]
pub enum KmlStatement {
    Upsert(UpsertBlock),
    Delete(DeleteStatement),
}

#[derive(Debug, PartialEq, Clone)]
pub struct UpsertBlock {
    pub items: Vec<UpsertItem>,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UpsertItem {
    Concept(ConceptBlock),
    Proposition(PropositionBlock),
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConceptBlock {
    pub handle: String,
    pub on: OnClause,
    pub set_attributes: Option<HashMap<String, Value>>,
    pub set_propositions: Option<Vec<SetProposition>>,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct SetProposition {
    pub predicate: String,
    pub object: PropObject,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct PropositionBlock {
    pub handle: String,
    pub subject: OnClause,
    pub predicate: String,
    pub object: PropObject,
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum DeleteStatement {
    DeleteAttributes {
        attributes: Vec<String>,
        from: OnClause,
    },
    DeleteProposition {
        subject: OnClause,
        predicate: String,
        object: OnClause,
    },
    // For `DELETE PROPOSITIONS WHERE`, we can reuse the KQL WHERE parser.
    DeletePropositionsWhere {
        where_clauses: Vec<WhereClause>,
    },
    DeleteConcept {
        on: OnClause,
    },
}

// --- META AST ---

#[derive(Debug, PartialEq, Clone)]
pub enum MetaCommand {
    Describe(DescribeTarget),
    Search(SearchCommand),
}

#[derive(Debug, PartialEq, Clone)]
pub enum DescribeTarget {
    Primer,
    Domains,
    ConceptTypes,
    ConceptType(String),
    PropositionTypes,
    PropositionType(String),
}

#[derive(Debug, PartialEq, Clone)]
pub struct SearchCommand {
    pub term: String,
    pub in_type: Option<String>,
    pub limit: Option<u64>,
}
