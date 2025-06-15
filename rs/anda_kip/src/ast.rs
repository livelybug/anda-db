//! # KIP (Knowledge Interaction Protocol) AST Definitions
//!
//! This module defines the Abstract Syntax Tree (AST) structures for the Knowledge Interaction Protocol (KIP),
//! a knowledge memory interaction protocol designed for Large Language Models (LLMs) to build sustainable
//! learning and self-evolving knowledge memory systems.
//!
//! KIP defines a complete interaction pattern for efficient, reliable, bidirectional knowledge exchange
//! between the neural core (LLM) and the symbolic core (Cognitive Nexus).
//!
//! The AST is organized into three main command categories:
//! - **KQL (Knowledge Query Language)**: For knowledge retrieval and reasoning
//! - **KML (Knowledge Manipulation Language)**: For knowledge evolution and updates
//! - **META**: For knowledge exploration and grounding

use serde::{Deserialize, Serialize};

pub use serde_json::{Map, Number};

/// Represents a primitive value in the KIP system.
/// This is the fundamental data type used throughout KIP for attributes, metadata, and literals.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Value {
    /// Represents a null value
    #[default]
    Null,
    /// Boolean value (true/false)
    Bool(bool),
    /// Numeric value (integer or floating-point)
    Number(Number),
    /// String value
    String(String),
}

pub type Json = serde_json::Value;

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Number(n) => write!(f, "{n}"),
            // format as JSON string (format_escaped_str)
            Value::String(s) => write!(f, "{}", Json::String(s.clone())),
        }
    }
}

impl From<Value> for Json {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Json::Null,
            Value::Bool(b) => Json::Bool(b),
            Value::Number(n) => Json::Number(n),
            Value::String(s) => Json::String(s),
        }
    }
}


/// Top-level command enum representing the three main KIP instruction sets.
/// Each command type serves a specific purpose in the knowledge interaction workflow.
#[derive(Debug, PartialEq, Clone)]
pub enum Command {
    /// KQL (Knowledge Query Language) - for knowledge retrieval and reasoning
    Kql(KqlQuery),
    /// KML (Knowledge Manipulation Language) - for knowledge evolution and updates
    Kml(KmlStatement),
    /// META commands - for knowledge exploration and grounding
    Meta(MetaCommand),
}

// --- Common AST Nodes ---

/// Represents a key-value pair used in various contexts throughout KIP.
/// Used for attributes, metadata, constraints, and unique key specifications.
#[derive(Debug, PartialEq, Clone)]
pub struct KeyValue {
    /// The key name
    pub key: String,
    /// The associated value
    pub value: Value,
}

/// Represents an "ON" clause used for concept identification and grounding.
/// Contains unique key attributes that identify a specific concept node in the knowledge graph.
/// Example: `ON { type: "Drug", name: "Aspirin" }`
#[derive(Debug, PartialEq, Clone)]
pub struct OnClause {
    /// List of key-value pairs that uniquely identify a concept
    pub keys: Vec<KeyValue>,
}

/// Represents the object part of a proposition, which can be either:
/// - A reference to an existing concept node (via OnClause)
/// - A local handle to a concept defined within the same transaction
#[derive(Debug, PartialEq, Clone)]
pub enum PropObject {
    /// Reference to an existing concept node using unique keys
    Node(OnClause),
    /// Local handle (starting with @) referencing a concept within the same transaction
    LocalHandle(String),
}

// --- KQL AST ---

/// Represents a complete KQL (Knowledge Query Language) query.
/// KQL is responsible for knowledge retrieval and reasoning within the Cognitive Nexus.
///
/// Structure: `FIND(...) WHERE { ... } ORDER BY ... LIMIT N OFFSET M`
#[derive(Debug, PartialEq, Clone)]
pub struct KqlQuery {
    /// The FIND clause specifying what to return
    pub find_clause: FindClause,
    /// WHERE clauses containing graph patterns and filters (all ANDed together)
    pub where_clauses: Vec<WhereClause>,
    /// Optional ORDER BY conditions for result sorting
    pub order_by: Option<Vec<OrderByCondition>>,
    /// Optional LIMIT for result count restriction
    pub limit: Option<u64>,
    /// Optional OFFSET for result pagination
    pub offset: Option<u64>,
}

/// Represents the FIND clause of a KQL query.
/// Declares the final output of the query, supporting both simple variables and aggregations.
#[derive(Debug, PartialEq, Clone)]
pub struct FindClause {
    /// List of expressions to be returned (variables or aggregations)
    pub expressions: Vec<FindExpression>,
}

/// Represents an expression in the FIND clause.
/// Can be either a simple variable or an aggregation function with alias.
#[derive(Debug, PartialEq, Clone)]
pub enum FindExpression {
    /// A simple variable reference (e.g., `?drug_name`)
    Variable(String),
    /// An aggregation function with alias (e.g., `COUNT(?drug) AS ?drug_count`)
    Aggregation {
        /// The aggregation function to apply
        func: AggregationFunction,
        /// The variable to aggregate
        var: String,
        /// Whether to use DISTINCT
        distinct: bool,
        /// The alias for the result
        alias: String,
    },
}

/// Supported aggregation functions in KQL.
/// These functions operate on grouped data to produce summary statistics.
#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum AggregationFunction {
    /// COUNT(?var) - counts the number of bindings
    Count,
    /// COLLECT(?var) - collects all values into a list
    Collect,
    /// SUM(?var) - sums numeric values
    Sum,
    /// AVG(?var) - calculates average of numeric values
    Avg,
    /// MIN(?var) - finds minimum value
    Min,
    /// MAX(?var) - finds maximum value
    Max,
}

/// Represents different types of clauses in the WHERE section of a KQL query.
/// All clauses are combined with logical AND by default.
#[derive(Debug, PartialEq, Clone)]
pub enum WhereClause {
    /// Type assertion/entity grounding: `?variable(type: "...", name: "...", id: "...")`
    Grounding(Grounding),
    /// Proposition pattern: `PROP(Subject, Predicate, Object) { metadata_filter }`
    Proposition(PropositionPattern),
    /// Attribute access: `ATTR(?node, "attribute_name", ?value_variable)`
    Attribute(AttributePattern),
    /// Filter condition: `FILTER(boolean_expression)`
    Filter(FilterCondition),
    /// Negation: `NOT { ... }`
    Not(Vec<WhereClause>),
    /// Optional matching: `OPTIONAL { ... }`
    Optional(Vec<WhereClause>),
    /// Union (logical OR): `UNION { ... }`
    Union(Vec<WhereClause>),
}

/// Represents type assertion/entity grounding clause.
/// Constrains a variable to a specific type or grounds it to a specific node in the graph.
/// Example: `?drug(type: "Drug")` or `?aspirin(name: "Aspirin")`
#[derive(Debug, PartialEq, Clone)]
pub struct Grounding {
    /// The variable to be constrained
    pub variable: String,
    /// Constraints (type, name, id, etc.)
    pub constraints: Vec<KeyValue>,
}

/// Represents a term in a proposition that can be a variable, node reference, or nested proposition.
/// Used for both subject and object positions in proposition patterns.
#[derive(Debug, PartialEq, Clone)]
pub enum PropTerm {
    /// A variable (e.g., `?drug`)
    Variable(String),
    /// A specific node reference using unique keys
    Node(OnClause),
    /// A nested proposition pattern (for complex relationships)
    NestedProp(Box<PropositionPattern>),
}

/// Represents a predicate term in a proposition.
/// Can be either a variable or a literal string.
#[derive(Debug, PartialEq, Clone)]
pub enum PredTerm {
    /// A variable predicate (e.g., `?relationship`)
    Variable(String),
    /// A literal predicate string (e.g., `"treats"`)
    Literal(String),
}

/// Represents a proposition pattern for graph traversal.
/// Follows the (Subject, Predicate, Object) triple pattern with optional metadata constraints.
/// Example: `PROP(?drug, "treats", ?symptom) { confidence: ?conf }`
#[derive(Debug, PartialEq, Clone)]
pub struct PropositionPattern {
    /// The subject of the proposition
    pub subject: PropTerm,
    /// The predicate (relationship type)
    pub predicate: PredTerm,
    /// The object of the proposition
    pub object: PropTerm,
    /// Optional metadata constraints for filtering
    pub metadata_constraints: Option<Vec<KeyValue>>,
}

/// Represents an attribute access pattern.
/// Retrieves an attribute value from a concept node and binds it to a variable.
/// Example: `ATTR(?drug, "name", ?drug_name)`
#[derive(Debug, PartialEq, Clone)]
pub struct AttributePattern {
    /// The node variable to get the attribute from
    pub node_variable: String,
    /// The name of the attribute to retrieve
    pub attribute_name: String,
    /// The variable to bind the attribute value to
    pub value_variable: String,
}

/// Represents a filter condition with optional subquery.
/// Applies complex filtering logic to bound variables.
/// Example: `FILTER(?risk < 3)` or `FILTER(?count > 5) { SELECT(COUNT(?item) AS ?count) WHERE { ... } }`
#[derive(Debug, PartialEq, Clone)]
pub struct FilterCondition {
    /// The main filter expression
    pub expression: FilterExpression,
    /// Optional subquery for complex filtering
    pub subquery: Option<SubqueryExpression>,
}

/// Represents different types of filter expressions.
/// Supports comparisons, logical operations, negation, and function calls.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterExpression {
    /// Comparison operations (==, !=, <, >, <=, >=)
    Comparison {
        left: FilterOperand,
        operator: ComparisonOperator,
        right: FilterOperand,
    },
    /// Logical operations (&&, ||)
    Logical {
        left: Box<FilterExpression>,
        operator: LogicalOperator,
        right: Box<FilterExpression>,
    },
    /// Unary negation (!)
    Not(Box<FilterExpression>),
    /// Function calls (CONTAINS, STARTS_WITH, etc.)
    Function {
        func: FilterFunction,
        args: Vec<FilterOperand>,
    },
}

/// Represents an operand in a filter expression.
/// Can be either a variable reference or a literal value.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterOperand {
    /// A variable reference (e.g., `?risk`)
    Variable(String),
    /// A literal value
    Literal(Value),
}

/// Comparison operators supported in filter expressions.
#[derive(Debug, PartialEq, Clone)]
pub enum ComparisonOperator {
    /// Equality (==)
    Equal,
    /// Inequality (!=)
    NotEqual,
    /// Less than (<)
    LessThan,
    /// Greater than (>)
    GreaterThan,
    /// Less than or equal (<=)
    LessEqual,
    /// Greater than or equal (>=)
    GreaterEqual,
}

/// Logical operators for combining filter expressions.
#[derive(Debug, PartialEq, Clone)]
pub enum LogicalOperator {
    /// Logical AND (&&)
    And,
    /// Logical OR (||)
    Or,
}

/// String manipulation and pattern matching functions for filters.
#[derive(Debug, PartialEq, Clone)]
pub enum FilterFunction {
    /// CONTAINS(?str, "substring") - checks if string contains substring
    Contains,
    /// STARTS_WITH(?str, "prefix") - checks if string starts with prefix
    StartsWith,
    /// ENDS_WITH(?str, "suffix") - checks if string ends with suffix
    EndsWith,
    /// REGEX(?str, "pattern") - checks if string matches regex pattern
    Regex,
}

/// Represents a subquery expression used within filter conditions.
/// Allows nested queries for complex filtering logic.
#[derive(Debug, PartialEq, Clone)]
pub struct SubqueryExpression {
    /// The SELECT clause of the subquery (similar to FIND)
    pub select_clause: FindClause,
    /// WHERE clauses of the subquery
    pub where_clauses: Vec<WhereClause>,
}

/// Represents an ORDER BY condition for result sorting.
#[derive(Debug, PartialEq, Clone)]
pub struct OrderByCondition {
    /// The variable to sort by
    pub variable: String,
    /// Sort direction (ascending or descending)
    pub direction: OrderDirection,
}

/// Sort direction for ORDER BY clauses.
#[derive(Debug, PartialEq, Clone)]
pub enum OrderDirection {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

// --- KML AST ---

/// Represents a KML (Knowledge Manipulation Language) statement.
/// KML is responsible for knowledge evolution and is the core tool for Agent learning.
#[derive(Debug, PartialEq, Clone)]
pub enum KmlStatement {
    /// UPSERT statement for atomic knowledge creation/updates
    Upsert(UpsertBlock),
    /// DELETE statement for knowledge removal
    Delete(DeleteStatement),
}

/// Represents an UPSERT block - the primary vehicle for "Knowledge Capsules".
/// Provides atomic creation or update of knowledge, ensuring idempotent operations.
/// Example: `UPSERT { CONCEPT @handle { ... } } WITH METADATA { ... }`
#[derive(Debug, PartialEq, Clone)]
pub struct UpsertBlock {
    /// List of concepts and propositions to upsert
    pub items: Vec<UpsertItem>,
    /// Global metadata for the entire upsert operation
    pub metadata: Option<Map<String, Json>>,
}

/// Represents an item within an UPSERT block.
/// Can be either a concept definition or a standalone proposition.
#[derive(Debug, PartialEq, Clone)]
pub enum UpsertItem {
    /// A concept block defining a concept node
    Concept(ConceptBlock),
    /// A proposition block defining a standalone proposition
    Proposition(PropositionBlock),
}

/// Represents a concept definition within an UPSERT block.
/// Defines a concept node with its attributes and outgoing propositions.
/// Example: `CONCEPT @handle { ON { ... } SET ATTRIBUTES { ... } SET PROPOSITIONS { ... } }`
#[derive(Debug, PartialEq, Clone)]
pub struct ConceptBlock {
    /// Local handle for referencing within the transaction (starts with @)
    pub handle: String,
    /// ON clause for matching existing concepts or creating new ones
    pub on: OnClause,
    /// Optional attributes to set on the concept
    pub set_attributes: Option<Map<String, Json>>,
    /// Optional propositions emanating from this concept
    pub set_propositions: Option<Vec<SetProposition>>,
    /// Optional metadata for this concept
    pub metadata: Option<Map<String, Json>>,
}

/// Represents a proposition to be set from a concept.
/// Used within the SET PROPOSITIONS block of a concept definition.
#[derive(Debug, PartialEq, Clone)]
pub struct SetProposition {
    /// The predicate (relationship type)
    pub predicate: String,
    /// The object of the proposition (node or local handle)
    pub object: PropObject,
    /// Optional metadata for this specific proposition
    pub metadata: Option<Map<String, Json>>,
}

/// Represents a standalone proposition definition within an UPSERT block.
/// Used for creating complex relationships that don't naturally belong to a single concept.
/// Example: `PROPOSITION @handle { (ON { ... }, "predicate", ON { ... }) }`
#[derive(Debug, PartialEq, Clone)]
pub struct PropositionBlock {
    /// Local handle for referencing within the transaction (starts with @)
    pub handle: String,
    /// Subject of the proposition (existing concept)
    pub subject: OnClause,
    /// Predicate (relationship type)
    pub predicate: String,
    /// Object of the proposition (node or local handle)
    pub object: PropObject,
    /// Optional metadata for this proposition
    pub metadata: Option<Map<String, Json>>,
}

/// Represents different types of DELETE statements in KML.
/// Provides targeted removal of knowledge components from the Cognitive Nexus.
#[derive(Debug, PartialEq, Clone)]
pub enum DeleteStatement {
    /// Delete specific attributes from a concept
    /// Example: `DELETE ATTRIBUTES { "risk_category" } FROM ON { type: "Drug", name: "Aspirin" }`
    DeleteAttributes {
        /// List of attribute names to delete
        attributes: Vec<String>,
        /// The concept to delete attributes from
        from: OnClause,
    },
    /// Delete an entire concept and all its relationships
    /// Example: `DELETE CONCEPT ON { type: "Drug", name: "OutdatedDrug" } DETACH`
    DeleteConcept {
        /// The concept to delete
        on: OnClause,
    },
    /// Delete a specific proposition
    /// Example: `DELETE PROPOSITION (ON { ... }, "treats", ON { ... })`
    DeleteProposition {
        /// Subject of the proposition to delete
        subject: OnClause,
        /// Predicate of the proposition to delete
        predicate: String,
        /// Object of the proposition to delete
        object: OnClause,
    },
    /// Delete propositions matching a pattern
    /// Example: `DELETE PROPOSITIONS WHERE { PROP(?s, ?p, ?o) { source: "untrusted" } }`
    DeletePropositionsWhere(Vec<WhereClause>),
}

// --- META AST ---

/// Represents META commands for knowledge exploration and grounding.
/// META is a lightweight subset focused on introspection and disambiguation.
/// These are fast, metadata-driven commands that don't involve complex graph traversal.
#[derive(Debug, PartialEq, Clone)]
pub enum MetaCommand {
    /// DESCRIBE commands for schema information and cognitive primers
    Describe(DescribeTarget),
    /// SEARCH commands for concept disambiguation
    Search(SearchCommand),
}

/// Represents different targets for DESCRIBE commands.
/// Used to query the "schema" information of the Cognitive Nexus.
#[derive(Debug, PartialEq, Clone)]
pub enum DescribeTarget {
    /// DESCRIBE PRIMER - gets the "Cognitive Primer" for LLM guidance
    Primer,
    /// DESCRIBE DOMAINS - lists all knowledge domains
    Domains,
    /// DESCRIBE CONCEPT_TYPES - lists all concept types
    ConceptTypes,
    /// DESCRIBE CONCEPT_TYPE "TypeName" - details about a specific concept type
    ConceptType(String),
    /// DESCRIBE PROPOSITION_TYPES - lists all proposition types
    PropositionTypes,
    /// DESCRIBE PROPOSITION_TYPE "TypeName" - details about a specific proposition type
    PropositionType(String),
}

/// Represents a SEARCH command for concept disambiguation.
/// Helps LLMs find and identify concepts when exact matches are unclear.
/// Example: `SEARCH "aspirin" IN_TYPE "Drug" LIMIT 10`
#[derive(Debug, PartialEq, Clone)]
pub struct SearchCommand {
    /// The search term
    pub term: String,
    /// Optional type constraint for the search
    pub in_type: Option<String>,
    /// Optional limit on the number of results
    pub limit: Option<u64>,
}
