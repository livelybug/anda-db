//! Error types for schema module
use thiserror::Error;

/// Schema related errors
#[derive(Error, Debug)]
pub enum SchemaError {
    #[error("Invalid schema: {0}")]
    Schema(String),
    /// Invalid field type error
    #[error("Invalid field type: {0}")]
    FieldType(String),

    /// Invalid field value error
    #[error("Invalid field value: {0}")]
    FieldValue(String),

    /// Invalid field name error
    #[error("Invalid field name: {0}")]
    FieldName(String),

    /// Field validation error
    #[error("Field validation failed: {0}")]
    Validation(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
}
