//! Error types for schema module
use thiserror::Error;

/// Schema related errors
#[derive(Error, Debug)]
pub enum SchemaError {
    #[error("Invalid schema: {0}")]
    InvalidSchema(String),
    /// Invalid field type error
    #[error("Invalid field type: {0}")]
    InvalidFieldType(String),

    /// Invalid field value error
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),

    /// Invalid field name error
    #[error("Invalid field name: {0}")]
    InvalidFieldName(String),

    /// Field validation error
    #[error("Field validation failed: {0}")]
    ValidationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}
