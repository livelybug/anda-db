//! Error types for schema module
use thiserror::Error;

use crate::schema::SchemaError;

/// A type alias for a boxed error that is thread-safe and sendable across threads.
/// This is commonly used as a return type for functions that can return various error types.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Anda DB related errors
#[derive(Error, Debug)]
pub enum DbError {
    #[error("Anda DB {name:?} error: {source:?}")]
    Generic { name: String, source: BoxError },

    #[error("Collection {name:?} error: {source:?}")]
    Collection { name: String, source: BoxError },

    #[error("Schema error: {source:?}")]
    Schema { name: String, source: BoxError },

    #[error("Storage error: {source:?}")]
    Storage { name: String, source: BoxError },

    #[error("Index error: {source:?}")]
    Index { name: String, source: BoxError },

    #[error("Object {name} at location {path} not found: {source:?}")]
    NotFound {
        name: String,
        path: String,
        source: BoxError,
    },

    #[error("Object {name} at location {path} already exists: {source:?}")]
    AlreadyExists {
        name: String,
        path: String,
        source: BoxError,
    },

    #[error("Serialization error: {source:?}")]
    Serialization { name: String, source: BoxError },
}

impl From<object_store::Error> for DbError {
    fn from(err: object_store::Error) -> Self {
        match err {
            object_store::Error::NotFound { path, source } => DbError::NotFound {
                name: "unknown".to_string(),
                path,
                source,
            },
            object_store::Error::AlreadyExists { path, source } => DbError::AlreadyExists {
                name: "unknown".to_string(),
                path,
                source,
            },
            err => DbError::Storage {
                name: "unknown".to_string(),
                source: err.into(),
            },
        }
    }
}

impl From<SchemaError> for DbError {
    fn from(err: SchemaError) -> Self {
        DbError::Schema {
            name: "unknown".to_string(),
            source: err.into(),
        }
    }
}
