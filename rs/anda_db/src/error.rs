//! Error types for schema module
use anda_db_btree::BTreeError;
use thiserror::Error;

use crate::schema::{BoxError, SchemaError};

/// Anda DB related errors
#[derive(Error, Debug)]
pub enum DBError {
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

impl From<object_store::Error> for DBError {
    fn from(err: object_store::Error) -> Self {
        match err {
            object_store::Error::NotFound { path, source } => DBError::NotFound {
                name: "unknown".to_string(),
                path,
                source,
            },
            object_store::Error::AlreadyExists { path, source } => DBError::AlreadyExists {
                name: "unknown".to_string(),
                path,
                source,
            },
            err => DBError::Storage {
                name: "unknown".to_string(),
                source: err.into(),
            },
        }
    }
}

impl From<SchemaError> for DBError {
    fn from(err: SchemaError) -> Self {
        DBError::Schema {
            name: "unknown".to_string(),
            source: err.into(),
        }
    }
}

impl From<BTreeError> for DBError {
    fn from(err: BTreeError) -> Self {
        match &err {
            BTreeError::Generic { name, .. } => DBError::Index {
                name: name.clone(),
                source: err.into(),
            },
            BTreeError::Serialization { name, .. } => DBError::Index {
                name: name.clone(),
                source: err.into(),
            },
            BTreeError::NotFound { name, .. } => DBError::Index {
                name: name.clone(),
                source: err.into(),
            },
            BTreeError::AlreadyExists { name, .. } => DBError::Index {
                name: name.clone(),
                source: err.into(),
            },
        }
    }
}
