use serde_json::Value;
use thiserror::Error;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Errors that can occur when working with B-tree index.
#[derive(Error, Debug)]
pub enum BTreeError {
    /// Index-related errors.
    #[error("BTree index {name:?}, error: {source:?}")]
    Generic { name: String, source: BoxError },

    /// CBOR serialization/deserialization errors
    #[error("BTree index {name:?}, CBOR serialization error: {source:?}")]
    Serialization { name: String, source: BoxError },

    /// Error when a token is not found.
    #[error("BTree index {name:?}, value {value:?} not found in document {id}")]
    NotFound {
        name: String,
        id: Value,
        value: Value,
    },

    /// Error when trying to add a document with an ID that already exists
    #[error("BTree index {name:?}, value {value} already exists in document {id}")]
    AlreadyExists {
        name: String,
        id: Value,
        value: Value,
    },
}
