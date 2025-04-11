use thiserror::Error;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Errors that can occur when working with B-tree index.
#[derive(Error, Debug)]
pub enum BtreeError {
    /// Index-related errors.
    #[error("Btree index {name:?}, error: {source:?}")]
    Generic { name: String, source: BoxError },

    /// CBOR serialization/deserialization errors
    #[error("Btree index {name:?}, CBOR serialization error: {source:?}")]
    Serialization { name: String, source: BoxError },

    /// Error when a token is not found.
    #[error("Btree index {name:?}, value not found: {value:?}")]
    NotFound { name: String, value: String },

    /// Error when trying to add a segment with an ID that already exists
    #[error("Btree index {name:?}, value {value} already exists in document {id}")]
    AlreadyExists {
        name: String,
        id: String,
        value: String,
    },
}
