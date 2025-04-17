use thiserror::Error;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Errors that can occur when working with HNSW index.
#[derive(Error, Debug)]
pub enum HnswError {
    /// Index-related errors.
    #[error("HNSW index {name:?}, error: {source:?}")]
    Generic { name: String, source: BoxError },

    /// CBOR serialization/deserialization errors.
    #[error("HNSW index {name:?}, CBOR serialization error: {source:?}")]
    Serialization { name: String, source: BoxError },

    /// Error when vector dimensions don't match the index dimension.
    #[error("HNSW index {name:?}, vector dimension mismatch, expected {expected}, got {got}")]
    DimensionMismatch {
        name: String,
        expected: usize,
        got: usize,
    },

    /// Error when a token is not found.
    #[error("HNSW index {name:?}, node not found: {id:?}")]
    NotFound { name: String, id: u64 },

    /// Error when trying to add a document with an ID that already exists
    #[error("HNSW index {name:?}, node {id} already exists")]
    AlreadyExists { name: String, id: u64 },
}
