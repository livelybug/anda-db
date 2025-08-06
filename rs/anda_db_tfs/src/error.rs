use thiserror::Error;

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Errors that can occur when working with BM25 index.
#[derive(Error, Debug)]
pub enum BM25Error {
    /// Index-related errors.
    #[error("BM25 index {name:?}, error: {source:?}")]
    Generic { name: String, source: BoxError },

    /// CBOR serialization/deserialization errors
    #[error("BM25 index {name:?}, CBOR serialization error: {source:?}")]
    Serialization { name: String, source: BoxError },

    /// Error when a token is not found.
    #[error("BM25 index {name:?}, document {id} not found")]
    NotFound { name: String, id: u64 },

    /// Error when trying to add a document with an ID that already exists
    #[error("BM25 index {name:?}, document {id} already exists")]
    AlreadyExists { name: String, id: u64 },

    /// Error when tokenization produces no tokens for a document
    #[error("BM25 index {name:?}, document {id} tokenization failed: {text:?}")]
    TokenizeFailed { name: String, id: u64, text: String },
}
