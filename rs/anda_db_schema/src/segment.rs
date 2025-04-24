use half::bf16;
use serde::{Deserialize, Serialize};

use crate::{FieldType, FieldTyped, Fv, Vector};

/// Type alias for a segment identifier.
pub type SegmentId = u64;

/// Segment represents a unit consisting of a piece of text and an optional embedding vector.
/// It is the basic unit for both Full-Text Search and Vector Search.
///
/// According to mainstream embedding model parameters, the text length is recommended not to exceed 512 tokens.
/// Typically, the conversion ratio between tokens and characters is approximately:
///   - 1 English character ≈ 0.3 tokens
///   - 1 Chinese character ≈ 0.6 tokens
///
/// The vector dimension is generally above 512. Higher dimensions yield better retrieval accuracy,
/// but consume more memory and computational resources.
///
/// A Document should be split into a group of Segments for storage and retrieval.
#[derive(Debug, Default, Clone, Serialize, Deserialize, FieldTyped)]
pub struct Segment {
    /// Unique identifier for the segment.
    #[serde(rename = "i")]
    pub id: u64,

    /// The text content of the segment.
    #[serde(rename = "t")]
    pub text: String,

    /// Optional embedding vector representation of the segment.
    #[serde(rename = "v")]
    pub vec: Option<Vec<bf16>>,
}

impl Segment {
    /// Creates a new Segment with the given text.
    /// The id is initialized to 0 and the vector is set to None.
    pub fn new(text: String, vec: Option<Vector>) -> Self {
        Self { id: 0, text, vec }
    }

    /// Returns the id of the segment.
    pub fn id(&self) -> SegmentId {
        self.id
    }

    /// Returns a reference to the text content of the segment.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns an optional reference to the embedding vector of the segment.
    pub fn vec(&self) -> Option<&Vector> {
        self.vec.as_ref()
    }

    pub fn with_id(self, id: SegmentId) -> Self {
        Self { id, ..self }
    }

    pub fn with_vec(self, vec: Vector) -> Self {
        Self {
            vec: Some(vec),
            ..self
        }
    }

    pub fn with_vec_f32(self, vec: Vec<f32>) -> Self {
        Self {
            vec: Some(
                vec.into_iter()
                    .map(bf16::from_f32)
                    .collect::<Vec<bf16>>(),
            ),
            ..self
        }
    }

    pub fn id_from(fv: &Fv) -> Option<&SegmentId> {
        fv.get_field_as("i")
    }

    pub fn text_from(fv: &Fv) -> Option<&str> {
        fv.get_field_as("t")
    }

    pub fn vec_from(fv: &Fv) -> Option<&Vector> {
        fv.get_field_as("v")
    }
}
