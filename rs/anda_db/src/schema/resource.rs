use serde::{Deserialize, Serialize};

use super::FieldType;

/// Represents a resource that can be sent to agents or tools.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// A tag that identifies the type of this resource.
    #[serde(rename = "t")]
    pub tag: String,

    /// The URI of this resource.
    #[serde(rename = "u")]
    pub uri: Option<String>,

    /// A human-readable name for this resource.
    #[serde(rename = "n")]
    pub name: Option<String>,

    /// A description of what this resource represents.
    /// This can be used by clients to improve the LLM's understanding of available resources.
    #[serde(rename = "d")]
    pub description: Option<String>,

    /// MIME type, https://developer.mozilla.org/zh-CN/docs/Web/HTTP/MIME_types/Common_types
    #[serde(rename = "m")]
    pub mime_type: Option<String>,

    /// The binary data of this resource.
    #[serde(rename = "b")]
    pub blob: Option<Vec<u8>>,

    /// The size of the resource in bytes.
    #[serde(rename = "s")]
    pub size: Option<usize>,

    /// The SHA3-256 hash of the resource.
    #[serde(rename = "h")]
    pub hash: Option<[u8; 32]>,
}

impl Resource {
    pub fn field_type() -> FieldType {
        FieldType::Map(
            vec![
                ("t".to_string(), FieldType::Text),
                (
                    "u".to_string(),
                    FieldType::Option(Box::new(FieldType::Text)),
                ),
                (
                    "n".to_string(),
                    FieldType::Option(Box::new(FieldType::Text)),
                ),
                (
                    "d".to_string(),
                    FieldType::Option(Box::new(FieldType::Text)),
                ),
                (
                    "m".to_string(),
                    FieldType::Option(Box::new(FieldType::Text)),
                ),
                (
                    "b".to_string(),
                    FieldType::Option(Box::new(FieldType::Bytes)),
                ),
                ("s".to_string(), FieldType::Option(Box::new(FieldType::U64))),
                (
                    "h".to_string(),
                    FieldType::Option(Box::new(FieldType::Bytes)),
                ),
            ]
            .into_iter()
            .collect(),
        )
    }
}
