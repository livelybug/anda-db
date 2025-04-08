use anda_db_derive::FieldTyped;
use serde::{Deserialize, Serialize};

use super::FieldType;

/// Represents a resource for AI Agents.
/// It can be a file, a URL, or any other type of resource.
#[derive(Debug, Default, Clone, Serialize, Deserialize, FieldTyped)]
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
    #[serde(with = "serde_bytes")]
    pub blob: Option<Vec<u8>>,

    /// The size of the resource in bytes.
    #[serde(rename = "s")]
    pub size: Option<usize>,

    /// The SHA3-256 hash of the resource.
    #[serde(rename = "h")]
    #[serde(with = "serde_bytes")]
    pub hash: Option<[u8; 32]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Ft;

    #[test]
    fn test_field_type() {
        let tf = Resource::field_type();
        assert_eq!(
            tf,
            Ft::Map(
                vec![
                    ("t".to_string(), Ft::Text),
                    ("u".to_string(), Ft::Option(Box::new(Ft::Text)),),
                    ("n".to_string(), Ft::Option(Box::new(Ft::Text)),),
                    ("d".to_string(), Ft::Option(Box::new(Ft::Text)),),
                    ("m".to_string(), Ft::Option(Box::new(Ft::Text)),),
                    ("b".to_string(), Ft::Option(Box::new(Ft::Bytes)),),
                    ("s".to_string(), FieldType::Option(Box::new(Ft::U64))),
                    ("h".to_string(), Ft::Option(Box::new(Ft::Bytes)),),
                ]
                .into_iter()
                .collect()
            )
        );
    }
}
