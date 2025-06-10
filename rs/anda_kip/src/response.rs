use crate::error::KipError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
pub struct KipResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorDetails>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<crate::nexus::KipValue>>,
}

pub type KipResult = Result<KipResponse, KipError>;

impl KipResponse {
    pub fn success(data: impl Serialize) -> Self {
        KipResponse {
            status: "ok".to_string(),
            data: Some(serde_json::to_value(data).unwrap()),
            error: None,
        }
    }

    pub fn success_message(message: &str) -> Self {
        KipResponse {
            status: "ok".to_string(),
            data: Some(serde_json::json!({ "message": message })),
            error: None,
        }
    }

    pub fn error(err: KipError) -> Self {
        let (code, message) = match &err {
            KipError::Parse(msg) => ("ParseError", msg.clone()),
            KipError::Execution(msg) => ("ExecutionError", msg.clone()),
            KipError::NotImplemented(msg) => ("NotImplemented", msg.clone()),
            KipError::InvalidCommand(msg) => ("InvalidCommand", msg.clone()),
        };
        KipResponse {
            status: "error".to_string(),
            data: None,
            error: Some(ErrorDetails {
                code: code.to_string(),
                message,
            }),
        }
    }
}
