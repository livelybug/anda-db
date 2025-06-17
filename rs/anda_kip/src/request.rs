/// KIP (Knowledge Interaction Protocol) Request and Response structures
///
/// This module implements the standardized request-response model for all interactions
/// with the Cognitive Nexus as defined in KIP specification section 6.
///
/// LLM Agents send structured requests (typically encapsulated in Function Calling)
/// containing KIP commands to the Cognitive Nexus, which returns structured JSON responses.
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::{
    Json, Map,
    error::KipError,
    executor::{Executor, execute_kip},
};

/// Request structure for KIP command execution
///
/// Represents the arguments for the `execute_kip` function as defined in the KIP specification.
/// LLM-generated KIP commands should be sent to the Cognitive Nexus through this structured request.
///
/// # Example
/// ```json
/// {
///   "command": "FIND(?drug_name) WHERE { ?symptom(name: $symptom_name) PROP(?drug, \"treats\", ?symptom) ATTR(?drug, \"name\", ?drug_name) } LIMIT $limit",
///   "parameters": {
///     "symptom_name": "Headache",
///     "limit": 10
///   },
///   "dry_run": false
/// }
/// ```
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Request {
    /// Complete, unmodified KIP command text
    ///
    /// Contains the full KIP command (KQL, KML, or META) to be executed.
    /// Placeholders in the format `$key_name` can be used for parameter substitution.
    pub command: String,

    /// Optional key-value pairs for execution context parameters
    ///
    /// Used to pass parameters outside of the command text itself.
    /// Command text can reference these parameters using `$key_name` placeholders.
    ///
    /// Optional keys include:
    /// - `domain` (String): Specifies the target domain if the Cognitive Nexus contains multi-domain knowledge
    pub parameters: Map<String, Json>,

    /// Dry run flag for command validation
    ///
    /// If `true`, only validates the command's syntax and logic without executing
    /// or persisting any changes. Useful for testing and validation purposes.
    pub dry_run: bool,
}

impl Request {
    /// Converts the request to a complete command string with parameter substitution
    ///
    /// Replaces all `$key_name` placeholders in the command with corresponding values
    /// from the parameters map. If no parameters are provided, returns the original command.
    ///
    /// # Returns
    /// - `Cow::Borrowed` if no parameters need substitution
    /// - `Cow::Owned` if parameter substitution was performed
    pub fn to_command(&self) -> Cow<str> {
        if self.parameters.is_empty() {
            Cow::Borrowed(&self.command)
        } else {
            let mut result = self.command.clone();

            // replace all occurrences of $key_name with the corresponding value
            for (key, value) in &self.parameters {
                let placeholder = format!("${}", key);
                let replacement = match value {
                    Json::Number(n) => n.to_string(),
                    Json::Bool(b) => b.to_string(),
                    Json::Null => "null".to_string(),
                    _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
                };
                result = result.replace(&placeholder, &replacement);
            }

            Cow::Owned(result)
        }
    }

    /// Executes the KIP command using the provided executor
    pub async fn execute(&self, nexus: &impl Executor) -> Response {
        let command = self.to_command();
        execute_kip(nexus, &command, self.dry_run).await
    }
}

/// Response structure from the Cognitive Nexus
///
/// All responses from the Cognitive Nexus are JSON objects with this structure.
/// Either `result` or `error` must be present, but never both.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Response {
    /// Successful response containing the request results
    ///
    /// Must be present when the request succeeds.
    /// The internal structure is defined by the specific KIP request command.
    Result(Json),

    /// Error response containing structured error details
    ///
    /// Must be present when the request fails.
    /// Contains detailed information about what went wrong.
    Error(ErrorObject),
}

/// Structured error details for failed requests
///
/// Provides comprehensive error information including error type,
/// human-readable message, and optional additional data.
#[derive(Debug, Deserialize, Serialize)]
pub struct ErrorObject {
    /// Error type/category name
    ///
    /// Identifies the specific type of error that occurred.
    /// Examples: "ParseError", "ExecutionError", "NotImplemented", "InvalidCommand"
    pub name: String,

    /// Human-readable error message
    ///
    /// Provides detailed information about the error for debugging and user feedback.
    pub message: String,

    /// Optional additional error data
    ///
    /// May contain structured data relevant to the specific error,
    /// such as validation details or context information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Json>,
}

// #[cfg(feature = "nightly")]
// impl<E> From<E> for ErrorObject
// where
//     E: std::error::Error,
// {
//     default fn from(error: E) -> Self {
//         ErrorObject {
//             name: std::any::type_name::<E>()
//                 .split("::")
//                 .last()
//                 .unwrap_or("Error")
//                 .to_string(),
//             message: error.to_string(),
//             data: None,
//         }
//     }
// }

/// Conversion from serde_json::Error to ErrorObject
///
/// Handles JSON serialization/deserialization errors
impl From<serde_json::Error> for ErrorObject {
    fn from(error: serde_json::Error) -> Self {
        ErrorObject {
            name: "SerializationError".to_string(),
            message: error.to_string(),
            data: None,
        }
    }
}

/// Conversion from KipError to ErrorObject
///
/// Maps internal KIP errors to the standardized error response format
impl From<KipError> for ErrorObject {
    fn from(error: KipError) -> Self {
        let (name, message) = match &error {
            KipError::Parse(msg) => ("ParseError".to_string(), msg.clone()),
            KipError::Execution(msg) => ("ExecutionError".to_string(), msg.clone()),
            KipError::NotImplemented(msg) => ("NotImplemented".to_string(), msg.clone()),
            KipError::InvalidCommand(msg) => ("InvalidCommand".to_string(), msg.clone()),
        };

        ErrorObject {
            name,
            message,
            data: None,
        }
    }
}

/// Conversion from KipError to Response
///
/// Automatically wraps KIP errors in the appropriate response format
impl From<KipError> for Response {
    fn from(error: KipError) -> Self {
        Response::Error(error.into())
    }
}

/// Generic conversion from Result to Response
///
/// Automatically converts successful results to Response::Result
/// and errors to Response::Error with proper serialization handling
impl<T, E> From<Result<T, E>> for Response
where
    T: Serialize,
    E: Into<ErrorObject>,
{
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(data) => match serde_json::to_value(data) {
                Ok(value) => Response::Result(value),
                Err(err) => Response::Error(err.into()),
            },
            Err(err) => Response::Error(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_to_command_empty_parameters() {
        let request = Request {
            command: "FIND(?drug_name) WHERE { ?drug(type: \"Drug\") }".to_string(),
            parameters: Map::new(),
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(result, "FIND(?drug_name) WHERE { ?drug(type: \"Drug\") }");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_to_command_string_parameter() {
        let mut parameters = Map::new();
        parameters.insert(
            "symptom_name".to_string(),
            Json::String("Headache".to_string()),
        );

        let request = Request {
            command: "FIND(?drug_name) WHERE { ?symptom(name: $symptom_name) }".to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(
            result,
            "FIND(?drug_name) WHERE { ?symptom(name: \"Headache\") }"
        );
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn test_to_command_number_parameter() {
        let mut parameters = Map::new();
        parameters.insert(
            "limit".to_string(),
            Json::Number(serde_json::Number::from(10)),
        );
        parameters.insert(
            "risk_level".to_string(),
            Json::Number(serde_json::Number::from_f64(3.5).unwrap()),
        );

        let request = Request {
            command: "FIND(?drug) WHERE { ATTR(?drug, \"risk_level\", ?risk) FILTER(?risk < $risk_level) } LIMIT $limit".to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(
            result,
            "FIND(?drug) WHERE { ATTR(?drug, \"risk_level\", ?risk) FILTER(?risk < 3.5) } LIMIT 10"
        );
    }

    #[test]
    fn test_to_command_object_parameter() {
        let mut parameters = Map::new();
        parameters.insert(
            "metadata".to_string(),
            json!({"confidence": 0.95, "source": "clinical_trial"}),
        );

        let request = Request {
            command:
                "UPSERT { CONCEPT @drug { ON { name: \"TestDrug\" } } } WITH METADATA $metadata"
                    .to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(
            result,
            "UPSERT { CONCEPT @drug { ON { name: \"TestDrug\" } } } WITH METADATA {\"confidence\":0.95,\"source\":\"clinical_trial\"}"
        );
    }

    #[test]
    fn test_to_command_multiple_parameters() {
        let mut parameters = Map::new();
        parameters.insert(
            "symptom_name".to_string(),
            Json::String("Headache".to_string()),
        );
        parameters.insert(
            "limit".to_string(),
            Json::Number(serde_json::Number::from(5)),
        );
        parameters.insert("include_experimental".to_string(), Json::Bool(false));

        let request = Request {
            command: r#"
                FIND(?drug_name)
                WHERE {
                    ?symptom(name: $symptom_name)
                    PROP(?drug, "treats", ?symptom)
                    ATTR(?drug, "name", ?drug_name)
                    ATTR(?drug, "experimental", $include_experimental)
                }
                LIMIT $limit
            "#
            .to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        let expected = r#"
                FIND(?drug_name)
                WHERE {
                    ?symptom(name: "Headache")
                    PROP(?drug, "treats", ?symptom)
                    ATTR(?drug, "name", ?drug_name)
                    ATTR(?drug, "experimental", false)
                }
                LIMIT 5
            "#;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_command_same_parameter_multiple_times() {
        let mut parameters = Map::new();
        parameters.insert(
            "drug_type".to_string(),
            Json::String("Analgesic".to_string()),
        );

        let request = Request {
            command:
                "FIND(?drug1, ?drug2) WHERE { ?drug1(type: $drug_type) ?drug2(type: $drug_type) }"
                    .to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(
            result,
            "FIND(?drug1, ?drug2) WHERE { ?drug1(type: \"Analgesic\") ?drug2(type: \"Analgesic\") }"
        );
    }

    #[test]
    fn test_to_command_parameter_not_found() {
        let mut parameters = Map::new();
        parameters.insert(
            "existing_param".to_string(),
            Json::String("value".to_string()),
        );

        let request = Request {
            command: "FIND(?item) WHERE { ?item(name: $missing_param) }".to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        // 不存在的参数应该保持原样
        assert_eq!(result, "FIND(?item) WHERE { ?item(name: $missing_param) }");
    }

    #[test]
    fn test_to_command_special_characters_in_string() {
        let mut parameters = Map::new();
        parameters.insert(
            "special_name".to_string(),
            Json::String("Drug with \"quotes\" and $symbols".to_string()),
        );

        let request = Request {
            command: "FIND(?drug) WHERE { ?drug(name: $special_name) }".to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        assert_eq!(
            result,
            "FIND(?drug) WHERE { ?drug(name: \"Drug with \\\"quotes\\\" and $symbols\") }"
        );
    }

    #[test]
    fn test_to_command_complex_kip_example() {
        // 测试一个符合 KIP 规范的完整示例
        let mut parameters = Map::new();
        parameters.insert(
            "symptom_name".to_string(),
            Json::String("Brain Fog".to_string()),
        );
        parameters.insert(
            "confidence_threshold".to_string(),
            Json::Number(serde_json::Number::from_f64(0.8).unwrap()),
        );
        parameters.insert(
            "max_results".to_string(),
            Json::Number(serde_json::Number::from(20)),
        );

        let request = Request {
            command: r#"
                FIND(?drug_name, ?confidence)
                WHERE {
                    ?symptom(name: $symptom_name)
                    PROP(?drug, "treats", ?symptom) { confidence: ?confidence }
                    ATTR(?drug, "name", ?drug_name)
                    FILTER(?confidence > $confidence_threshold)
                }
                ORDER BY ?confidence DESC
                LIMIT $max_results
            "#
            .to_string(),
            parameters,
            dry_run: false,
        };

        let result = request.to_command();
        let expected = r#"
                FIND(?drug_name, ?confidence)
                WHERE {
                    ?symptom(name: "Brain Fog")
                    PROP(?drug, "treats", ?symptom) { confidence: ?confidence }
                    ATTR(?drug, "name", ?drug_name)
                    FILTER(?confidence > 0.8)
                }
                ORDER BY ?confidence DESC
                LIMIT 20
            "#;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_response() {
        let res = Response::Result(json!("Success"));
        assert_eq!(
            serde_json::to_string(&res).unwrap(),
            r#"{"result":"Success"}"#
        );

        let res = Response::Error(ErrorObject {
            name: "TestError".to_string(),
            message: "An error occurred".to_string(),
            data: Some(json!("Additional info")),
        });
        assert_eq!(
            serde_json::to_string(&res).unwrap(),
            r#"{"error":{"name":"TestError","message":"An error occurred","data":"Additional info"}}"#
        );
    }
}
