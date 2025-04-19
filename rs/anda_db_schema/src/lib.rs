mod document;
mod error;
mod field;
mod resource;
mod schema;
mod segment;
mod value_serde;

pub use anda_db_derive::FieldTyped;
pub use ic_auth_types::{EMPTY_XID, Xid};

pub use document::*;
pub use error::*;
pub use field::*;
pub use resource::*;
pub use schema::*;
pub use segment::*;

/// Validate a field name
///
/// Field names must:
/// - Not be empty
/// - Not exceed 64 characters
/// - Contain only lowercase letters, numbers, and underscores
///
/// # Arguments
/// * `s` - The field name to validate
///
/// # Returns
/// * `Result<(), SchemaError>` - Ok if valid, or an error message if invalid
pub fn validate_field_name(s: &str) -> Result<(), SchemaError> {
    if s.is_empty() {
        return Err(SchemaError::FieldName("empty string".to_string()));
    }

    if s.len() > 64 {
        return Err(SchemaError::FieldName(format!(
            "string length {} exceeds the limit 64",
            s.len()
        )));
    }

    for c in s.chars() {
        if !matches!(c, 'a'..='z' | '0'..='9' | '_' ) {
            return Err(SchemaError::FieldName(format!(
                "Invalid character {:?} in {:?}",
                c, s
            )));
        }
    }
    Ok(())
}
