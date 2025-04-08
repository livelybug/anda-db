use proc_macro::TokenStream;

mod field_typed;

/// A derive macro that generates a `field_type()` function for structs.
///
/// This macro analyzes the struct fields and their types, mapping them to the
/// appropriate `FieldType` enum variants. It supports common Rust types and
/// handles Option<T> wrappers.
///
/// # Attributes
///
/// - `field_type = "TypeName"`: Override the inferred type with a specific FieldType
///
/// # Example
///
/// ```rust
/// use anda_db_derive::FieldTyped;
/// use anda_db::schema::{FieldType, Xid};
///
/// #[derive(FieldTyped)]
/// struct User {
///     #[field_type = "Bytes"]
///     id: Xid,
///     name: String,
///     age: u32,
/// }
/// ```
///
/// This will generate a `field_type()` method that returns a `FieldType::Map`
/// containing the type information for each field.
#[proc_macro_derive(FieldTyped, attributes(field_type))]
pub fn field_typed_derive(input: TokenStream) -> TokenStream {
    field_typed::field_typed_derive(input)
}
