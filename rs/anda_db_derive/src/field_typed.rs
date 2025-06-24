use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, ext::IdentExt, parse_macro_input};

use crate::common::{determine_field_type, find_field_type_attr, find_rename_attr};

/// A derive macro that generates a `field_type()` function for structs.
/// FieldType represents field types in AndaDB schema.
///
/// This macro analyzes the struct fields and their types, mapping them to the
/// appropriate `FieldType` enum variants. It supports common Rust types and
/// handles Option<T> wrappers.
pub fn field_typed_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident.unraw();

    // Process struct fields
    let fields = if let Data::Struct(data_struct) = &input.data {
        match &data_struct.fields {
            Fields::Named(fields_named) => &fields_named.named,
            _ => {
                return TokenStream::from(quote! {
                    compile_error!("FieldTyped only supports structs with named fields");
                });
            }
        }
    } else {
        return TokenStream::from(quote! {
            compile_error!("FieldTyped only supports structs");
        });
    };

    // Generate field type mappings
    let field_type_mappings = fields.iter().map(|field| {
        let field_name = field.ident.as_ref().unwrap().unraw();
        let field_name_str = field_name.to_string();

        // Get renamed field from serde attribute if present
        let rename_attr = find_rename_attr(&field.attrs).unwrap_or_else(|| field_name_str.clone());

        // Check if there's a custom field_type attribute
        let custom_field_type = find_field_type_attr(&field.attrs);

        // Determine the field type
        let field_type = if let Some(field_type) = custom_field_type {
            quote! { #field_type }
        } else {
            match determine_field_type(&field.ty) {
                Ok(field_type) => field_type,
                Err(err_msg) => {
                    // Generate a compile error for unsupported types
                    return quote! {
                        compile_error!(#err_msg)
                    };
                }
            }
        };

        quote! {
            (#rename_attr.to_string(), #field_type)
        }
    });

    // Generate the field_type function implementation
    let expanded = quote! {
        impl #name {
            pub fn field_type() -> FieldType {
                FieldType::Map(
                    vec![
                        #(#field_type_mappings),*
                    ]
                    .into_iter()
                    .collect(),
                )
            }
        }
    };

    TokenStream::from(expanded)
}
