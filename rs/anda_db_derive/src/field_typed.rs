use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::{
    Attribute, Data, DeriveInput, Expr, Fields, GenericArgument, Lit, Meta, PathArguments, Type,
    parse_macro_input,
};

/// A derive macro that generates a `field_type()` function for structs.
/// FieldType represents field types in AndaDB schema.
///
/// This macro analyzes the struct fields and their types, mapping them to the
/// appropriate `FieldType` enum variants. It supports common Rust types and
/// handles Option<T> wrappers.
pub fn field_typed_derive(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

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
        let field_name = field.ident.as_ref().unwrap();
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

// 查找并解析 serde 的 rename 属性解析
fn find_rename_attr(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        let args = match attr
            .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)
        {
            Ok(args) => args,
            Err(_) => continue,
        };

        // 遍历所有参数，查找 rename 属性
        for meta in args {
            if let Meta::NameValue(name_value) = meta {
                if name_value.path.is_ident("rename") {
                    if let Expr::Lit(expr_lit) = &name_value.value {
                        if let Lit::Str(s) = &expr_lit.lit {
                            return Some(s.value());
                        }
                    }
                }
            }
        }
    }
    None
}

// 查找并解析 field_type 属性
fn find_field_type_attr(attrs: &[Attribute]) -> Option<proc_macro2::TokenStream> {
    for attr in attrs {
        if attr.path().is_ident("field_type") {
            // 尝试访问属性的参数
            if let Ok(meta_name_value) = attr.meta.require_name_value() {
                if let Expr::Lit(expr_lit) = &meta_name_value.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        return Some(parse_field_type_str(&lit_str.value()));
                    }
                }
            }
        }
    }
    None
}

// 解析类型字符串为 TokenStream
fn parse_field_type_str(type_str: &str) -> proc_macro2::TokenStream {
    match type_str {
        // 基本类型
        "Bytes" => quote! { FieldType::Bytes },
        "Text" => quote! { FieldType::Text },
        "U64" => quote! { FieldType::U64 },
        "I64" => quote! { FieldType::I64 },
        "F64" => quote! { FieldType::F64 },
        "F32" => quote! { FieldType::F32 },
        "Bool" => quote! { FieldType::Bool },
        "Json" => quote! { FieldType::Json },
        "Vector" => quote! { FieldType::Vector },

        // 复合类型 - 支持 Array 和 Option
        s if s.starts_with("Array<") && s.ends_with(">") => {
            let inner = &s[6..s.len() - 1];
            let inner_type = parse_field_type_str(inner);
            quote! { FieldType::Array(vec![#inner_type]) }
        }
        s if s.starts_with("Option<") && s.ends_with(">") => {
            let inner = &s[7..s.len() - 1];
            let inner_type = parse_field_type_str(inner);
            quote! { FieldType::Option(Box::new(#inner_type)) }
        }

        // 默认或不支持的类型
        _ => quote! { FieldType::Any },
    }
}

// 根据 Rust 类型确定对应的 FieldType
fn determine_field_type(ty: &Type) -> Result<proc_macro2::TokenStream, String> {
    match ty {
        Type::Path(type_path) if !type_path.path.segments.is_empty() => {
            let path = &type_path.path;
            let segment = &path.segments[0];
            let type_name = segment.ident.to_string();

            match type_name.as_str() {
                "Option" => {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(GenericArgument::Type(inner_type)) = args.args.first() {
                            let inner_field_type = determine_field_type(inner_type)?;
                            return Ok(quote! { FieldType::Option(Box::new(#inner_field_type)) });
                        }
                    }
                    Ok(quote! { FieldType::Option(Box::new(FieldType::U64)) })
                }
                "String" | "str" => Ok(quote! { FieldType::Text }),
                "Vec" => {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(GenericArgument::Type(inner_type)) = args.args.first() {
                            if is_u8_type(inner_type) {
                                return Ok(quote! { FieldType::Bytes });
                            } else if is_bf16_type(inner_type) {
                                return Ok(quote! { FieldType::Vector });
                            } else {
                                let inner_field_type = determine_field_type(inner_type)?;
                                return Ok(quote! { FieldType::Array(vec![#inner_field_type]) });
                            }
                        }
                    }
                    Err(format!(
                        "Unable to determine Vec element type for: {}",
                        type_name
                    ))
                }
                "bool" => Ok(quote! { FieldType::Bool }),
                "i8" | "i16" | "i32" | "i64" | "isize" => Ok(quote! { FieldType::I64 }),
                "u8" | "u16" | "u32" | "u64" | "usize" => Ok(quote! { FieldType::U64 }),
                "f32" => Ok(quote! { FieldType::F32 }),
                "f64" => Ok(quote! { FieldType::F64 }),
                "Bytes" | "ByteArray" | "ByteBuf" | "BytesB64" | "ByteArrayB64" | "ByteBufB64" => {
                    Ok(quote! { FieldType::Bytes })
                }
                "HashMap" | "BTreeMap" | "Map" => {
                    // 处理 HashMap 和 BTreeMap 类型
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if args.args.len() >= 2 {
                            // 检查第一个泛型参数是否为 String 类型
                            let key_type = &args.args[0];
                            if let GenericArgument::Type(key_type) = key_type {
                                if is_string_type(key_type) {
                                    if let GenericArgument::Type(value_type) = &args.args[1] {
                                        let value_field_type = determine_field_type(value_type)?;
                                        return Ok(quote! {
                                            FieldType::Map(std::collections::BTreeMap::from([(
                                                "*".to_string(),
                                                #value_field_type
                                            )]))
                                        });
                                    }
                                } else {
                                    return Err(format!(
                                        "Map key type must be String, found: {:?}",
                                        key_type
                                    ));
                                }
                            }
                        }
                    }
                    Err(format!("Invalid map type: {}", type_name))
                }
                _ => {
                    // 处理嵌套路径
                    // if path.segments.len() > 1 {
                    //     // 尝试检查完整路径
                    //     let full_path = path
                    //         .segments
                    //         .iter()
                    //         .map(|seg| seg.ident.to_string())
                    //         .collect::<Vec<_>>()
                    //         .join("::");

                    //     match full_path.as_str() {
                    //         "half::bf16" => return Ok(quote! { FieldType::Bf16 }),
                    //         "serde_bytes::ByteArray" | "serde_bytes::ByteBuf" => {
                    //             return Ok(quote! { FieldType::Bytes });
                    //         }
                    //         _ => {}
                    //     }
                    // }

                    // 处理自定义结构体类型 - 尝试使用该结构体的 field_type 方法
                    // 将其他结构体类型也映射为 FieldType::Map
                    let type_ident = Ident::new(&type_name, Span::call_site());
                    Ok(quote! {
                        #type_ident::field_type()
                    })
                }
            }
        }
        Type::Array(array) if is_u8_type(&array.elem) => Ok(quote! { FieldType::Bytes }),
        Type::Array(array) if is_bf16_type(&array.elem) => Ok(quote! { FieldType::Vector }),
        Type::Array(array) => {
            let inner_type = determine_field_type(&array.elem)?;
            Ok(quote! { FieldType::Array(vec![#inner_type]) })
        }
        _ => Err(format!("Unsupported type: {:?}", ty)),
    }
}

// 检查类型是否为 u8
fn is_u8_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "u8";
        }
    }
    false
}

// 检查类型是否为 String 或 &str
fn is_string_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "String" || segment.ident == "str";
        }
    }
    false
}

fn is_bf16_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "bf16";
        }
    }
    false
}
