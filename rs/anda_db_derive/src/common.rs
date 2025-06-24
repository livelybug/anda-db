use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Expr, GenericArgument, Lit, Meta, PathArguments, Type, ext::IdentExt};

/// 查找并解析 serde 的 rename 属性
pub fn find_rename_attr(attrs: &[Attribute]) -> Option<String> {
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

/// 查找并解析 field_type 属性
pub fn find_field_type_attr(attrs: &[Attribute]) -> Option<TokenStream> {
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

/// 解析类型字符串为 TokenStream
pub fn parse_field_type_str(type_str: &str) -> TokenStream {
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

        // 添加对 Map<String, Type> 的支持
        s if s.starts_with("Map<String, ") && s.ends_with(">") => {
            let inner = &s[12..s.len() - 1];
            let inner_type = parse_field_type_str(inner);
            quote! {
                FieldType::Map(std::collections::BTreeMap::from([(
                    "*".to_string(),
                    #inner_type
                )]))
            }
        }

        // 默认或不支持的类型
        _ => {
            let error_msg = format!(
                "Unsupported field type: '{}'. Supported types: Bytes, Text, U64, I64, F64, F32, Bool, Json, Vector, Array<T>, Option<T>, Map<String, T>",
                type_str
            );
            quote! { compile_error!(#error_msg) }
        }
    }
}

/// 根据 Rust 类型确定对应的 FieldType
pub fn determine_field_type(ty: &Type) -> Result<TokenStream, String> {
    match ty {
        Type::Path(type_path) if !type_path.path.segments.is_empty() => {
            let path = &type_path.path;
            let segment = &path.segments[0];
            let type_name = segment.ident.unraw().to_string();

            match type_name.as_str() {
                "Option" => {
                    if let PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let Some(GenericArgument::Type(inner_type)) = args.args.first() {
                            let inner_field_type = determine_field_type(inner_type)?;
                            return Ok(quote! { FieldType::Option(Box::new(#inner_field_type)) });
                        }
                    }
                    Ok(quote! { FieldType::Option(Box::new(FieldType::Json)) })
                }
                "String" | "str" => Ok(quote! { FieldType::Text }),
                "Vec" | "HashSet" | "BTreeSet" => {
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
                "Json" => Ok(quote! { FieldType::Json }),
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
                    if path.segments.len() > 1 {
                        // 尝试检查完整路径
                        let full_path = path
                            .segments
                            .iter()
                            .map(|seg| seg.ident.unraw().to_string())
                            .collect::<Vec<_>>()
                            .join("::");

                        match full_path.as_str() {
                            "half::bf16" => return Ok(quote! { FieldType::Bf16 }),
                            "serde_bytes::ByteArray"
                            | "serde_bytes::ByteBuf"
                            | "serde_bytes::Bytes" => {
                                return Ok(quote! { FieldType::Bytes });
                            }
                            "serde_json::Value" => return Ok(quote! { FieldType::Json }),
                            _ => {}
                        }
                    }

                    // 处理自定义结构体类型 - 尝试使用该结构体的 field_type 方法
                    let type_ident =
                        proc_macro2::Ident::new(&type_name, proc_macro2::Span::call_site());
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
        _ => {
            // 对于未知类型，提供更详细的错误信息
            let error_msg = format!(
                "Unsupported type: '{:?}'. Consider:\n1. Using a supported primitive type\n2. Adding #[field_type = \"SupportedType\"] attribute\n3. Implementing FieldTyped for this type",
                ty
            );
            Err(error_msg)
        }
    }
}

/// 检查类型是否为 u8
pub fn is_u8_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "u8";
        }
    }
    false
}

/// 检查类型是否为 String 或 &str
pub fn is_string_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "String" || segment.ident == "str";
        }
    }
    false
}

/// 检查类型是否为 bf16
pub fn is_bf16_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "bf16";
        }
    }
    false
}

/// 检查类型是否为 u64
pub fn is_u64_type(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "u64";
        }
    }
    false
}
