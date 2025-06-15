use anda_db_derive::AndaDBSchema;
use anda_db_schema::{FieldEntry, FieldType, Schema, SchemaError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
struct TestUser {
    /// User's unique username handle
    #[unique]
    handle: String,
    /// User's display name
    name: String,
    /// User's age in years
    age: Option<u64>,
    /// Whether the user account is active
    active: bool,
    /// User tags for categorization
    tags: Vec<String>,
    /// User metadata with creation and update timestamps
    #[serde(rename = "metadata")]
    meta: Option<BTreeMap<String, u64>>,
}

// 测试包含 _id 字段的结构体
#[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
struct TestUserWithId {
    _id: u64,
    username: String,
    email: String,
}

// // 测试包含错误类型 _id 字段的结构体
// #[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
// struct TestUserWithStringId {
//     _id: String,
//     username: String,
//     email: String,
// }

// 测试各种数据类型
#[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
struct TestAllTypes {
    // 数字类型
    byte_val: u8,
    short_val: u16,
    int_val: u32,
    long_val: u64,
    signed_byte: i8,
    signed_short: i16,
    signed_int: i32,
    signed_long: i64,
    float_val: f32,
    double_val: f64,

    // 文本类型
    text: String,

    // 布尔类型
    flag: bool,

    // 字节数组
    data: Vec<u8>,

    // 数组类型
    numbers: Vec<i32>,
    strings: Vec<String>,

    // 可选类型
    optional_text: Option<String>,
    optional_number: Option<i64>,

    // Map 类型
    string_map: BTreeMap<String, String>,
    number_map: BTreeMap<String, i64>,
}

// 测试自定义字段类型属性
#[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
struct TestCustomFieldType {
    #[field_type = "Json"]
    custom_field: String,
    #[field_type = "Bytes"]
    binary_data: String,
    #[field_type = "Vector"]
    embedding: Vec<f32>,
}

// 测试重命名和唯一性约束
#[derive(Debug, Serialize, Deserialize, AndaDBSchema)]
struct TestConstraints {
    #[unique]
    #[serde(rename = "user_id")]
    id: String,

    #[unique]
    email: String,

    /// User's full name with description
    #[serde(rename = "full_name")]
    name: String,

    /// Optional bio information
    bio: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generated_schema() {
        let schema = TestUser::schema().unwrap();
        println!("{schema:#?}");

        // 验证字段数量 (包含 _id 字段)
        assert_eq!(schema.len(), 7);

        // 验证 handle 字段
        let handle_field = schema.get_field("handle").unwrap();
        assert_eq!(handle_field.r#type(), &FieldType::Text);
        assert!(handle_field.unique());
        assert!(handle_field.required());

        // 验证 name 字段
        let name_field = schema.get_field("name").unwrap();
        assert_eq!(name_field.r#type(), &FieldType::Text);
        assert!(!name_field.unique());
        assert!(name_field.required());

        // 验证 age 字段 (Optional)
        let age_field = schema.get_field("age").unwrap();
        if let FieldType::Option(inner) = age_field.r#type() {
            assert_eq!(**inner, FieldType::U64);
        } else {
            panic!("Expected Option<U64>");
        }
        assert!(!age_field.required());

        // 验证 active 字段
        let active_field = schema.get_field("active").unwrap();
        assert_eq!(active_field.r#type(), &FieldType::Bool);
        assert!(active_field.required());

        // 验证 tags 字段
        let tags_field = schema.get_field("tags").unwrap();
        if let FieldType::Array(types) = tags_field.r#type() {
            assert_eq!(types.len(), 1);
            assert_eq!(types[0], FieldType::Text);
        } else {
            panic!("Expected Array<Text>");
        }

        // 验证 meta 字段 (重命名为 metadata)
        let meta_field = schema.get_field("metadata").unwrap();
        if let FieldType::Option(inner) = meta_field.r#type() {
            if let FieldType::Map(map_types) = inner.as_ref() {
                assert_eq!(map_types.len(), 1);
                assert_eq!(map_types.get("*"), Some(&FieldType::U64));
            } else {
                panic!("Expected Map");
            }
        } else {
            panic!("Expected Option<Map>");
        }
    }

    #[test]
    fn test_schema_with_id_field() {
        let schema = TestUserWithId::schema().unwrap();

        assert_eq!(schema.len(), 3);

        // 验证 username 字段
        let username_field = schema.get_field("username").unwrap();
        assert_eq!(username_field.r#type(), &FieldType::Text);

        // 验证 email 字段
        let email_field = schema.get_field("email").unwrap();
        assert_eq!(email_field.r#type(), &FieldType::Text);

        // 确认 _id 字段在 schema 中
        assert!(schema.get_field("_id").is_some());
    }

    #[test]
    fn test_all_data_types() {
        let schema = TestAllTypes::schema().unwrap();

        // 验证数字类型
        assert_eq!(
            schema.get_field("byte_val").unwrap().r#type(),
            &FieldType::U64
        );
        assert_eq!(
            schema.get_field("short_val").unwrap().r#type(),
            &FieldType::U64
        );
        assert_eq!(
            schema.get_field("int_val").unwrap().r#type(),
            &FieldType::U64
        );
        assert_eq!(
            schema.get_field("long_val").unwrap().r#type(),
            &FieldType::U64
        );

        assert_eq!(
            schema.get_field("signed_byte").unwrap().r#type(),
            &FieldType::I64
        );
        assert_eq!(
            schema.get_field("signed_short").unwrap().r#type(),
            &FieldType::I64
        );
        assert_eq!(
            schema.get_field("signed_int").unwrap().r#type(),
            &FieldType::I64
        );
        assert_eq!(
            schema.get_field("signed_long").unwrap().r#type(),
            &FieldType::I64
        );

        assert_eq!(
            schema.get_field("float_val").unwrap().r#type(),
            &FieldType::F32
        );
        assert_eq!(
            schema.get_field("double_val").unwrap().r#type(),
            &FieldType::F64
        );

        // 验证其他基本类型
        assert_eq!(schema.get_field("text").unwrap().r#type(), &FieldType::Text);
        assert_eq!(schema.get_field("flag").unwrap().r#type(), &FieldType::Bool);
        assert_eq!(
            schema.get_field("data").unwrap().r#type(),
            &FieldType::Bytes
        );

        // 验证数组类型
        let numbers_field = schema.get_field("numbers").unwrap();
        if let FieldType::Array(types) = numbers_field.r#type() {
            assert_eq!(types[0], FieldType::I64);
        } else {
            panic!("Expected Array<I64>");
        }

        let strings_field = schema.get_field("strings").unwrap();
        if let FieldType::Array(types) = strings_field.r#type() {
            assert_eq!(types[0], FieldType::Text);
        } else {
            panic!("Expected Array<Text>");
        }

        // 验证可选类型
        let optional_text_field = schema.get_field("optional_text").unwrap();
        if let FieldType::Option(inner) = optional_text_field.r#type() {
            assert_eq!(**inner, FieldType::Text);
        } else {
            panic!("Expected Option<Text>");
        }

        // 验证 Map 类型
        let string_map_field = schema.get_field("string_map").unwrap();
        if let FieldType::Map(map_types) = string_map_field.r#type() {
            assert_eq!(map_types.get("*"), Some(&FieldType::Text));
        } else {
            panic!("Expected Map<String, String>");
        }
    }

    #[test]
    fn test_custom_field_types() {
        let schema = TestCustomFieldType::schema().unwrap();

        // 验证自定义字段类型
        assert_eq!(
            schema.get_field("custom_field").unwrap().r#type(),
            &FieldType::Json
        );
        assert_eq!(
            schema.get_field("binary_data").unwrap().r#type(),
            &FieldType::Bytes
        );
        assert_eq!(
            schema.get_field("embedding").unwrap().r#type(),
            &FieldType::Vector
        );
    }

    #[test]
    fn test_constraints_and_renaming() {
        let schema = TestConstraints::schema().unwrap();

        // 验证重命名字段
        let id_field = schema.get_field("user_id").unwrap();
        assert_eq!(id_field.r#type(), &FieldType::Text);
        assert!(id_field.unique());

        // 验证唯一性约束
        let email_field = schema.get_field("email").unwrap();
        assert!(email_field.unique());

        // 验证可选字段
        let bio_field = schema.get_field("bio").unwrap();
        if let FieldType::Option(inner) = bio_field.r#type() {
            assert_eq!(**inner, FieldType::Text);
        } else {
            panic!("Expected Option<Text>");
        }
        assert!(!bio_field.required());
    }

    #[test]
    fn test_schema_errors() {
        // 这个测试需要在编译时进行，无法在运行时测试
        // 但我们可以确保正常的 schema 生成不会出错
        assert!(TestUser::schema().is_ok());
        assert!(TestAllTypes::schema().is_ok());
        assert!(TestCustomFieldType::schema().is_ok());
        assert!(TestConstraints::schema().is_ok());
    }

    #[test]
    fn test_field_requirements() {
        let schema = TestUser::schema().unwrap();

        // 必需字段
        assert!(schema.get_field("handle").unwrap().required());
        assert!(schema.get_field("name").unwrap().required());
        assert!(schema.get_field("active").unwrap().required());
        assert!(schema.get_field("tags").unwrap().required());

        // 可选字段
        assert!(!schema.get_field("age").unwrap().required());
        assert!(!schema.get_field("metadata").unwrap().required());
    }
}
