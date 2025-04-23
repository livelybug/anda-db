use anda_db_derive::FieldTyped;
use anda_db_schema::FieldType;
use half::bf16;
use ic_auth_types::Xid;
use serde::{Deserialize, Serialize};
use serde_bytes::{ByteArray, ByteBuf};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Serialize, Deserialize, FieldTyped)]
struct User {
    name: String,
    age: u32,
    tags: HashMap<String, String>,               // 会被正确映射为 Map
    properties: BTreeMap<String, Vec<u8>>,       // 会被正确映射为 Map 包含 Bytes
    optional_data: Option<HashMap<String, f64>>, // 会被正确映射为 Option<Map>
    vector1: Vec<bf16>,                          // 会被正确映射为 Vector

    #[serde(rename = "b1")]
    blob1: ByteArray<64>, // 会被正确映射为 Bytes
    blob2: ByteBuf, // 会被正确映射为 Bytes
}

#[derive(Debug, Serialize, Deserialize, FieldTyped)]
struct Doc {
    #[field_type = "Bytes"] // 将 Xid 类型映射为 FieldType::Bytes
    id: Xid,

    #[field_type = "Option<Array<Bytes>>"]
    #[serde(rename = "ids")]
    user_ids: Option<Vec<Xid>>,
    user: User,
}

#[test]
fn field_typed_derive_works() {
    let user_ft = User::field_type();
    assert_eq!(
        user_ft,
        FieldType::Map(
            vec![
                ("name".to_string(), FieldType::Text),
                ("age".to_string(), FieldType::U64),
                (
                    "tags".to_string(),
                    FieldType::Map(std::collections::BTreeMap::from([(
                        "*".to_string(),
                        FieldType::Text
                    )]))
                ),
                (
                    "properties".to_string(),
                    FieldType::Map(std::collections::BTreeMap::from([(
                        "*".to_string(),
                        FieldType::Bytes
                    )]))
                ),
                (
                    "optional_data".to_string(),
                    FieldType::Option(Box::new(FieldType::Map(std::collections::BTreeMap::from(
                        [("*".to_string(), FieldType::F64)]
                    ))))
                ),
                ("vector1".to_string(), FieldType::Vector),
                ("b1".to_string(), FieldType::Bytes),
                ("blob2".to_string(), FieldType::Bytes),
            ]
            .into_iter()
            .collect()
        )
    );

    let doc_ft = Doc::field_type();
    assert_eq!(
        doc_ft,
        FieldType::Map(
            vec![
                ("id".to_string(), FieldType::Bytes),
                (
                    "ids".to_string(),
                    FieldType::Option(Box::new(FieldType::Array(vec![FieldType::Bytes])))
                ),
                ("user".to_string(), user_ft),
            ]
            .into_iter()
            .collect()
        )
    );
}
