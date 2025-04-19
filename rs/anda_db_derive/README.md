# `anda_db_derive`

A Rust procedural macro crate that provides derive macros for AndaDB schema types.

## Features

- `FieldTyped` derive macro: Automatically generates a `field_type()` function for structs that maps Rust types to AndaDB schema field types.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
anda_db_schema = "0.2"
```

## Usage

### FieldTyped Derive Macro

The `FieldTyped` derive macro analyzes struct fields and their types, mapping them to the appropriate `FieldType` enum variants. It supports common Rust types and handles `Option<T>` wrappers.

#### Supported Type Mappings

| Rust Type                                   | AndaDB FieldType        |
| ------------------------------------------- | ----------------------- |
| `String`, `&str`                            | `FieldType::Text`       |
| `bool`                                      | `FieldType::Bool`       |
| `i8`, `i16`, `i32`, `i64`, `isize`          | `FieldType::I64`        |
| `u8`, `u16`, `u32`, `u64`, `usize`          | `FieldType::U64`        |
| `f32`                                       | `FieldType::F32`        |
| `f64`                                       | `FieldType::F64`        |
| `half::bf16`                                | `FieldType::Bf16`       |
| `Vec<u8>`, `[u8]`, `ByteArray`, `ByteBuf`   | `FieldType::Bytes`      |
| `Vec<T>`                                    | `FieldType::Array`      |
| `Option<T>`                                 | `FieldType::Option`     |
| `HashMap<String, T>`, `BTreeMap<String, T>` | `FieldType::Map`        |
| Custom structs                              | Nested `FieldType::Map` |

#### Attributes

- `field_type = "TypeName"`: Override the inferred type with a specific FieldType
- `serde(rename = "field_name")`: Use the renamed field in the generated schema

#### Example

```rust
use anda_db_schema::{FieldTyped, FieldType, Xid};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Define a struct with the FieldTyped derive macro
#[derive(Serialize, Deserialize, FieldTyped)]
struct User {
    name: String,
    age: u32,

    // HashMap will be correctly mapped to FieldType::Map
    tags: HashMap<String, String>,

    // Optional fields are handled automatically
    email: Option<String>,

    #[field_type = "Array<Bytes>"]
    #[serde(rename = "ids")]
    follow_ids: Vec<Xid>,
}

// Define a nested struct
#[derive(Serialize, Deserialize, FieldTyped)]
struct Document {
    // Custom field type override
    #[field_type = "Bytes"]
    id: Xid,

    #[serde(rename = "c")]
    content: String,

    // Support nested struct
    author: User,
}

fn main() {
    // Get the field type schema for User
    let user_schema = User::field_type();
    println!("User schema: {:?}", user_schema);

    // Get the field type schema for Document
    let doc_schema = Document::field_type();
    println!("Document schema: {:?}", doc_schema);
}
```

The generated `field_type()` method returns a `FieldType::Map` containing the type information for each field, which can be used for schema validation, serialization, or other purposes in AndaDB.

## License
Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](../../LICENSE) for the full license text.
