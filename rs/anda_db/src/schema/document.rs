use ciborium::Value;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::Arc;

pub use ic_auth_types::Xid;

use super::{Cbor, Fe, Ft, Fv, IndexedFieldValues, Schema, SchemaError, Segment};

/// Document represents a single document in the Anda DB.
/// It contains an ID, a set of fields, and a reference to its schema.
#[derive(Clone, Debug)]
pub struct Document {
    /// Unique identifier for the document
    pub id: Xid,
    /// Collection of field values indexed by their position in the schema
    pub fields: IndexedFieldValues,
    /// Reference to the schema that defines the document structure
    schema: Arc<Schema>,
}

/// DocumentOwned represents a standalone document without schema reference.
/// It can be serialized and deserialized for storage or transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentOwned {
    /// Collection of field values indexed by their position in the schema.
    /// It should include the ID field.
    #[serde(rename = "f")]
    pub fields: IndexedFieldValues,
}

#[derive(Clone, Debug, Serialize)]
struct DocumentRef<'a> {
    #[serde(rename = "f")]
    pub fields: &'a IndexedFieldValues,
}

impl Default for DocumentOwned {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentOwned {
    /// Creates a new empty DocumentOwned instance.
    ///
    /// # Returns
    /// A new DocumentOwned with empty fields.
    pub fn new() -> Self {
        Self {
            fields: IndexedFieldValues::new(),
        }
    }

    /// Creates a new DocumentOwned instance with the specified ID.
    ///
    /// # Arguments
    /// * `id` - The unique identifier for the document
    ///
    /// # Returns
    /// A new DocumentOwned with the ID field set.
    pub fn with_id(id: Xid) -> Self {
        Self {
            fields: IndexedFieldValues::from([(0, Fv::Bytes((*id).into()))]),
        }
    }

    /// Gets a field value by index or returns an error if it doesn't exist.
    ///
    /// # Arguments
    /// * `idx` - The index of the field to retrieve
    ///
    /// # Returns
    /// * `Result<&Fv, SchemaError>` - The field value or an error if not found
    pub fn get_field_or_err(&self, idx: usize) -> Result<&Fv, SchemaError> {
        self.fields.get(&idx).ok_or_else(|| {
            SchemaError::Validation(format!("field at {:?} not found in document", idx))
        })
    }

    /// Gets a field value by index and deserializes it to the specified type.
    ///
    /// # Arguments
    /// * `idx` - The index of the field to retrieve
    ///
    /// # Returns
    /// * `Result<T, SchemaError>` - The deserialized value or an error
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize the field value to
    pub fn get_field_as<T>(&self, idx: usize) -> Result<T, SchemaError>
    where
        T: DeserializeOwned,
    {
        let value = self.get_field_or_err(idx)?;
        value.to_owned().deserialized()
    }

    /// Sets a field value by serializing the provided value.
    ///
    /// # Arguments
    /// * `idx` - The index of the field to set
    /// * `value` - The value to serialize and store
    /// * `ft` - Optional field type for validation during serialization
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Success or an error
    ///
    /// # Type Parameters
    /// * `T` - The type of the value to serialize
    pub fn set_field_as<T>(
        &mut self,
        idx: usize,
        value: &T,
        ft: Option<&Ft>,
    ) -> Result<(), SchemaError>
    where
        T: Serialize,
    {
        let value = Fv::serialized(&value, ft)?;
        self.fields.insert(idx, value);
        Ok(())
    }
}

impl From<Document> for DocumentOwned {
    /// Converts a Document to DocumentOwned.
    ///
    /// # Arguments
    /// * `doc` - The Document to convert
    ///
    /// # Returns
    /// A new DocumentOwned containing the fields from the Document
    fn from(doc: Document) -> Self {
        Self { fields: doc.fields }
    }
}

impl Document {
    /// Creates a new Document with the specified schema and ID.
    ///
    /// # Arguments
    /// * `schema` - The schema that defines the document structure
    /// * `id` - The unique identifier for the document
    ///
    /// # Returns
    /// A new Document instance
    pub fn with_id(schema: Arc<Schema>, id: Xid) -> Self {
        let doc = DocumentOwned::with_id(id.clone());
        Self {
            id: id.clone(),
            fields: doc.fields,
            schema,
        }
    }

    /// Creates a Document from a DocumentOwned, validating against the schema.
    ///
    /// # Arguments
    /// * `schema` - The schema to validate against
    /// * `doc` - The DocumentOwned to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The validated Document or an error
    pub fn try_with_doc(schema: Arc<Schema>, doc: DocumentOwned) -> Result<Self, SchemaError> {
        schema.validate(&doc.fields)?;
        let id: Xid = doc.get_field_as(0)?;

        Ok(Self {
            id,
            fields: doc.fields,
            schema,
        })
    }

    /// Creates a Document by serializing and validating a value against the schema.
    ///
    /// # Arguments
    /// * `schema` - The schema to validate against
    /// * `doc` - The value to serialize into a document
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The validated document or an error
    ///
    /// # Type Parameters
    /// * `T` - The type of the value to serialize
    pub fn try_from<T>(schema: Arc<Schema>, doc: &T) -> Result<Self, SchemaError>
    where
        T: Serialize,
    {
        let doc = Value::serialized(doc).map_err(|err| {
            SchemaError::Serialization(format!("failed to serialize document: {err:?}"))
        })?;
        let doc = doc.into_map().map_err(|err| {
            SchemaError::Validation(format!(
                "invalid document, expected CBOR map value, got {err:?}"
            ))
        })?;

        let mut doc_owned = DocumentOwned::new();
        for (k, v) in doc {
            let k = k.into_text().map_err(|err| {
                SchemaError::Validation(format!(
                    "invalid document field key, expected CBOR text value, got {err:?}"
                ))
            })?;

            let field = schema.get_field_or_err(&k)?;
            let value = field.extract(v, false)?;
            doc_owned.fields.insert(field.idx(), value);
        }

        schema.validate(&doc_owned.fields)?;
        Ok(Self {
            id: doc_owned.get_field_as(0)?,
            fields: doc_owned.fields,
            schema,
        })
    }

    /// Deserializes the document into the specified type.
    ///
    /// # Returns
    /// * `Result<T, SchemaError>` - The deserialized value or an error
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize the document to
    pub fn try_as<T>(&self) -> Result<T, SchemaError>
    where
        T: DeserializeOwned,
    {
        let mut doc: Vec<(Cbor, Cbor)> = Vec::with_capacity(self.schema.len() + 1);
        for field in self.schema.iter() {
            if let Some(value) = self.fields.get(&field.idx()) {
                doc.push((field.name().into(), value.clone().into()));
            } else if field.required() {
                return Err(SchemaError::Validation(format!(
                    "field {:?} is required",
                    field.name()
                )));
            }
        }

        if self.schema.get_field("id").is_none() {
            doc.push((
                "id".to_string().into(),
                Cbor::serialized(&self.id).expect("Xid should be serializable"),
            ));
        }

        Cbor::Map(doc)
            .deserialized()
            .map_err(|err| SchemaError::Serialization(format!("Failed to deserialize: {}", err)))
    }

    /// Gets the document's unique identifier.
    ///
    /// # Returns
    /// A reference to the document's ID
    pub fn id(&self) -> &Xid {
        &self.id
    }

    /// Gets the fields of the document.
    pub fn fields(&self) -> &IndexedFieldValues {
        &self.fields
    }

    /// Gets a field value by name.
    ///
    /// # Arguments
    /// * `name` - The name of the field to retrieve
    ///
    /// # Returns
    /// * `Option<&Fv>` - The field value or None if not found
    pub fn get_field(&self, name: &str) -> Option<&Fv> {
        if let Some(field) = self.schema.get_field(name) {
            self.fields.get(&field.idx())
        } else {
            None
        }
    }

    /// Gets a field value by name or returns an error if it doesn't exist.
    ///
    /// # Arguments
    /// * `name` - The name of the field to retrieve
    ///
    /// # Returns
    /// * `Result<&Fv, SchemaError>` - The field value or an error if not found
    pub fn get_field_or_err(&self, name: &str) -> Result<&Fv, SchemaError> {
        if let Some(field) = self.schema.get_field(name) {
            self.fields.get(&field.idx()).ok_or_else(|| {
                SchemaError::Validation(format!(
                    "field {:?} at {} not found in document",
                    name,
                    field.idx()
                ))
            })
        } else {
            Err(SchemaError::Validation(format!(
                "field {:?} not found in schema",
                name
            )))
        }
    }

    /// Sets a field value by name.
    ///
    /// # Arguments
    /// * `name` - The name of the field to set
    /// * `value` - The value to store
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Success or an error
    pub fn set_field(&mut self, name: &str, value: Fv) -> Result<&mut Self, SchemaError> {
        if let Some(field) = self.schema.get_field(name) {
            field.validate(&value)?;
            self.fields.insert(field.idx(), value);
            return Ok(self);
        }

        Err(SchemaError::Validation(format!(
            "field {:?} not found in schema",
            name
        )))
    }

    /// Sets multiple field values at once.
    ///
    /// # Arguments
    /// * `fields` - The field values to set, indexed by their position
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Success or an error
    pub fn set_fields(&mut self, fields: IndexedFieldValues) -> Result<(), SchemaError> {
        for (idx, field) in fields {
            self.fields.insert(idx, field);
        }
        self.schema.validate(&self.fields)?;
        Ok(())
    }

    /// Gets a field value by name and deserializes it to the specified type.
    ///
    /// # Arguments
    /// * `name` - The name of the field to retrieve
    ///
    /// # Returns
    /// * `Result<T, SchemaError>` - The deserialized value or an error
    ///
    /// # Type Parameters
    /// * `T` - The type to deserialize the field value to
    pub fn get_field_as<T>(&self, name: &str) -> Result<T, SchemaError>
    where
        T: DeserializeOwned,
    {
        if let Some(field) = self.schema.get_field(name) {
            if let Some(value) = self.fields.get(&field.idx()) {
                return value.to_owned().deserialized();
            } else {
                return Err(SchemaError::Validation(format!(
                    "field {:?} not found in document",
                    name
                )));
            }
        }
        Err(SchemaError::Validation(format!(
            "field {:?} not found in schema",
            name
        )))
    }

    /// Sets a field value by serializing the provided value.
    ///
    /// # Arguments
    /// * `name` - The name of the field to set
    /// * `value` - The value to serialize and store
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Success or an error
    ///
    /// # Type Parameters
    /// * `T` - The type of the value to serialize
    pub fn set_field_as<T>(&mut self, name: &str, value: &T) -> Result<&mut Self, SchemaError>
    where
        T: Serialize,
    {
        let field = self.schema.get_field_or_err(name)?;
        let value = Fv::serialized(&value, Some(field.r#type()))?;
        field.validate(&value)?;
        self.fields.insert(field.idx(), value);
        Ok(self)
    }

    /// Updates the document with values from a DocumentOwned.
    ///
    /// # Arguments
    /// * `doc` - The DocumentOwned containing the new values
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Success or an error
    pub fn set_doc(&mut self, doc: DocumentOwned) -> Result<(), SchemaError> {
        self.schema.validate(&doc.fields)?;
        self.id = doc.get_field_as(0)?;
        self.fields = doc.fields;

        Ok(())
    }
}

impl Serialize for Document {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let doc = DocumentRef {
            fields: &self.fields,
        };
        doc.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Fe, Ft, Fv};
    use serde::{Deserialize, Serialize};
    use std::{collections::BTreeMap, str::FromStr};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestUser {
        id: Xid, // 9m4e2mr0ui3e8a215n4g
        name: String,
        age: u64,
        active: bool,
        tags: Vec<String>,
        meta: BTreeMap<String, u64>,
    }

    fn create_test_schema() -> Arc<Schema> {
        let mut builder = Schema::builder();

        // 添加 ID 字段
        let id_field = Fe::new("id".to_string(), Ft::Bytes).unwrap();
        builder.add_field(id_field).unwrap();

        // 添加必填字段
        let name_field = Fe::new("name".to_string(), Ft::Text)
            .unwrap()
            .with_required();
        builder.add_field(name_field).unwrap();

        let age_field = Fe::new("age".to_string(), Ft::U64).unwrap().with_required();
        builder.add_field(age_field).unwrap();

        // 添加普通字段
        let active_field = Fe::new("active".to_string(), Ft::Bool).unwrap();
        builder.add_field(active_field).unwrap();

        let tags_field = Fe::new("tags".to_string(), Ft::Array(vec![Ft::Text])).unwrap();
        builder.add_field(tags_field).unwrap();

        let meta_field = Fe::new(
            "meta".to_string(),
            Ft::Map(BTreeMap::from([
                ("created".to_string(), Ft::U64),
                ("updated".to_string(), Ft::U64),
            ])),
        )
        .unwrap();
        builder.add_field(meta_field).unwrap();

        Arc::new(builder.build().unwrap())
    }

    #[test]
    fn test_document_with_id() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let doc = Document::with_id(schema.clone(), id.clone());

        assert_eq!(doc.id(), &id);
        assert!(doc.fields.len() == 1);
        assert_eq!(doc.get_field("id").unwrap(), &Fv::Bytes((*id).into()));
    }

    #[test]
    fn test_document_try_with_doc() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        // 创建有效的字段值
        let mut fields = IndexedFieldValues::new();
        fields.insert(1, Fv::Text("John Doe".to_string()));
        fields.insert(2, Fv::U64(30));

        let mut owned_doc = DocumentOwned { fields };

        assert!(Document::try_with_doc(schema.clone(), owned_doc.clone()).is_err());
        owned_doc.set_field_as(0, &id, Some(&Ft::Bytes)).unwrap();

        let doc = Document::try_with_doc(schema.clone(), owned_doc).unwrap();
        assert_eq!(doc.id(), &id);
        assert_eq!(doc.fields.len(), 3);
        assert_eq!(doc.get_field("age").unwrap(), &Fv::U64(30));
    }

    #[test]
    fn test_document_try_from() {
        let schema = create_test_schema();

        let test_user = TestUser {
            id: Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap(),
            name: "John Doe".to_string(),
            age: 30,
            active: true,
            tags: vec!["user".to_string(), "admin".to_string()],
            meta: BTreeMap::from([
                ("created".to_string(), 1625097600),
                ("updated".to_string(), 1625097600),
            ]),
        };

        let doc = Document::try_from(schema.clone(), &test_user).unwrap();

        assert_eq!(doc.id().to_string(), "9m4e2mr0ui3e8a215n4g");
        assert_eq!(
            doc.get_field("name").unwrap(),
            &Fv::Text("John Doe".to_string())
        );
        assert_eq!(doc.get_field("age").unwrap(), &Fv::U64(30));
        assert_eq!(doc.get_field("active").unwrap(), &Fv::Bool(true));

        // 检查数组字段
        if let Fv::Array(tags) = doc.get_field("tags").unwrap() {
            assert_eq!(tags.len(), 2);
            assert_eq!(tags[0], Fv::Text("user".to_string()));
            assert_eq!(tags[1], Fv::Text("admin".to_string()));
        } else {
            panic!("Expected Array field");
        }

        // 检查映射字段
        if let Fv::Map(meta) = doc.get_field("meta").unwrap() {
            assert_eq!(meta.len(), 2);
            assert_eq!(meta.get("created").unwrap(), &Fv::U64(1625097600));
            assert_eq!(meta.get("updated").unwrap(), &Fv::U64(1625097600));
        } else {
            panic!("Expected Map field");
        }
    }

    #[test]
    fn test_document_try_as() {
        let schema = create_test_schema();

        let test_user = TestUser {
            id: Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap(),
            name: "John Doe".to_string(),
            age: 30,
            active: true,
            tags: vec!["user".to_string(), "admin".to_string()],
            meta: BTreeMap::from([
                ("created".to_string(), 1625097600),
                ("updated".to_string(), 1625097600),
            ]),
        };

        let doc = Document::try_from(schema.clone(), &test_user).unwrap();

        // 测试反序列化回原始结构体
        let deserialized: TestUser = doc.try_as().unwrap();

        assert_eq!(deserialized, test_user);
    }

    #[test]
    fn test_document_get_set_field() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let mut doc = Document::with_id(schema.clone(), id);

        // 测试设置字段
        doc.set_field("name", Fv::Text("John Doe".to_string()))
            .unwrap()
            .set_field("age", Fv::U64(30))
            .unwrap();

        // 测试获取字段
        assert_eq!(
            doc.get_field("name").unwrap(),
            &Fv::Text("John Doe".to_string())
        );
        assert_eq!(doc.get_field("age").unwrap(), &Fv::U64(30));

        // 测试设置不存在的字段
        assert!(
            doc.set_field("unknown", Fv::Text("value".to_string()))
                .is_err()
        );

        // 测试获取不存在的字段
        assert!(doc.get_field("unknown").is_none());
    }

    #[test]
    fn test_document_get_set_field_as() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let mut doc = Document::with_id(schema.clone(), id);

        // 测试设置字段（使用序列化）
        doc.set_field_as("name", &"John Doe".to_string()).unwrap();
        doc.set_field_as("age", &30u64).unwrap();
        doc.set_field_as("active", &true).unwrap();

        // 测试获取字段（使用反序列化）
        let name: String = doc.get_field_as("name").unwrap();
        let age: u64 = doc.get_field_as("age").unwrap();
        let active: bool = doc.get_field_as("active").unwrap();

        assert_eq!(name, "John Doe");
        assert_eq!(age, 30);
        assert!(active);

        // 测试设置不存在的字段
        assert!(doc.set_field_as("unknown", &"value".to_string()).is_err());

        // 测试获取不存在的字段
        let result: Result<String, _> = doc.get_field_as("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_document_set_fields() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let mut doc = Document::with_id(schema.clone(), id.clone());

        // 创建字段集合
        let mut fields = IndexedFieldValues::new();
        fields.insert(1, Fv::Text("John Doe".to_string())); // name
        fields.insert(2, Fv::U64(30)); // age

        // 设置字段集合
        doc.set_fields(fields.clone()).unwrap();

        // 验证字段是否正确设置
        assert_eq!(doc.fields.len(), 3);
        assert_eq!(
            doc.get_field("name").unwrap(),
            &Fv::Text("John Doe".to_string())
        );
        assert_eq!(doc.get_field("age").unwrap(), &Fv::U64(30));

        // 测试设置无效的字段集合（缺少必填字段）
        let mut invalid_fields = IndexedFieldValues::new();
        invalid_fields.insert(1, Fv::Text("John Doe".to_string())); // name，缺少 age

        let mut doc2 = Document::with_id(schema.clone(), id);
        assert!(doc2.set_fields(invalid_fields).is_err());
    }

    #[test]
    fn test_document_set_doc() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let mut doc = Document::with_id(schema.clone(), id.clone());

        // 创建有效的字段值
        let mut fields = IndexedFieldValues::new();
        fields.insert(1, Fv::Text("John Doe".to_string())); // name
        fields.insert(2, Fv::U64(30)); // age

        let mut owned_doc = DocumentOwned { fields };

        assert!(doc.set_doc(owned_doc.clone()).is_err());
        owned_doc.fields.insert(
            0,
            Fv::Bytes((*Xid::from_str("9m5e2mr0ui3e8a215n4g").unwrap()).into()),
        ); // id
        doc.set_doc(owned_doc).unwrap();

        // 验证文档是否正确设置
        assert_eq!(doc.id().to_string(), "9m5e2mr0ui3e8a215n4g");
        assert_eq!(doc.fields.len(), 3);
        assert_eq!(
            doc.get_field("name").unwrap(),
            &Fv::Text("John Doe".to_string())
        );
        assert_eq!(doc.get_field("age").unwrap(), &Fv::U64(30));
    }

    #[test]
    fn test_document_from_to_owned() {
        let schema = create_test_schema();
        let id = Xid::from_str("9m4e2mr0ui3e8a215n4g").unwrap();

        let mut doc = Document::with_id(schema.clone(), id.clone());
        doc.set_field("name", Fv::Text("John Doe".to_string()))
            .unwrap();
        doc.set_field("age", Fv::U64(30)).unwrap();

        // 转换为 DocumentOwned
        let owned: DocumentOwned = doc.into();

        // 验证转换是否正确
        assert_eq!(owned.fields.len(), 3);

        // 转换回 Document
        let doc2 = Document::try_with_doc(schema.clone(), owned).unwrap();

        // 验证转换是否正确
        assert_eq!(doc2.id(), &id);
        assert_eq!(doc2.fields.len(), 3);
        assert_eq!(
            doc2.get_field("name").unwrap(),
            &Fv::Text("John Doe".to_string())
        );
        assert_eq!(doc2.get_field("age").unwrap(), &Fv::U64(30));
    }

    #[test]
    fn test_document_validation_errors() {
        let schema = create_test_schema();

        // 测试缺少必填字段
        let test_user_missing_required = serde_json::json!({
            "id": "9m4e2mr0ui3e8a215n4g",
            "name": "John Doe",
            // 缺少必填的 age 字段
            "active": true
        });

        let result = Document::try_from(schema.clone(), &test_user_missing_required);
        assert!(result.is_err());

        // 测试字段类型不匹配
        let test_user_wrong_type = serde_json::json!({
            "id": "9m4e2mr0ui3e8a215n4g",
            "name": "John Doe",
            "age": "thirty", // 应该是数字
            "active": true
        });

        let result = Document::try_from(schema.clone(), &test_user_wrong_type);
        assert!(result.is_err());
    }
}
