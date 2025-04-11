use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{FieldEntry, IndexedFieldValues, SchemaError};

/// Schema represents Anda DB document schema definition.
/// It contains a collection of fields and their indexes.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Set of field indexes for quick lookup
    idx: BTreeSet<usize>,
    /// Map of field names to field entries
    fields: BTreeMap<String, FieldEntry>,
}

impl Schema {
    /// The key name for the ID field
    const ID_KEY: &str = "id";

    /// Creates a new SchemaBuilder instance.
    ///
    /// # Returns
    /// A new SchemaBuilder with default settings.
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    /// Returns the number of fields in the schema.
    ///
    /// # Returns
    /// The number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Checks if the schema has no fields.
    ///
    /// # Returns
    /// `true` if the schema has no fields, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Gets a field by name.
    ///
    /// # Arguments
    /// * `name` - The name of the field to get.
    ///
    /// # Returns
    /// Some(&FieldEntry) if the field exists, None otherwise.
    pub fn get_field(&self, name: &str) -> Option<&FieldEntry> {
        self.fields.get(name)
    }

    /// Gets a field by name or returns an error if it doesn't exist.
    ///
    /// # Arguments
    /// * `name` - The name of the field to get.
    ///
    /// # Returns
    /// Ok(&FieldEntry) if the field exists, Err(SchemaError) otherwise.
    pub fn get_field_or_err(&self, name: &str) -> Result<&FieldEntry, SchemaError> {
        self.fields
            .get(name)
            .ok_or_else(|| SchemaError::Validation(format!("field {:?} not found in schema", name)))
    }

    /// Returns an iterator over all fields in the schema.
    ///
    /// # Returns
    /// An iterator yielding references to FieldEntry.
    pub fn iter(&self) -> impl Iterator<Item = &FieldEntry> {
        self.fields.values()
    }

    /// Validates a set of field values against this schema.
    ///
    /// # Arguments
    /// * `values` - The field values to validate.
    ///
    /// # Returns
    /// Ok(()) if validation succeeds, Err(SchemaError) otherwise.
    ///
    /// # Errors
    /// Returns an error if:
    /// - A field index in values doesn't exist in the schema
    /// - A required field is missing
    /// - A field value doesn't match the field type
    pub fn validate(&self, values: &IndexedFieldValues) -> Result<(), SchemaError> {
        // Validate that all field indexes in values exist in the schema
        for idx in values.keys() {
            if !self.idx.contains(idx) {
                return Err(SchemaError::Validation(format!(
                    "field index {:?} not found in schema",
                    idx
                )));
            }
        }

        // Validate each field's value and check for required fields
        for field in self.fields.values() {
            if let Some(value) = values.get(&field.idx()) {
                field.validate(value)?;
            } else if field.required() {
                return Err(SchemaError::Validation(format!(
                    "field {:?} is required",
                    field.name()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
struct SchemaRef<'a> {
    fields: Vec<&'a FieldEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct SchemaOwned {
    fields: Vec<FieldEntry>,
}

impl Serialize for Schema {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let val = SchemaRef {
            fields: self.fields.values().collect(),
        };
        val.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let val = SchemaOwned::deserialize(deserializer)?;

        Ok(Schema {
            idx: val.fields.iter().map(|f| f.idx()).collect(),
            fields: val
                .fields
                .into_iter()
                .map(|f| (f.name().to_string(), f))
                .collect(),
        })
    }
}

/// SchemaBuilder is used to construct a Schema instance.
/// It provides methods to add fields and build the final schema.
#[derive(Clone, Debug, Default)]
pub struct SchemaBuilder {
    idx: usize,
    fields: BTreeMap<String, FieldEntry>,
}

impl SchemaBuilder {
    /// Creates a new SchemaBuilder instance.
    ///
    /// # Returns
    /// A new SchemaBuilder with default settings.
    pub fn new() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    /// Adds a field to the schema.
    ///
    /// # Arguments
    /// * `entry` - The field entry to add.
    ///
    /// # Returns
    /// Ok(()) if the field was added successfully, Err(SchemaError) otherwise.
    ///
    /// # Errors
    /// Returns an error if:
    /// - A field with the same name already exists
    /// - The maximum number of fields has been reached
    pub fn add_field(&mut self, entry: FieldEntry) -> Result<(), SchemaError> {
        if self.fields.contains_key(entry.name()) {
            return Err(SchemaError::Schema(format!(
                "Field {:?} already exists in schema",
                entry.name()
            )));
        }

        if entry.name() == Schema::ID_KEY {
            // ID field is always the first field at index 0
            self.fields.insert(
                Schema::ID_KEY.to_string(),
                entry.with_required().with_unique().with_idx(0),
            );
            return Ok(());
        }

        self.idx += 1;
        if self.idx > u16::MAX as usize {
            return Err(SchemaError::Schema(
                "Schema has reached the maximum number of fields".to_string(),
            ));
        }

        self.fields
            .insert(entry.name().to_string(), entry.with_idx(self.idx));
        Ok(())
    }

    /// Builds the final Schema from this builder.
    ///
    /// # Returns
    /// Ok(Schema) if the schema is valid, Err(SchemaError) otherwise.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The schema has no fields
    /// - The schema has too many fields (more than u8::MAX)
    pub fn build(self) -> Result<Schema, SchemaError> {
        if self.fields.is_empty() {
            return Err(SchemaError::Schema(
                "Schema must have at least one field".to_string(),
            ));
        }

        if self.fields.len() > u8::MAX as usize {
            return Err(SchemaError::Schema(
                "Schema has reached the maximum number of fields".to_string(),
            ));
        }

        Ok(Schema {
            idx: self.fields.values().map(|f| f.idx()).collect(),
            fields: self.fields,
        })
    }
}

impl PartialEq for Schema {
    /// Compares two Schema instances for equality.
    /// Two schemas are equal if they have the same fields.
    fn eq(&self, other: &Schema) -> bool {
        self.fields == other.fields
    }
}

impl Eq for Schema {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Fe, Ft, Fv};

    #[test]
    fn test_schema_builder() {
        let mut builder = SchemaBuilder::new();

        // 测试添加 ID 字段
        let id_field = Fe::new("id".to_string(), Ft::Text).unwrap();
        assert!(builder.add_field(id_field).is_ok());

        // 测试添加普通字段
        let name_field = Fe::new("name".to_string(), Ft::Text).unwrap();
        assert!(builder.add_field(name_field).is_ok());

        let age_field = Fe::new("age".to_string(), Ft::U64).unwrap().with_required();
        assert!(builder.add_field(age_field).is_ok());

        // 测试添加重复字段
        let duplicate_field = Fe::new("name".to_string(), Ft::Text).unwrap();
        assert!(builder.add_field(duplicate_field).is_err());

        // 构建 Schema
        let schema = builder.build().unwrap();

        // 验证 Schema 字段数量
        assert_eq!(schema.len(), 3);
        assert!(!schema.is_empty());

        // 验证字段索引
        assert!(schema.idx.contains(&0)); // id
        assert!(schema.idx.contains(&1)); // name
        assert!(schema.idx.contains(&2)); // age

        // 验证获取字段
        let id_field = schema.get_field("id").unwrap();
        assert_eq!(id_field.name(), "id");
        assert_eq!(id_field.idx(), 0);
        assert!(id_field.required());
        assert!(id_field.unique());

        let name_field = schema.get_field("name").unwrap();
        assert_eq!(name_field.name(), "name");
        assert_eq!(name_field.idx(), 1);
        assert!(!name_field.required());

        let age_field = schema.get_field("age").unwrap();
        assert_eq!(age_field.name(), "age");
        assert_eq!(age_field.idx(), 2);
        assert!(age_field.required());

        // 测试不存在的字段
        assert!(schema.get_field("unknown").is_none());
    }

    #[test]
    fn test_schema_validation() {
        let mut builder = SchemaBuilder::new();

        // 添加字段
        let id_field = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder.add_field(id_field).unwrap();

        let name_field = Fe::new("name".to_string(), Ft::Text).unwrap();
        builder.add_field(name_field).unwrap();

        let age_field = Fe::new("age".to_string(), Ft::U64).unwrap().with_required();
        builder.add_field(age_field).unwrap();

        let schema = builder.build().unwrap();

        // 创建有效的字段值
        let mut valid_values = IndexedFieldValues::new();
        valid_values.insert(0, Fv::Text("user1".to_string()));
        valid_values.insert(1, Fv::Text("John".to_string()));
        valid_values.insert(2, Fv::U64(30));

        // 验证有效值
        assert!(schema.validate(&valid_values).is_ok());

        // 缺少必填字段
        let mut missing_required = IndexedFieldValues::new();
        missing_required.insert(0, Fv::Text("user1".to_string()));
        missing_required.insert(1, Fv::Text("John".to_string()));
        // 缺少 age 字段
        assert!(schema.validate(&missing_required).is_err());

        // 无效的字段索引
        let mut invalid_index = IndexedFieldValues::new();
        invalid_index.insert(0, Fv::Text("user1".to_string()));
        invalid_index.insert(1, Fv::Text("John".to_string()));
        invalid_index.insert(2, Fv::U64(30));
        invalid_index.insert(99, Fv::Text("Invalid".to_string())); // 无效索引
        assert!(schema.validate(&invalid_index).is_err());

        // 字段类型不匹配
        let mut invalid_type = IndexedFieldValues::new();
        invalid_type.insert(0, Fv::Text("user1".to_string()));
        invalid_type.insert(1, Fv::Text("John".to_string()));
        invalid_type.insert(2, Fv::Text("30".to_string())); // 应该是 Integer
        assert!(schema.validate(&invalid_type).is_err());
    }

    #[test]
    fn test_schema_builder_limits() {
        // 测试空 Schema
        let empty_builder = SchemaBuilder::new();
        assert!(empty_builder.build().is_err());

        // 测试最大字段数限制
        let mut builder = SchemaBuilder::new();
        let id_field = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder.add_field(id_field).unwrap();

        // 设置 idx 接近 u16::MAX
        builder.idx = u16::MAX as usize - 1;
        let test_field = Fe::new("test".to_string(), Ft::Text).unwrap();
        assert!(builder.add_field(test_field).is_ok());

        // 添加超过限制的字段
        let overflow_field = Fe::new("overflow".to_string(), Ft::Text).unwrap();
        assert!(builder.add_field(overflow_field).is_err());
    }

    #[test]
    fn test_schema_equality() {
        let mut builder1 = SchemaBuilder::new();
        let id_field1 = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder1.add_field(id_field1).unwrap();
        let name_field1 = Fe::new("name".to_string(), Ft::Text).unwrap();
        builder1.add_field(name_field1).unwrap();
        let schema1 = builder1.build().unwrap();

        let mut builder2 = SchemaBuilder::new();
        let id_field2 = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder2.add_field(id_field2).unwrap();
        let name_field2 = Fe::new("name".to_string(), Ft::Text).unwrap();
        builder2.add_field(name_field2).unwrap();
        let schema2 = builder2.build().unwrap();

        // 相同结构的 Schema 应该相等
        assert_eq!(schema1, schema2);

        // 不同结构的 Schema
        let mut builder3 = SchemaBuilder::new();
        let id_field3 = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder3.add_field(id_field3).unwrap();
        let age_field3 = Fe::new("age".to_string(), Ft::U64).unwrap();
        builder3.add_field(age_field3).unwrap();
        let schema3 = builder3.build().unwrap();

        assert_ne!(schema1, schema3);
    }

    #[test]
    fn test_schema_iter() {
        let mut builder = SchemaBuilder::new();
        let id_field = Fe::new("id".to_string(), Ft::Text).unwrap();
        builder.add_field(id_field).unwrap();
        let name_field = Fe::new("name".to_string(), Ft::Text).unwrap();
        builder.add_field(name_field).unwrap();
        let schema = builder.build().unwrap();

        let fields: Vec<&FieldEntry> = schema.iter().collect();
        assert_eq!(fields.len(), 2);

        // 验证迭代器返回的字段
        let field_names: Vec<&str> = fields.iter().map(|f| f.name()).collect();
        assert!(field_names.contains(&"id"));
        assert!(field_names.contains(&"name"));
    }
}
