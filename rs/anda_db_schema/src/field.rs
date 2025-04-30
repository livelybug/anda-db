//! Field type and value definitions for Anda DB, stored in CBOR format.
//!
//! This module defines the field type system for Anda DB, including:
//! - [`FieldType`] or [`Ft`]: Field type definitions, supporting basic types, composite types, and optional types
//! - [`FieldValue`] or [`Fv`]: Field value definitions, corresponding to actual values of various types
//! - [`FieldEntry`] or [`Fe`]: Field entry definitions, containing field name, type, constraints, and other information
//!
//! All data is serialized into CBOR format for storage to improve storage efficiency and cross-platform compatibility.
//!
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{collections::BTreeMap, fmt};

use crate::{BoxError, SchemaError, validate_field_name};

/// Re-export bf16 from half crate
pub use half::bf16;

/// Type alias for Vec<bf16>
pub type Vector = Vec<bf16>;

/// Type alias for FieldType
pub type Ft = FieldType;

/// Type alias for FieldValue
pub type Fv = FieldValue;

/// Type alias for FieldEntry
pub type Fe = FieldEntry;

/// Type alias for ciborium::Value
pub type Cbor = ciborium::Value;

/// Type alias for serde_json::Value
pub type Json = serde_json::Value;

/// Type alias for BTreeMap<usize, FieldValue>, used to store indexed field values
pub type IndexedFieldValues = BTreeMap<usize, FieldValue>;

/// Field type definitions for Anda DB
///
/// Supports various basic types (U64, I64, F64, F32, Bf16, Bytes, Text, Bool, Json)
/// and composite types (Array, Map, Option)
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    /// Boolean value
    Bool,
    /// Signed 64-bit integer
    I64,
    /// Unsigned 64-bit integer
    U64,
    /// 64-bit floating point number
    F64,
    /// 32-bit floating point number
    F32,
    /// Binary data
    Bytes,
    /// UTF-8 encoded text
    Text,
    /// JSON value
    Json,
    /// Vec<bf16>, bf16: 16-bit floating point type implementing the bfloat16 format.
    /// Detail: https://docs.rs/half/latest/half/struct.bf16.html
    Vector,
    /// Array of field types
    Array(Vec<FieldType>),
    /// Map with string keys and field type values
    Map(BTreeMap<String, FieldType>),
    /// Optional field type
    Option(Box<FieldType>),
}

impl fmt::Debug for FieldType {
    /// Debug formatting for FieldType
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldType::Bool => write!(f, "Bool"),
            FieldType::I64 => write!(f, "I64"),
            FieldType::U64 => write!(f, "U64"),
            FieldType::F64 => write!(f, "F64"),
            FieldType::F32 => write!(f, "F32"),
            FieldType::Bytes => write!(f, "Bytes"),
            FieldType::Text => write!(f, "Text"),
            FieldType::Json => write!(f, "Json"),
            FieldType::Vector => write!(f, "Vector"),
            FieldType::Array(v) => write!(f, "Array({:?})", v),
            FieldType::Map(v) => write!(f, "Map({:?})", v),
            FieldType::Option(v) => write!(f, "Option({:?})", v),
        }
    }
}

impl FieldType {
    /// Extract a FieldValue from a CBOR value according to this field type
    ///
    /// # Arguments
    /// * `value` - The CBOR value to extract from
    ///
    /// # Returns
    /// * `Result<FieldValue, SchemaError>` - The extracted field value or an error message
    pub fn extract(&self, value: Cbor) -> Result<FieldValue, SchemaError> {
        match &self {
            FieldType::Bool => FieldValue::bool_from(value),
            FieldType::I64 => FieldValue::i64_from(value),
            FieldType::U64 => FieldValue::u64_from(value),
            FieldType::F64 => FieldValue::f64_from(value),
            FieldType::F32 => FieldValue::f32_from(value),
            FieldType::Bytes => FieldValue::bytes_from(value),
            FieldType::Text => FieldValue::text_from(value),
            FieldType::Json => FieldValue::json_from(value),
            FieldType::Vector => FieldValue::vector_from(value),
            FieldType::Array(types) => FieldValue::array_from(value, types),
            FieldType::Map(types) => FieldValue::map_from(value, types),
            FieldType::Option(ft) => {
                if value == Cbor::Null {
                    return Ok(FieldValue::Null);
                }
                ft.extract(value)
            }
        }
    }

    /// Validate if a FieldValue matches this field type
    ///
    /// # Arguments
    /// * `value` - The FieldValue to validate
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Ok if valid, or an error message if invalid
    pub fn validate(&self, value: &FieldValue) -> Result<(), SchemaError> {
        match (self, value) {
            (FieldType::Bool, FieldValue::Bool(_)) => Ok(()),
            (FieldType::I64, FieldValue::I64(_)) => Ok(()),
            (FieldType::U64, FieldValue::U64(_)) => Ok(()),
            (FieldType::F64, FieldValue::F64(_)) => Ok(()),
            (FieldType::F32, FieldValue::F32(_)) => Ok(()),
            (FieldType::Bytes, FieldValue::Bytes(_)) => Ok(()),
            (FieldType::Text, FieldValue::Text(_)) => Ok(()),
            (FieldType::Json, FieldValue::Json(_)) => Ok(()),
            (FieldType::Vector, FieldValue::Vector(_)) => Ok(()),
            (FieldType::Vector, FieldValue::Array(values)) => {
                if let Some(FieldValue::U64(_)) = values.iter().next() {
                    return Ok(());
                }
                Err(SchemaError::FieldValue(format!(
                    "expected Vector, got {values:?}"
                )))
            }
            (FieldType::Array(types), FieldValue::Array(values)) => match types.len() {
                0 => Ok(()),
                1 => {
                    let ft = types.first().unwrap();
                    for fv in values.iter() {
                        ft.validate(fv)?;
                    }
                    Ok(())
                }
                _ => {
                    if values.len() != types.len() {
                        return Err(SchemaError::FieldValue(format!(
                            "invalid array length, expected {:?}, got {:?}",
                            types.len(),
                            values.len()
                        )));
                    }

                    for (i, ft) in types.iter().enumerate() {
                        if let Some(fv) = values.get(i) {
                            ft.validate(fv)?;
                        } else {
                            return Err(SchemaError::FieldValue(format!(
                                "no value at array[{}], expected type {:?}",
                                i, ft,
                            )));
                        }
                    }
                    Ok(())
                }
            },
            (FieldType::Map(types), FieldValue::Map(values)) => match types.len() {
                0 => Ok(()),
                1 => {
                    let ft = types.values().next().unwrap();
                    for fv in values.iter() {
                        ft.validate(fv.1)?;
                    }
                    Ok(())
                }
                _ => {
                    if let Some(k) = values.keys().find(|k| !types.contains_key(*k)) {
                        return Err(SchemaError::FieldValue(format!("invalid map key {:?}", k)));
                    }

                    for (k, ft) in types.iter() {
                        ft.validate(values.get(k).unwrap_or(&FieldValue::Null))
                            .map_err(|err| {
                                SchemaError::FieldValue(format!(
                                    "invalid map value at key {:?}, error: {}",
                                    k, err
                                ))
                            })?;
                    }
                    Ok(())
                }
            },
            (FieldType::Option(ft), val) => {
                if val == &FieldValue::Null {
                    return Ok(());
                }
                ft.validate(val)
            }
            _ => Err(SchemaError::FieldValue(format!(
                "expected type {:?}, got value {:?}",
                self, value
            ))),
        }
    }
}

/// Field value definitions for Anda DB
///
/// Corresponds to the various field types, storing actual data values
#[derive(Clone, PartialEq)]
pub enum FieldValue {
    /// Boolean value
    Bool(bool),
    /// Signed 64-bit integer value
    I64(i64),
    /// Unsigned 64-bit integer value
    U64(u64),
    /// 64-bit floating point value
    F64(f64),
    /// 32-bit floating point value
    F32(f32),
    /// Binary data value
    Bytes(Vec<u8>),
    /// UTF-8 encoded text value
    Text(String),
    /// JSON value
    Json(Json),
    /// Vec<bf16>, bf16: 16-bit floating point type implementing the bfloat16 format.
    /// Detail: https://docs.rs/half/latest/half/struct.bf16.html
    Vector(Vec<bf16>),
    /// Array of field values
    Array(Vec<FieldValue>),
    /// Map with string keys and field values
    Map(BTreeMap<String, FieldValue>),
    /// Null value (for optional fields)
    Null,
}

impl From<FieldValue> for Cbor {
    /// Convert a FieldValue to a CBOR value
    ///
    /// # Arguments
    /// * `value` - The FieldValue to convert
    ///
    /// # Returns
    /// * `Cbor` - The converted CBOR value
    fn from(value: FieldValue) -> Self {
        match value {
            FieldValue::Bool(b) => Cbor::Bool(b),
            FieldValue::I64(i) => Cbor::Integer(i.into()),
            FieldValue::U64(u) => Cbor::Integer(u.into()),
            FieldValue::F64(f) => Cbor::Float(f),
            FieldValue::F32(f) => Cbor::Float(f as f64),
            FieldValue::Bytes(b) => Cbor::Bytes(b),
            FieldValue::Text(t) => Cbor::Text(t),
            // JSON value can always be serialized to CBOR format!
            FieldValue::Json(obj) => {
                Cbor::serialized(&obj).expect("Failed to serialize JSON to CBOR")
            }
            FieldValue::Vector(arr) => {
                Cbor::Array(arr.into_iter().map(|f| f.to_bits().into()).collect())
            }
            FieldValue::Array(arr) => Cbor::Array(arr.into_iter().map(Cbor::from).collect()),
            FieldValue::Map(obj) => {
                let obj = obj
                    .into_iter()
                    .map(|(k, v)| (Cbor::Text(k), Cbor::from(v)))
                    .collect();
                Cbor::Map(obj)
            }

            FieldValue::Null => Cbor::Null,
        }
    }
}

impl TryFrom<FieldValue> for bool {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Bool(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Bool, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for bool {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Bool(v) => Ok(*v),
            _ => Err(SchemaError::FieldValue(format!("expected Bool, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for i64 {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::I64(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected I64, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for i64 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::I64(v) => Ok(*v),
            _ => Err(SchemaError::FieldValue(format!("expected I64, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a i64 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::I64(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected I64, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for u64 {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::U64(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected U64, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for u64 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::U64(v) => Ok(*v),
            _ => Err(SchemaError::FieldValue(format!("expected U64, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a u64 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::U64(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected U64, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for f64 {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::F64(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected F64, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for f64 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::F64(v) => Ok(*v),
            _ => Err(SchemaError::FieldValue(format!("expected F64, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for f32 {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::F32(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected F32, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for f32 {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::F32(v) => Ok(*v),
            _ => Err(SchemaError::FieldValue(format!("expected F32, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for Vec<u8> {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Bytes(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Bytes, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a Vec<u8> {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Bytes(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Bytes, got {value:?}")).into()),
        }
    }
}

impl<const N: usize> TryFrom<FieldValue> for [u8; N] {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Bytes(v) => Ok(v.try_into().map_err(|v: Vec<u8>| {
                SchemaError::FieldValue(format!("expected {N} bytes, got {}", v.len()))
            })?),
            _ => Err(SchemaError::FieldValue(format!("expected Bytes, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for String {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Text(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Text, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a String {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Text(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Text, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a str {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Text(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Text, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for Json {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Json(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Json, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a Json {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Json(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Json, got {value:?}")).into()),
        }
    }
}

impl TryFrom<FieldValue> for Vec<bf16> {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Vector(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Vector, got {value:?}")).into()),
        }
    }
}

impl<'a> TryFrom<&'a FieldValue> for &'a Vec<bf16> {
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Vector(v) => Ok(v),
            _ => Err(SchemaError::FieldValue(format!("expected Vector, got {value:?}")).into()),
        }
    }
}

impl<const N: usize> TryFrom<FieldValue> for [bf16; N] {
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Vector(v) => Ok(v.try_into().map_err(|v: Vec<bf16>| {
                SchemaError::FieldValue(format!("expected {N} elements, got {}", v.len()))
            })?),
            _ => Err(SchemaError::FieldValue(format!("expected Bytes, got {value:?}")).into()),
        }
    }
}

impl<T> TryFrom<FieldValue> for Vec<T>
where
    T: TryFrom<FieldValue, Error = BoxError>,
{
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Array(arr) => {
                let mut rt = Vec::with_capacity(arr.len());
                for v in arr {
                    rt.push(v.try_into()?);
                }
                Ok(rt)
            }
            _ => Err(SchemaError::FieldValue(format!("expected Array, got {value:?}")).into()),
        }
    }
}

impl<'a, T> TryFrom<&'a FieldValue> for Vec<&'a T>
where
    &'a T: TryFrom<&'a FieldValue, Error = BoxError>,
{
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Array(arr) => {
                let mut rt = Vec::with_capacity(arr.len());
                for v in arr {
                    rt.push(v.try_into()?);
                }
                Ok(rt)
            }
            _ => Err(SchemaError::FieldValue(format!("expected Array, got {value:?}")).into()),
        }
    }
}

impl<T> TryFrom<FieldValue> for BTreeMap<String, T>
where
    T: TryFrom<FieldValue, Error = BoxError>,
{
    type Error = BoxError;

    fn try_from(value: FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Map(map) => {
                let mut rt = BTreeMap::new();
                for (k, v) in map {
                    rt.insert(k, v.try_into()?);
                }
                Ok(rt)
            }
            _ => Err(SchemaError::FieldValue(format!("expected Map, got {value:?}")).into()),
        }
    }
}

impl<'a, T> TryFrom<&'a FieldValue> for BTreeMap<&'a String, &'a T>
where
    &'a T: TryFrom<&'a FieldValue, Error = BoxError>,
{
    type Error = BoxError;

    fn try_from(value: &'a FieldValue) -> Result<Self, Self::Error> {
        match value {
            FieldValue::Map(map) => {
                let mut rt = BTreeMap::new();
                for (k, v) in map {
                    rt.insert(k, v.try_into()?);
                }
                Ok(rt)
            }
            _ => Err(SchemaError::FieldValue(format!("expected Map, got {value:?}")).into()),
        }
    }
}

impl fmt::Debug for FieldValue {
    /// Debug formatting for FieldValue
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldValue::Bool(v) => write!(f, "Bool({})", v),
            FieldValue::I64(v) => write!(f, "I64({})", v),
            FieldValue::U64(v) => write!(f, "U64({})", v),
            FieldValue::F64(v) => write!(f, "F64({})", v),
            FieldValue::F32(v) => write!(f, "F32({})", v),
            FieldValue::Bytes(v) => write!(f, "Bytes({:?})", v),
            FieldValue::Text(v) => write!(f, "Text({:?})", v),
            FieldValue::Json(v) => write!(f, "Json({:?})", v),
            FieldValue::Vector(v) => write!(f, "Vector({:?})", v),
            FieldValue::Array(v) => write!(f, "Array({:?})", v),
            FieldValue::Map(v) => write!(f, "Map({:?})", v),
            FieldValue::Null => write!(f, "Null"),
        }
    }
}

impl FieldValue {
    /// Create a Bool FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn bool_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Bool(b) => Ok(FieldValue::Bool(b)),
            v => Err(SchemaError::FieldValue(format!("expected Bool, got {v:?}"))),
        }
    }

    /// Create an I64 FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn i64_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Integer(i) => {
                Ok(FieldValue::I64(i.try_into().map_err(|v| {
                    SchemaError::FieldValue(format!("expected I64, got {v:?}"))
                })?))
            }
            v => Err(SchemaError::FieldValue(format!("expected I64, got {v:?}"))),
        }
    }

    /// Create a U64 FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn u64_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Integer(i) => {
                Ok(FieldValue::U64(i.try_into().map_err(|v| {
                    SchemaError::FieldValue(format!("expected U64, got {v:?}"))
                })?))
            }
            v => Err(SchemaError::FieldValue(format!("expected U64, got {v:?}"))),
        }
    }

    /// Create an F64 FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn f64_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Float(f) if !f.is_nan() => Ok(FieldValue::F64(f)),
            v => Err(SchemaError::FieldValue(format!("expected F64, got {v:?}"))),
        }
    }

    /// Create an F32 FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn f32_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Float(f) if !f.is_nan() => Ok(FieldValue::F32(f as f32)),
            v => Err(SchemaError::FieldValue(format!("expected F32, got {v:?}"))),
        }
    }

    /// Create a Bytes FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn bytes_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Bytes(b) => Ok(FieldValue::Bytes(b)),
            v => Err(SchemaError::FieldValue(format!(
                "expected Bytes, got {v:?}"
            ))),
        }
    }

    /// Create a Text FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn text_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Text(t) => Ok(FieldValue::Text(t)),
            v => Err(SchemaError::FieldValue(format!("expected Text, got {v:?}"))),
        }
    }

    /// Create a Json FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn json_from(value: Cbor) -> Result<Self, SchemaError> {
        let val: Json = value
            .deserialized()
            .map_err(|v| SchemaError::FieldValue(format!("expected Json, got {v:?}")))?;
        Ok(FieldValue::Json(val))
    }

    /// Create a Vector FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn vector_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Array(arr) => Ok(FieldValue::Vector(
                arr.into_iter()
                    .map(Self::bf16_from)
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            v => Err(SchemaError::FieldValue(format!(
                "expected Vector, got {v:?}"
            ))),
        }
    }

    /// Create a bf16 from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn bf16_from(value: Cbor) -> Result<bf16, SchemaError> {
        match value {
            Cbor::Integer(i) => {
                Ok(bf16::from_bits(i.try_into().map_err(|v| {
                    SchemaError::FieldValue(format!("expected u16, got {v:?}"))
                })?))
            }
            v => Err(SchemaError::FieldValue(format!("expected bf64, got {v:?}"))),
        }
    }

    /// Create an Array FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    /// * `types` - The field types for the array elements
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn array_from(value: Cbor, types: &[FieldType]) -> Result<Self, SchemaError> {
        match value {
            Cbor::Array(values) => match types.len() {
                0 => Ok(FieldValue::Array(
                    values
                        .into_iter()
                        .map(FieldValue::try_from)
                        .collect::<Result<Vec<_>, _>>()?,
                )),
                1 => {
                    let ft = types.first().unwrap();
                    Ok(FieldValue::Array(
                        values
                            .into_iter()
                            .map(|v| ft.extract(v))
                            .collect::<Result<Vec<_>, _>>()?,
                    ))
                }
                _ => {
                    if types.len() != values.len() {
                        return Err(SchemaError::FieldValue(format!(
                            "Invalid array length, expected {:?}, got {:?}",
                            types.len(),
                            values.len()
                        )));
                    }

                    let mut rt: Vec<FieldValue> = Vec::with_capacity(types.len());
                    for (i, ft) in types.iter().enumerate() {
                        let val = ft.extract(values[i].clone())?;
                        rt.push(val);
                    }

                    Ok(FieldValue::Array(rt))
                }
            },
            v => Err(SchemaError::FieldValue(format!(
                "expected Array, got {v:?}"
            ))),
        }
    }

    /// Create a Map FieldValue from a CBOR value
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    /// * `types` - The field types for the map values, keyed by field name
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn map_from(value: Cbor, types: &BTreeMap<String, FieldType>) -> Result<Self, SchemaError> {
        match value {
            Cbor::Map(values) => {
                if types.is_empty() {
                    return Ok(FieldValue::Map(
                        values
                            .into_iter()
                            .map(|(k, v)| {
                                let k = k.into_text().map_err(|v| {
                                    SchemaError::FieldValue(format!("invalid map key: {:?}", v))
                                })?;
                                Ok::<_, SchemaError>((k, FieldValue::try_from(v)?))
                            })
                            .collect::<Result<BTreeMap<_, _>, _>>()?,
                    ));
                }

                let mut vals: BTreeMap<String, Cbor> = BTreeMap::new();
                for (k, v) in values {
                    let k = k.into_text().map_err(|v| {
                        SchemaError::FieldValue(format!("invalid map key: {:?}", v))
                    })?;
                    if !types.contains_key(&k) {
                        return Err(SchemaError::FieldValue(format!("invalid map key {:?}", k)));
                    }
                    if vals.contains_key(&k) {
                        return Err(SchemaError::FieldValue(format!(
                            "duplicate map key {:?}",
                            k
                        )));
                    }
                    vals.insert(k, v);
                }

                let mut rt: BTreeMap<String, FieldValue> = BTreeMap::new();
                for (k, ft) in types.iter() {
                    let (key, val) = vals
                        .remove_entry(k)
                        .unwrap_or_else(|| (k.clone(), Cbor::Null));
                    let val = ft.extract(val)?;
                    rt.insert(key, val);
                }

                Ok(FieldValue::Map(rt))
            }
            v => Err(SchemaError::FieldValue(format!("expected Map, got {v:?}"))),
        }
    }

    /// Try to create a FieldValue from a CBOR value, inferring the type
    ///
    /// # Arguments
    /// * `value` - The CBOR value to convert
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The converted FieldValue or an error message
    pub fn try_from(value: Cbor) -> Result<Self, SchemaError> {
        match value {
            Cbor::Bool(_) => Self::bool_from(value),
            Cbor::Integer(i) => {
                let z = ciborium::value::Integer::from(0);
                if i >= z {
                    Self::u64_from(value)
                } else {
                    Self::i64_from(value)
                }
            }
            Cbor::Float(_) => Self::f64_from(value),
            Cbor::Bytes(_) => Self::bytes_from(value),
            Cbor::Text(_) => Self::text_from(value),
            Cbor::Array(_) => Self::array_from(value, &[]),
            Cbor::Map(_) => Self::map_from(value, &BTreeMap::new()),
            Cbor::Null => Ok(FieldValue::Null),
            Cbor::Tag(_, val) => Self::try_from(*val),
            v => Err(SchemaError::FieldValue(format!(
                "invalid CBOR value: {v:?}"
            ))),
        }
    }

    /// Create a FieldValue by serializing a value
    ///
    /// # Arguments
    /// * `value` - The value to serialize
    /// * `ft` - Optional field type to use for extraction
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The serialized FieldValue or an error message
    pub fn serialized<T: ?Sized + Serialize>(
        value: &T,
        ft: Option<&FieldType>,
    ) -> Result<Self, SchemaError> {
        let rt = Cbor::serialized(value)
            .map_err(|v| SchemaError::FieldValue(format!("failed to serialize: {v:?}")))?;
        match ft {
            Some(ft) => ft.extract(rt),
            None => FieldValue::try_from(rt),
        }
    }

    /// Deserialize a FieldValue into a value of type T
    ///
    /// # Returns
    /// * `Result<T, SchemaError>` - The deserialized value or an error message
    pub fn deserialized<T: DeserializeOwned>(self) -> Result<T, SchemaError> {
        let val: Cbor = self.into();
        val.deserialized()
            .map_err(|v| SchemaError::FieldValue(format!("Failed to deserialize: {v:?}")))
    }

    /// Get a field value from a map as a reference T
    pub fn get_field_as<'a, T: ?Sized>(&'a self, field: &str) -> Option<&'a T>
    where
        &'a T: TryFrom<&'a FieldValue>,
    {
        if let Fv::Map(m) = self {
            if let Some(v) = m.get(field) {
                return v.try_into().ok();
            }
        }
        None
    }
}

/// Field entry definition for Anda DB
///
/// Contains field name, type, constraints, and index information
/// Field names are renamed in serialization to save storage space
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldEntry {
    /// Field name, must:
    /// - Be unique in document schema
    /// - Not be empty
    /// - Not exceed 64 characters
    /// - Contain only lowercase letters, numbers, and underscores
    #[serde(rename = "n")]
    name: String,

    /// Field description
    /// This can be used by clients to improve the LLM's understanding of available fields
    /// and their expected values.
    #[serde(rename = "d")]
    description: String,

    /// Field type
    #[serde(rename = "t")]
    r#type: FieldType,

    /// Whether the field value must be unique in the collection
    #[serde(rename = "u")]
    unique: bool,

    /// Field index value - field names are not stored with each record,
    /// but are referenced by index to save storage space
    #[serde(rename = "i")]
    idx: usize,
}

impl FieldEntry {
    /// Create a new field entry
    ///
    /// # Arguments
    /// * `name` - Field name
    /// * `r#type` - Field type
    ///
    /// # Returns
    /// * `Result<Self, SchemaError>` - The created field entry or an error message
    pub fn new(name: String, r#type: FieldType) -> Result<Self, SchemaError> {
        validate_field_name(&name)?;
        Ok(Self {
            name,
            r#type,
            description: String::new(),
            unique: false,
            idx: 0,
        })
    }

    /// Set the field description
    ///
    /// # Arguments
    /// * `description` - Field description
    ///
    /// # Returns
    /// * `Self` - The modified field entry
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Mark the field as unique
    ///
    /// # Returns
    /// * `Self` - The modified field entry
    pub fn with_unique(mut self) -> Self {
        self.unique = true;
        self
    }

    /// Set the field index
    ///
    /// # Arguments
    /// * `idx` - Field index value
    ///
    /// # Returns
    /// * `Self` - The modified field entry
    pub fn with_idx(mut self, idx: usize) -> Self {
        self.idx = idx;
        self
    }

    /// Get the field name
    ///
    /// # Returns
    /// * `&str` - The field name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the field type
    ///
    /// # Returns
    /// * `&FieldType` - The field type
    pub fn r#type(&self) -> &FieldType {
        &self.r#type
    }

    pub fn required(&self) -> bool {
        if let FieldType::Option(_) = self.r#type {
            return false;
        }
        true
    }

    /// Check if the field is unique
    ///
    /// # Returns
    /// * `bool` - True if the field is unique
    pub fn unique(&self) -> bool {
        self.unique
    }

    /// Get the field index
    ///
    /// # Returns
    /// * `usize` - The field index
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// Extract a field value from a CBOR value
    ///
    /// # Arguments
    /// * `val` - The CBOR value to extract from
    /// * `validate` - Whether to validate the extracted value
    ///
    /// # Returns
    /// * `Result<FieldValue, SchemaError>` - The extracted field value or an error message
    pub fn extract(&self, val: Cbor, validate: bool) -> Result<FieldValue, SchemaError> {
        match self.r#type.extract(val) {
            Ok(v) => {
                if validate {
                    self.validate(&v)?;
                }
                Ok(v)
            }
            Err(e) => Err(SchemaError::FieldValue(format!(
                "field {} is invalid, error: {}",
                self.name, e
            ))),
        }
    }

    /// Validate a field value against this field entry's constraints
    ///
    /// # Arguments
    /// * `value` - The field value to validate
    ///
    /// # Returns
    /// * `Result<(), SchemaError>` - Ok if valid, or an error message if invalid
    pub fn validate(&self, value: &FieldValue) -> Result<(), SchemaError> {
        if value == &FieldValue::Null {
            if let FieldType::Option(_) = self.r#type {
                return Ok(());
            }

            return Err(SchemaError::FieldValue(format!(
                "field {} is required, expected type {:?}",
                self.name, self.r#type
            )));
        }

        self.r#type.validate(value).map_err(|err| {
            SchemaError::FieldValue(format!("field {} is invalid, error: {}", self.name, err))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ciborium::{cbor, from_reader, into_writer};
    use ic_auth_types::Xid;
    use serde_json::json;

    #[test]
    fn test_field_type_debug() {
        assert_eq!(format!("{:?}", FieldType::Bool), "Bool");
        assert_eq!(format!("{:?}", FieldType::I64), "I64");
        assert_eq!(format!("{:?}", FieldType::U64), "U64");
        assert_eq!(format!("{:?}", FieldType::F64), "F64");
        assert_eq!(format!("{:?}", FieldType::F32), "F32");
        assert_eq!(format!("{:?}", FieldType::Bytes), "Bytes");
        assert_eq!(format!("{:?}", FieldType::Text), "Text");
        assert_eq!(format!("{:?}", FieldType::Json), "Json");
        assert_eq!(format!("{:?}", FieldType::Vector), "Vector");

        let array_type = FieldType::Array(vec![FieldType::U64]);
        assert_eq!(format!("{:?}", array_type), "Array([U64])");

        let mut map = BTreeMap::new();
        map.insert("key".to_string(), FieldType::Text);
        let map_type = FieldType::Map(map);
        assert_eq!(format!("{:?}", map_type), "Map({\"key\": Text})");

        let option_type = FieldType::Option(Box::new(FieldType::Bool));
        assert_eq!(format!("{:?}", option_type), "Option(Bool)");
    }

    #[test]
    fn test_field_value_debug() {
        assert_eq!(format!("{:?}", FieldValue::Bool(true)), "Bool(true)");
        assert_eq!(format!("{:?}", FieldValue::I64(-42)), "I64(-42)");
        assert_eq!(format!("{:?}", FieldValue::U64(42)), "U64(42)");
        assert_eq!(format!("{:?}", FieldValue::F64(3.15)), "F64(3.15)");
        assert_eq!(format!("{:?}", FieldValue::F32(2.71)), "F32(2.71)");
        assert_eq!(
            format!("{:?}", FieldValue::Bytes(vec![1, 2, 3])),
            "Bytes([1, 2, 3])"
        );
        assert_eq!(
            format!("{:?}", FieldValue::Text("hello".to_string())),
            "Text(\"hello\")"
        );

        let json_val = FieldValue::Json(json!({"name": "test"}));
        assert_eq!(
            format!("{:?}", json_val),
            "Json(Object {\"name\": String(\"test\")})"
        );

        assert_eq!(
            format!("{:?}", FieldValue::Vector(vec![bf16::from_f32(1.5)])),
            "Vector([1.5])"
        );

        let array_val = FieldValue::Array(vec![FieldValue::U64(1), FieldValue::U64(2)]);
        assert_eq!(format!("{:?}", array_val), "Array([U64(1), U64(2)])");

        let mut map = BTreeMap::new();
        map.insert("key".to_string(), FieldValue::Text("value".to_string()));
        let map_val = FieldValue::Map(map);
        assert_eq!(format!("{:?}", map_val), "Map({\"key\": Text(\"value\")})");

        assert_eq!(format!("{:?}", FieldValue::Null), "Null");
    }

    #[test]
    fn test_field_type_extract() {
        // Bool
        let bool_val = FieldType::Bool.extract(Cbor::Bool(true)).unwrap();
        assert_eq!(bool_val, FieldValue::Bool(true));

        // U64
        let u64_val = FieldType::U64.extract(cbor!(42).unwrap()).unwrap();
        assert_eq!(u64_val, FieldValue::U64(42));

        // I64
        let i64_val = FieldType::I64.extract(Cbor::Integer((-42).into())).unwrap();
        assert_eq!(i64_val, FieldValue::I64(-42));

        // F64
        let f64_val = FieldType::F64.extract(Cbor::Float(3.15)).unwrap();
        assert_eq!(f64_val, FieldValue::F64(3.15));

        // F32
        let f32_val = FieldType::F32.extract(Cbor::Float(2.71)).unwrap();
        assert_eq!(f32_val, FieldValue::F32(2.71_f32));

        // Bytes
        let bytes_val = FieldType::Bytes
            .extract(Cbor::Bytes(vec![1, 2, 3]))
            .unwrap();
        assert_eq!(bytes_val, FieldValue::Bytes(vec![1, 2, 3]));

        // Text
        let text_val = FieldType::Text
            .extract(Cbor::Text("hello".to_string()))
            .unwrap();
        assert_eq!(text_val, FieldValue::Text("hello".to_string()));

        // Vector
        let vector_val = FieldType::Vector
            .extract(Cbor::Array(vec![
                Cbor::Integer(bf16::from_f32(1.1).to_bits().into()),
                Cbor::Integer(bf16::from_f32(1.2).to_bits().into()),
            ]))
            .unwrap();
        assert_eq!(
            vector_val,
            FieldValue::Vector(vec![bf16::from_f32(1.1), bf16::from_f32(1.2)])
        );

        // Array with single type
        let array_type = FieldType::Array(vec![FieldType::U64]);
        let array_cbor = Cbor::Array(vec![Cbor::Integer(1.into()), Cbor::Integer(2.into())]);
        let array_val = array_type.extract(array_cbor).unwrap();
        assert_eq!(
            array_val,
            FieldValue::Array(vec![FieldValue::U64(1), FieldValue::U64(2)])
        );

        // Array with multiple types
        let array_type = FieldType::Array(vec![FieldType::U64, FieldType::Text]);
        let array_cbor = Cbor::Array(vec![
            Cbor::Integer(1.into()),
            Cbor::Text("hello".to_string()),
        ]);
        let array_val = array_type.extract(array_cbor).unwrap();
        assert_eq!(
            array_val,
            FieldValue::Array(vec![
                FieldValue::U64(1),
                FieldValue::Text("hello".to_string()),
            ])
        );

        // Map
        let mut map_type = BTreeMap::new();
        map_type.insert("id".to_string(), FieldType::U64);
        map_type.insert("name".to_string(), FieldType::Text);
        let map_type = FieldType::Map(map_type);

        let map_cbor = Cbor::Map(vec![
            (Cbor::Text("id".to_string()), Cbor::Integer(1.into())),
            (
                Cbor::Text("name".to_string()),
                Cbor::Text("test".to_string()),
            ),
        ]);

        let map_val = map_type.extract(map_cbor).unwrap();
        let mut expected_map = BTreeMap::new();
        expected_map.insert("id".to_string(), FieldValue::U64(1));
        expected_map.insert("name".to_string(), FieldValue::Text("test".to_string()));
        assert_eq!(map_val, FieldValue::Map(expected_map));

        // Option (Some)
        let option_type = FieldType::Option(Box::new(FieldType::Bool));
        let option_val = option_type.extract(Cbor::Bool(true)).unwrap();
        assert_eq!(option_val, FieldValue::Bool(true));

        // Option (None)
        let option_val = option_type.extract(Cbor::Null).unwrap();
        assert_eq!(option_val, FieldValue::Null);
    }

    #[test]
    fn test_field_type_validate() {
        // Bool
        assert!(FieldType::Bool.validate(&FieldValue::Bool(true)).is_ok());
        assert!(FieldType::Bool.validate(&FieldValue::U64(1)).is_err());

        // U64
        assert!(FieldType::U64.validate(&FieldValue::U64(42)).is_ok());
        assert!(FieldType::U64.validate(&FieldValue::I64(42)).is_err());

        // I64
        assert!(FieldType::I64.validate(&FieldValue::I64(-42)).is_ok());
        assert!(FieldType::I64.validate(&FieldValue::U64(42)).is_err());

        // F64
        assert!(FieldType::F64.validate(&FieldValue::F64(3.15)).is_ok());
        assert!(FieldType::F64.validate(&FieldValue::F32(3.15)).is_err());

        // F32
        assert!(FieldType::F32.validate(&FieldValue::F32(2.71)).is_ok());
        assert!(FieldType::F32.validate(&FieldValue::F64(2.71)).is_err());

        // Bytes
        assert!(
            FieldType::Bytes
                .validate(&FieldValue::Bytes(vec![1, 2, 3]))
                .is_ok()
        );
        assert!(
            FieldType::Bytes
                .validate(&FieldValue::Text("bytes".to_string()))
                .is_err()
        );

        // Text
        assert!(
            FieldType::Text
                .validate(&FieldValue::Text("hello".to_string()))
                .is_ok()
        );
        assert!(
            FieldType::Text
                .validate(&FieldValue::Bytes(vec![104, 101, 108, 108, 111]))
                .is_err()
        );

        // Json
        assert!(
            FieldType::Json
                .validate(&FieldValue::Json(json!({"key": "value"})))
                .is_ok()
        );
        assert!(
            FieldType::Json
                .validate(&FieldValue::Text("json".to_string()))
                .is_err()
        );

        // Vector
        assert!(
            FieldType::Vector
                .validate(&FieldValue::Vector(vec![bf16::from_f32(1.5)]))
                .is_ok()
        );
        assert!(
            FieldType::Vector
                .validate(&FieldValue::Array(vec![FieldValue::U64(1)]))
                .is_ok()
        );
        assert!(
            FieldType::Vector
                .validate(&FieldValue::Array(vec![FieldValue::I64(-1)]))
                .is_err()
        );

        // Array with single type
        let array_type = FieldType::Array(vec![FieldType::U64]);
        let array_val = FieldValue::Array(vec![FieldValue::U64(1), FieldValue::U64(2)]);
        assert!(array_type.validate(&array_val).is_ok());

        let invalid_array_val = FieldValue::Array(vec![
            FieldValue::U64(1),
            FieldValue::Text("invalid".to_string()),
        ]);
        assert!(array_type.validate(&invalid_array_val).is_err());

        // Array with multiple types
        let array_type = FieldType::Array(vec![FieldType::U64, FieldType::Text]);
        let array_val = FieldValue::Array(vec![
            FieldValue::U64(1),
            FieldValue::Text("hello".to_string()),
        ]);
        assert!(array_type.validate(&array_val).is_ok());

        let invalid_array_val = FieldValue::Array(vec![FieldValue::U64(1)]);
        assert!(array_type.validate(&invalid_array_val).is_err());

        // Map
        let mut map_type = BTreeMap::new();
        map_type.insert("id".to_string(), FieldType::U64);
        map_type.insert("name".to_string(), FieldType::Text);
        let map_type = FieldType::Map(map_type);

        let mut map_val = BTreeMap::new();
        map_val.insert("id".to_string(), FieldValue::U64(1));
        map_val.insert("name".to_string(), FieldValue::Text("test".to_string()));
        let map_val = FieldValue::Map(map_val);
        assert!(map_type.validate(&map_val).is_ok());

        let mut invalid_map_val = BTreeMap::new();
        invalid_map_val.insert("id".to_string(), FieldValue::Text("invalid".to_string()));
        invalid_map_val.insert("name".to_string(), FieldValue::Text("test".to_string()));
        let invalid_map_val = FieldValue::Map(invalid_map_val);
        assert!(map_type.validate(&invalid_map_val).is_err());

        // Option (Some)
        let option_type = FieldType::Option(Box::new(FieldType::Bool));
        assert!(option_type.validate(&FieldValue::Bool(true)).is_ok());
        assert!(option_type.validate(&FieldValue::Null).is_ok());
        assert!(option_type.validate(&FieldValue::U64(42)).is_err());
    }

    #[test]
    fn test_field_value_conversion() {
        // Bool
        let bool_val = FieldValue::Bool(true);
        let cbor: Cbor = bool_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), bool_val);

        // U64
        let u64_val = FieldValue::U64(42);
        let cbor: Cbor = u64_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), u64_val);

        // I64
        let i64_val = FieldValue::I64(-42);
        let cbor: Cbor = i64_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), i64_val);

        // F64
        let f64_val = FieldValue::F64(3.15);
        let cbor: Cbor = f64_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), f64_val);

        // F32
        let f32_val = FieldValue::F32(2.71);
        let cbor: Cbor = f32_val.clone().into();
        // 注意：F32转换为CBOR后再转回来会变成F64
        if let FieldValue::F64(f64_val) = FieldValue::try_from(cbor).unwrap() {
            assert!((f64_val - 2.71).abs() < f32::EPSILON as f64);
        } else {
            panic!("Expected F64");
        }

        // Bytes
        let bytes_val = FieldValue::Bytes(vec![1, 2, 3]);
        let cbor: Cbor = bytes_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), bytes_val);

        // Text
        let text_val = FieldValue::Text("hello".to_string());
        let cbor: Cbor = text_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), text_val);

        // Json
        let json_val = FieldValue::Json(json!({"name": "test"}));
        let cbor: Cbor = json_val.into();
        // JSON转换为CBOR后再转回来会变成Map
        let mut expected_map = BTreeMap::new();
        expected_map.insert("name".to_string(), FieldValue::Text("test".to_string()));
        assert_eq!(
            FieldValue::try_from(cbor).unwrap(),
            FieldValue::Map(expected_map)
        );

        // Vector
        let vector_val = FieldValue::Vector(vec![bf16::from_f32(1.5)]);
        let cbor: Cbor = vector_val.clone().into();
        // Vector转换为CBOR后再转回来会变成Array
        let expected_array =
            FieldValue::Array(vec![FieldValue::U64(bf16::from_f32(1.5).to_bits() as u64)]);
        assert_eq!(FieldValue::try_from(cbor).unwrap(), expected_array);

        // Array
        let array_val = FieldValue::Array(vec![FieldValue::U64(1), FieldValue::U64(2)]);
        let cbor: Cbor = array_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), array_val);

        // Map
        let mut map = BTreeMap::new();
        map.insert("key".to_string(), FieldValue::Text("value".to_string()));
        let map_val = FieldValue::Map(map);
        let cbor: Cbor = map_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), map_val);

        // Null
        let null_val = FieldValue::Null;
        let cbor: Cbor = null_val.clone().into();
        assert_eq!(FieldValue::try_from(cbor).unwrap(), null_val);
    }

    #[test]
    fn test_field_entry() {
        // 创建字段
        let field = FieldEntry::new("user_id".to_string(), FieldType::U64)
            .unwrap()
            .with_unique()
            .with_idx(1);

        assert_eq!(field.name(), "user_id");
        assert_eq!(field.r#type(), &FieldType::U64);
        assert!(field.unique());
        assert_eq!(field.idx(), 1);

        // 测试提取值
        let val = field.extract(Cbor::Integer(42.into()), true).unwrap();
        assert_eq!(val, FieldValue::U64(42));

        // 测试验证值
        assert!(field.validate(&FieldValue::U64(42)).is_ok());
        assert!(field.validate(&FieldValue::I64(42)).is_err());

        // 测试必填字段的空值验证
        assert!(field.validate(&FieldValue::Null).is_err());

        // 测试非必填字段的空值验证
        let optional_field = FieldEntry::new(
            "optional".to_string(),
            FieldType::Option(Box::new(FieldType::U64)),
        )
        .unwrap();
        assert!(optional_field.validate(&FieldValue::Null).is_ok());
    }

    #[test]
    fn test_validate_field_name() {
        // 有效的字段名
        assert!(validate_field_name("user_id").is_ok());
        assert!(validate_field_name("a").is_ok());
        assert!(validate_field_name("a1").is_ok());
        assert!(validate_field_name("a_1").is_ok());

        // 无效的字段名
        assert!(validate_field_name("").is_err()); // 空字符串
        assert!(validate_field_name("A").is_err()); // 大写字母
        assert!(validate_field_name("user-id").is_err()); // 包含连字符
        assert!(validate_field_name("user.id").is_err()); // 包含点
        assert!(validate_field_name("user id").is_err()); // 包含空格

        // 超长字段名
        let long_name = "a".repeat(65);
        assert!(validate_field_name(&long_name).is_err());
    }

    #[test]
    fn test_serialization() {
        // 测试 FieldType 序列化和反序列化
        let field_type = Ft::Array(vec![Ft::U64, Ft::Text]);
        let serialized = serde_json::to_string(&field_type).unwrap();
        println!("Serialized FieldType: {}", serialized);
        let deserialized: Ft = serde_json::from_str(&serialized).unwrap();
        assert_eq!(field_type, deserialized);
        let mut serialized = Vec::new();
        into_writer(&field_type, &mut serialized).unwrap();
        println!("Serialized FieldType: {:?}", const_hex::encode(&serialized));
        let deserialized: Ft = from_reader(&serialized[..]).unwrap();
        assert_eq!(field_type, deserialized);

        // 测试 FieldValue 序列化和反序列化
        let field_value = Fv::Array(vec![Fv::U64(1), Fv::Text("hello".to_string())]);
        let mut serialized = Vec::new();
        into_writer(&field_value, &mut serialized).unwrap();
        println!(
            "Serialized FieldValue: {:?}",
            const_hex::encode(&serialized)
        );
        assert_eq!(const_hex::encode(&serialized), "82016568656c6c6f");
        let deserialized: Fv = from_reader(&serialized[..]).unwrap();
        assert_eq!(field_value, deserialized);

        let field_value = Fv::Bytes(vec![1, 2, 3, 4]);
        let mut serialized = Vec::new();
        into_writer(&field_value, &mut serialized).unwrap();
        println!("Serialized bytes: {:?}", const_hex::encode(&serialized));
        assert_eq!(const_hex::encode(&serialized), "4401020304");
        let deserialized: Fv = from_reader(&serialized[..]).unwrap();
        assert_eq!(field_value, deserialized);

        // 测试 FieldEntry 序列化和反序列化
        let field_entry = Fe::new("id".to_string(), Ft::Bytes)
            .unwrap()
            .with_unique()
            .with_idx(0);
        let mut serialized = Vec::new();
        into_writer(&field_entry, &mut serialized).unwrap();
        let deserialized: Fe = from_reader(&serialized[..]).unwrap();
        assert_eq!(field_entry, deserialized);

        let xid = Xid([1u8; 12]);
        let mut data = Vec::new();
        into_writer(&xid, &mut data).unwrap();
        println!("Serialized Xid: {:?}", const_hex::encode(&data));
        assert_eq!(const_hex::encode(&data), "4c010101010101010101010101");
        let cb: Cbor = from_reader(&data[..]).unwrap();
        let fv: FieldValue = FieldValue::try_from(cb).unwrap();
        let deserialized_xid: Xid = fv.deserialized().unwrap();
        assert_eq!(xid, deserialized_xid);

        let vv = vec![
            [bf16::from_f32(1.0), bf16::from_f32(1.1)],
            [bf16::from_f32(2.0), bf16::from_f32(2.1)],
        ];
        // bf16 使用了 u16 存储，未提供 Ft 时将序列化成 u64
        let fv = Fv::serialized(&vv, None).unwrap();
        assert_eq!(
            fv,
            Fv::Array(vec![
                Fv::Array(vec![Fv::U64(16256), Fv::U64(16269)]),
                Fv::Array(vec![Fv::U64(16384), Fv::U64(16390),])
            ])
        );
        // 虽然 Fv 类型不对，但还是可以反序列化成 Vec<[bf16; 2]>
        let vv2: Vec<[bf16; 2]> = fv.deserialized().unwrap();
        assert_eq!(vv, vv2);

        // 提供了 Ft 后才能完全正确的序列化
        let fv = Fv::serialized(&vv, Some(&Ft::Array(vec![Ft::Vector]))).unwrap();
        assert_eq!(
            fv,
            Fv::Array(vec![
                Fv::Vector(vec![bf16::from_f32(1.0), bf16::from_f32(1.1)]),
                Fv::Vector(vec![bf16::from_f32(2.0), bf16::from_f32(2.1),])
            ])
        );
        let vv2: Vec<[bf16; 2]> = fv.deserialized().unwrap();
        assert_eq!(vv, vv2);
    }
}
