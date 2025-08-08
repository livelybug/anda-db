use half::f16;
use serde::{
    Serialize,
    ser::{
        self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
        SerializeTupleStruct, SerializeTupleVariant,
    },
};
use std::fmt;

/// 估算任意 `Serialize` 值经 CBOR 序列化后的字节大小（不实际写入字节）。
pub fn estimate_cbor_size<T: ?Sized + Serialize>(value: &T) -> usize {
    let mut s = CborSizer { count: 0 };
    // 忽略错误：本实现不会产生序列化错误，返回 Ok(())
    let _ = value.serialize(&mut s);
    s.count
}

// ---- CBOR sizer 实现：仅依据 CBOR 头部规则与结构遍历累加大小 ----
#[derive(Debug, Clone, Copy)]
struct CborSizeError;

impl fmt::Display for CborSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cbor size estimator error")
    }
}
impl std::error::Error for CborSizeError {}
impl ser::Error for CborSizeError {
    fn custom<T: fmt::Display>(_msg: T) -> Self {
        CborSizeError
    }
}

struct CborSizer {
    count: usize,
}

impl CborSizer {
    #[inline]
    fn add_head_len(&mut self, major: u8, len: u64) {
        // CBOR 头部：1字节(主类型+附加信息) + 可能的长度扩展
        // <24: 1; <= u8: 2; <= u16: 3; <= u32: 5; 否则: 9
        self.count += match len {
            0..=23 => 1,
            24..=0xFF => 2,
            0x100..=0xFFFF => 3,
            0x1_0000..=0xFFFF_FFFF => 5,
            _ => 9,
        };
        let _ = major; // 仅用于语义说明（主类型在首字节中，但不改变字节数）
    }

    #[inline]
    fn add_uint(&mut self, v: u64) {
        self.add_head_len(0, v);
    }

    #[inline]
    fn add_nint_i64(&mut self, v: i64) {
        // 负整数编码：-1 - n 作为无符号整数长度
        let u = -1i128 - v as i128;
        let u = if u < 0 { 0 } else { u as u64 };
        self.add_head_len(1, u);
    }

    #[inline]
    fn add_tag_small(&mut self, tag: u64) {
        // Tag 主类型 6，tag 值一般很小（如 2/3，用于 bignum）
        self.add_head_len(6, tag);
    }

    #[inline]
    fn add_bytes(&mut self, len: usize) {
        self.add_head_len(2, len as u64);
        self.count += len;
    }

    #[inline]
    fn add_text(&mut self, len: usize) {
        self.add_head_len(3, len as u64);
        self.count += len;
    }

    #[inline]
    fn add_array_header(&mut self, len: Option<usize>) -> bool {
        match len {
            Some(n) => {
                self.add_head_len(4, n as u64);
                false
            }
            None => {
                // 不定长数组起始 0x9f
                self.count += 1;
                true
            }
        }
    }

    #[inline]
    fn end_indefinite(&mut self, indefinite: bool) {
        if indefinite {
            // break 0xff
            self.count += 1;
        }
    }

    #[inline]
    fn add_map_header(&mut self, len: Option<usize>) -> bool {
        match len {
            Some(n) => {
                self.add_head_len(5, n as u64);
                false
            }
            None => {
                // 不定长 map 起始 0xbf
                self.count += 1;
                true
            }
        }
    }

    #[inline]
    fn add_f16(&mut self) {
        self.count += 1 /* 头 */ + 2;
    }

    #[inline]
    fn add_f32(&mut self) {
        self.count += 1 /* 头 */ + 4;
    }

    #[inline]
    fn add_f64(&mut self) {
        self.count += 1 /* 头 */ + 8;
    }

    #[inline]
    fn add_simple1(&mut self) {
        // 单字节简单值（false/true/null/undefined）：各占 1 字节
        self.count += 1;
    }

    #[inline]
    fn add_u128(&mut self, v: u128) {
        if v <= u64::MAX as u128 {
            self.add_uint(v as u64);
            return;
        }
        // 超过 u64 范围，使用 bignum(tag: 2) + bytes
        self.add_tag_small(2);
        let nbytes = (128 - v.leading_zeros()).div_ceil(8) as usize;
        self.add_bytes(nbytes);
    }

    #[inline]
    fn add_i128(&mut self, v: i128) {
        if v >= i64::MIN as i128 && v <= i64::MAX as i128 {
            if v >= 0 {
                self.add_uint(v as u64);
            } else {
                self.add_nint_i64(v as i64);
            }
            return;
        }
        // 负大整数使用 tag 3；按 CBOR 规则编码 abs(-1 - v) 的字节串
        if v >= 0 {
            self.add_u128(v as u128);
        } else {
            self.add_tag_small(3);
            let mag = (-1i128 - v) as u128;
            let nbytes = (128 - mag.leading_zeros()).div_ceil(8) as usize;
            self.add_bytes(nbytes);
        }
    }
}

impl<'a> ser::Serializer for &'a mut CborSizer {
    type Ok = ();
    type Error = CborSizeError;

    type SerializeSeq = SeqSizer<'a>;
    type SerializeTuple = SeqSizer<'a>;
    type SerializeTupleStruct = SeqSizer<'a>;
    type SerializeTupleVariant = TupleVariantSizer<'a>;
    type SerializeMap = MapSizer<'a>;
    type SerializeStruct = StructSizer<'a>;
    type SerializeStructVariant = StructVariantSizer<'a>;

    #[inline]
    fn serialize_bool(self, _v: bool) -> Result<Self::Ok, Self::Error> {
        self.add_simple1();
        Ok(())
    }
    #[inline]
    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        if v >= 0 {
            self.add_uint(v as u64);
        } else {
            self.add_nint_i64(v as i64);
        }
        Ok(())
    }
    #[inline]
    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        if v >= 0 {
            self.add_uint(v as u64);
        } else {
            self.add_nint_i64(v as i64);
        }
        Ok(())
    }
    #[inline]
    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        if v >= 0 {
            self.add_uint(v as u64);
        } else {
            self.add_nint_i64(v as i64);
        }
        Ok(())
    }
    #[inline]
    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        if v >= 0 {
            self.add_uint(v as u64);
        } else {
            self.add_nint_i64(v);
        }
        Ok(())
    }
    #[inline]
    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        self.add_i128(v);
        Ok(())
    }

    #[inline]
    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.add_uint(v as u64);
        Ok(())
    }
    #[inline]
    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.add_uint(v as u64);
        Ok(())
    }
    #[inline]
    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.add_uint(v as u64);
        Ok(())
    }
    #[inline]
    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        self.add_uint(v);
        Ok(())
    }
    #[inline]
    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        self.add_u128(v);
        Ok(())
    }

    #[inline]
    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        self.serialize_f64(v.into())
    }

    #[inline]
    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        let n16 = f16::from_f64(v);
        let n32 = v as f32;
        let vbits = v.to_bits();
        if f64::from(n16).to_bits() == vbits {
            self.add_f16();
        } else if f64::from(n32).to_bits() == vbits {
            self.add_f32();
        } else {
            self.add_f64();
        };
        Ok(())
    }

    #[inline]
    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        self.serialize_str(&v.to_string())
    }

    #[inline]
    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.add_text(v.len());
        Ok(())
    }

    #[inline]
    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.add_bytes(v.len());
        Ok(())
    }

    #[inline]
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        // serde_cbor/ciborium 通常将 None 编码为 null
        self.add_simple1();
        Ok(())
    }
    #[inline]
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        // Some(x) 直接编码为 x
        value.serialize(self)
    }

    #[inline]
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        // unit -> null
        self.add_simple1();
        Ok(())
    }

    #[inline]
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.serialize_str(variant)
    }

    #[inline]
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        // 透明包装
        value.serialize(self)
    }

    #[inline]
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        // { "Variant": value }
        self.add_map_header(Some(1));
        self.add_text(variant.len());
        value.serialize(self)
    }

    #[inline]
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        let indefinite = self.add_array_header(len);
        Ok(SeqSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        let indefinite = self.add_array_header(Some(len));
        Ok(SeqSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        let indefinite = self.add_array_header(Some(len));
        Ok(SeqSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        // { "Variant": [ ... ] }
        self.add_map_header(Some(1));
        self.add_text(variant.len());
        let indefinite = self.add_array_header(Some(len));
        Ok(TupleVariantSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        let indefinite = self.add_map_header(len);
        Ok(MapSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        let indefinite = self.add_map_header(Some(len));
        Ok(StructSizer {
            s: self,
            indefinite,
        })
    }

    #[inline]
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        // { "Variant": { k:v, ... } }
        self.add_map_header(Some(1));
        self.add_text(variant.len());
        let indefinite = self.add_map_header(Some(len));
        Ok(StructVariantSizer {
            s: self,
            indefinite,
        })
    }
}

struct SeqSizer<'a> {
    s: &'a mut CborSizer,
    indefinite: bool,
}
impl SerializeSeq for SeqSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.s)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        self.s.end_indefinite(self.indefinite);
        Ok(())
    }
}
impl SerializeTuple for SeqSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::serialize_element(self, value)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::end(self)
    }
}
impl SerializeTupleStruct for SeqSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::serialize_element(self, value)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        <Self as SerializeSeq>::end(self)
    }
}

struct TupleVariantSizer<'a> {
    s: &'a mut CborSizer,
    indefinite: bool,
}
impl SerializeTupleVariant for TupleVariantSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.s)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        self.s.end_indefinite(self.indefinite);
        Ok(())
    }
}

struct MapSizer<'a> {
    s: &'a mut CborSizer,
    indefinite: bool,
}
impl SerializeMap for MapSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Self::Error> {
        key.serialize(&mut *self.s)
    }
    #[inline]
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        value.serialize(&mut *self.s)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        self.s.end_indefinite(self.indefinite);
        Ok(())
    }
}

struct StructSizer<'a> {
    s: &'a mut CborSizer,
    indefinite: bool,
}
impl SerializeStruct for StructSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.s.add_text(key.len());
        value.serialize(&mut *self.s)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        self.s.end_indefinite(self.indefinite);
        Ok(())
    }
}

struct StructVariantSizer<'a> {
    s: &'a mut CborSizer,
    indefinite: bool,
}
impl SerializeStructVariant for StructVariantSizer<'_> {
    type Ok = ();
    type Error = CborSizeError;
    #[inline]
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.s.add_text(key.len());
        value.serialize(&mut *self.s)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        self.s.end_indefinite(self.indefinite);
        Ok(())
    }
}

// ...existing code...
#[cfg(test)]
mod tests {
    use super::*;
    use ciborium::into_writer;
    use serde::Serialize;
    use std::collections::BTreeMap;

    fn measured_size<T: ?Sized + Serialize>(v: &T) -> usize {
        let mut buf = Vec::new();
        into_writer(v, &mut buf).expect("serialize with ciborium");
        buf.len()
    }

    fn assert_estimate_eq<T: ?Sized + Serialize>(label: &str, v: &T) {
        let est = estimate_cbor_size(v);
        let real = measured_size(v);
        assert_eq!(
            est, real,
            "CBOR size mismatch for {label}: est={est}, real={real}"
        );
    }

    #[derive(Debug, Serialize)]
    struct S {
        a: u8,
        b: String,
    }

    #[derive(Debug, Serialize)]
    struct N(u64);

    #[derive(Debug, Serialize)]
    enum E {
        A,
        B(u32),
        C { x: u8 },
    }

    #[derive(Debug, Serialize)]
    enum NE {
        V(u64),
    }

    #[test]
    fn test_cbor_size_primitives() {
        // bool
        assert_estimate_eq("bool:true", &true);
        assert_estimate_eq("bool:false", &false);

        // u64 边界
        for &v in &[
            0u64,
            23,
            24,
            255,
            256,
            65_535,
            65_536,
            u32::MAX as u64,
            u64::MAX,
        ] {
            assert_estimate_eq(&format!("u64:{v}"), &v);
        }

        // i64 边界（包括负数附加信息边界）
        for &v in &[
            -1i64,
            -24,
            -25,
            -255,
            -256,
            -257,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ] {
            assert_estimate_eq(&format!("i64:{v}"), &v);
        }

        // f32/f64
        assert_estimate_eq("f32:1.0", &1.0f32);
        assert_estimate_eq("f64:1.0", &1.0f64);

        // char（ASCII、3字节、4字节）
        assert_estimate_eq("char:a", &'a');
        assert_estimate_eq("char:中", &'中');
        assert_estimate_eq("char:🦀", &'🦀');
    }

    #[test]
    fn test_cbor_size_text_and_bytes() {
        // 字符串长度边界：0, 23, 24, 255, 256
        let lens = [0usize, 1, 23, 24, 255, 256, 1024];
        for &len in &lens {
            let s = "a".repeat(len);
            assert_estimate_eq(&format!("str:len={len}"), &s);
        }

        // bytes 长度边界：0, 23, 24, 255, 256
        for &len in &lens {
            let v = vec![0xABu8; len];
            assert_estimate_eq(&format!("bytes:len={len}"), &v.as_slice());
        }
    }

    #[test]
    fn test_cbor_size_collections() {
        // Option
        let none_val: Option<u64> = None;
        let some_val: Option<u64> = Some(42);
        assert_estimate_eq("option:none", &none_val);
        assert_estimate_eq("option:some", &some_val);

        // Vec/Seq
        let v: Vec<u64> = (0..30).collect();
        assert_estimate_eq("vec<u64>:30", &v);

        // Tuple
        let t = (1u8, "hi".to_string(), 3u64);
        assert_estimate_eq("tuple(u8,String,u64)", &t);

        // Map（使用 BTreeMap 以固定顺序）
        let mut m: BTreeMap<String, u64> = BTreeMap::new();
        m.insert("a".into(), 1);
        m.insert("b".into(), 2);
        m.insert("long_key".into(), 3);
        assert_estimate_eq("btreemap<string,u64>", &m);
    }

    #[test]
    fn test_cbor_size_structs_enums() {
        // 结构体
        let s = S {
            a: 7,
            b: "hello".into(),
        };
        assert_estimate_eq("struct S", &s);

        // newtype struct
        let n = N(123456789);
        assert_estimate_eq("newtype struct N(u64)", &n);

        // 枚举：unit variant / tuple variant / struct variant
        let e1 = E::A;
        let e2 = E::B(123);
        let e3 = E::C { x: 9 };
        assert_estimate_eq("enum E::A", &e1);
        assert_estimate_eq("enum E::B(123)", &e2);
        assert_estimate_eq("enum E::C{x}", &e3);

        // newtype variant
        let ne = NE::V(888);
        assert_estimate_eq("enum NE::V(u64)", &ne);
    }

    #[test]
    fn test_cbor_size_bignum() {
        // 大整数（超出 u64）
        let big_u: u128 = (u64::MAX as u128) + 1;
        let bigger_u: u128 = 1u128 << 127;
        assert_estimate_eq("u128:u64::MAX+1", &big_u);
        assert_estimate_eq("u128:1<<127", &bigger_u);

        // 大负整数（i128 使用 tag 3）
        let big_neg: i128 = -(1i128 << 100);
        let near_min: i128 = i128::MIN + 1; // 仍远小于 i64::MIN
        assert_estimate_eq("i128:-1<<100", &big_neg);
        assert_estimate_eq("i128:near_min", &near_min);
    }
}
