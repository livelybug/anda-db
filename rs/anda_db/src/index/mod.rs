use anda_db_schema::{Document, Fv, Json, Vector};
use ic_auth_types::canonical_cbor_into_vec;
use std::borrow::Cow;

mod bm25;
mod btree;
mod hnsw;

pub use bm25::*;
pub use btree::*;
pub use hnsw::*;

pub trait IndexHooks: Send + Sync {
    fn btree_index_value<'a>(&self, index: &BTree, doc: &'a Document) -> Option<Cow<'a, Fv>> {
        let fields = index.virtual_field();
        match fields {
            [] => None,
            [name] => doc.get_field(name).map(Cow::Borrowed),
            _ => {
                let mut vals: Vec<Option<&Fv>> = Vec::with_capacity(fields.len());
                for name in fields {
                    vals.push(doc.get_field(name));
                }

                virtual_field_value(&vals).map(Cow::Owned)
            }
        }
    }

    fn bm25_index_value<'a>(&self, index: &BM25, doc: &'a Document) -> Option<Cow<'a, str>> {
        let fields = index.virtual_field();
        let mut vals: Vec<Option<&Fv>> = Vec::with_capacity(fields.len());
        for name in fields {
            vals.push(doc.get_field(name));
        }

        virtual_searchable_text(&vals)
    }

    fn hnsw_index_value<'a>(&self, index: &Hnsw, doc: &'a Document) -> Option<Cow<'a, Vector>> {
        if let Some(Fv::Vector(vector)) = doc.get_field(index.field_name()) {
            return Some(Cow::Borrowed(vector));
        }
        None
    }
}

pub fn virtual_field_name(fields: &[&str]) -> String {
    fields.join("-")
}

pub fn from_virtual_field_name(name: &str) -> Vec<String> {
    name.split('-').map(String::from).collect()
}

pub fn virtual_field_value(vals: &[Option<&Fv>]) -> Option<Fv> {
    if vals.is_empty() {
        return None;
    }

    let data = canonical_cbor_into_vec(vals).ok()?;
    Some(Fv::Bytes(data))
}

pub fn virtual_searchable_text<'a>(vals: &[Option<&'a Fv>]) -> Option<Cow<'a, str>> {
    let mut texts: Vec<&str> = Vec::new();
    for val in vals.iter().flatten() {
        extract_text(&mut texts, val)
    }

    match texts.len() {
        0 => None,
        1 => Some(Cow::Borrowed(texts[0])),
        _ => Some(Cow::Owned(texts.join("\n"))),
    }
}

fn extract_text<'a>(texts: &mut Vec<&'a str>, val: &'a Fv) {
    match val {
        Fv::Text(text) => texts.push(text),
        Fv::Array(vals) => {
            for val in vals {
                extract_text(texts, val);
            }
        }
        Fv::Map(vals) => {
            for val in vals.values() {
                extract_text(texts, val);
            }
        }
        Fv::Json(json) => extract_json_text(texts, json),
        _ => {}
    }
}

fn extract_json_text<'a>(texts: &mut Vec<&'a str>, val: &'a Json) {
    match val {
        Json::String(s) => texts.push(s),
        Json::Object(obj) => {
            for val in obj.values() {
                extract_json_text(texts, val);
            }
        }
        Json::Array(arr) => {
            if !arr.is_empty() && !matches!(arr[0], Json::String(_) | Json::Object(_)) {
                return;
            }

            for val in arr {
                extract_json_text(texts, val);
            }
        }
        _ => {}
    }
}

pub struct DefaultIndexHooks;

impl IndexHooks for DefaultIndexHooks {}
