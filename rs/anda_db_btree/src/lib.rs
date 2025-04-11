//! # Anda-DB B-tree Index Library
//!
//! This module provides a B-tree based index implementation for Anda-DB.
//! It supports indexing fields of various types including u64, i64, String, and binary data.

mod btree;
mod error;

pub use btree::*;
pub use error::*;
