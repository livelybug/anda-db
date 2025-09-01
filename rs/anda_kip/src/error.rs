//! # Comprehensive error types and handling

use nom_language::error::{VerboseError, VerboseErrorKind, convert_error};
use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum KipError {
    #[error("ParseError: {0}")]
    Parse(String),

    #[error("InvalidCommand: {0}")]
    InvalidCommand(String),

    #[error("ExecutionError: {0}")]
    Execution(String),

    #[error("NotFound: {0}")]
    NotFound(String),

    #[error("AlreadyExists: {0}")]
    AlreadyExists(String),

    #[error("NotImplemented: {0}")]
    NotImplemented(String),
}

impl KipError {
    pub fn parse(err: impl Display) -> Self {
        KipError::Parse(format!("{err}"))
    }

    pub fn invalid_command(err: impl Display) -> Self {
        KipError::InvalidCommand(format!("{err}"))
    }

    pub fn execution(err: impl Display) -> Self {
        KipError::Execution(format!("{err}"))
    }

    pub fn not_found(err: impl Display) -> Self {
        KipError::NotFound(format!("{err}"))
    }

    pub fn already_exists(err: impl Display) -> Self {
        KipError::AlreadyExists(format!("{err}"))
    }

    pub fn not_implemented(err: impl Display) -> Self {
        KipError::NotImplemented(format!("{err}"))
    }
}

pub fn format_nom_error(input: &str, err: nom::Err<VerboseError<&str>>) -> String {
    match err {
        nom::Err::Incomplete(needed) => {
            format!("parse incomplete, need more string: {needed:?}")
        }
        nom::Err::Error(ve) | nom::Err::Failure(ve) => format_verbose_error(input, ve),
    }
}

fn format_verbose_error(input: &str, ve: VerboseError<&str>) -> String {
    let mut msg = String::new();
    if let Some((slice, kind)) = ve.errors.first() {
        let offending = take_first_chars(slice, 1024);
        let kind_str = match kind {
            VerboseErrorKind::Context(ctx) => format!("Context: {ctx}"),
            VerboseErrorKind::Char(c) => format!("Expected char: '{c}'"),
            VerboseErrorKind::Nom(e) => format!("Parser: {:?}", e),
        };

        let fragment = format!("{}\nOffending slice:\n{}", kind_str, offending);
        msg.push_str(&fragment);
        if offending.len() < slice.len() {
            msg.push_str("(truncated...)");
        }
        msg.push_str("\n\n");
    }

    msg.push_str(&convert_error(input, ve));

    msg
}

// 取 slice 的前 n 个字符（避免中途截断多字节字符）
fn take_first_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
