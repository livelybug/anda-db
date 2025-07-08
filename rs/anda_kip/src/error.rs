//! # Comprehensive error types and handling

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
