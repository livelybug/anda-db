use thiserror::Error;

#[derive(Error, Debug)]
pub enum KipError {
    #[error("Parse Error: {0}")]
    Parse(String),
    #[error("Execution Error: {0}")]
    Execution(String),
    #[error("Not Implemented: {0}")]
    NotImplemented(String),
    #[error("Invalid Command: {0}")]
    InvalidCommand(String),
}
