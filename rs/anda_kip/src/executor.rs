use crate::ast::Command;
use crate::error::KipError;
use crate::parser::parse_kip_command;
use crate::response::Response;

pub trait Executor {
    fn execute(&self, command: Command) -> impl Future<Output = Result<Response, KipError>> + Send;
}

pub async fn execute_kip(executor: &impl Executor, command: &str) -> Result<Response, KipError> {
    let (_, cmd) = parse_kip_command(command)
        .map_err(|e| KipError::Parse(format!("Failed to parse command: {}", e)))?;

    executor.execute(cmd).await
}
