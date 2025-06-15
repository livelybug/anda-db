//! # KIP Command Executor Module
//!
//! This module provides the execution framework for Knowledge Interaction Protocol (KIP) commands.
//! It defines the core `Executor` trait that must be implemented by any KIP command processor,
//! and provides a convenient high-level function for executing KIP commands from string input.
//!
//! The executor is responsible for taking parsed KIP commands (KQL queries, KML statements,
//! or META commands) and executing them against a knowledge graph or cognitive nexus,
//! returning structured responses.

use crate::ast::Command;
use crate::error::KipError;
use crate::parser::parse_kip;
use crate::response::Response;

/// The core trait that defines how KIP commands are executed.
///
/// This trait must be implemented by any system that wants to process KIP commands.
/// It provides a single asynchronous method for executing parsed commands and returning
/// structured responses.
///
/// # Design Philosophy
///
/// The `Executor` trait is designed to be:
/// - **Asynchronous**: All operations return futures to support non-blocking I/O
/// - **Generic**: Can be implemented by different backend systems (databases, APIs, etc.)
/// - **Error-safe**: Uses `Result` types for proper error handling
/// - **Send-safe**: Futures are `Send` to support multi-threaded execution
///
/// # Implementation Examples
///
/// ```rust,no_run
/// use anda_kip::Executor;
/// use anda_kip::Command;
/// use anda_kip::KipError;
/// use anda_kip::Response;
///
/// struct MyKnowledgeGraph {
///     // Your knowledge graph implementation
/// }
///
/// impl Executor for MyKnowledgeGraph {
///     async fn execute(&self, command: Command) -> Result<Response, KipError> {
///         match command {
///             Command::Kql(query) => {
///                 // Execute KQL query against knowledge graph
///                 todo!("Implement KQL execution")
///             },
///             Command::Kml(statement) => {
///                 // Execute KML statement to modify knowledge graph
///                 todo!("Implement KML execution")
///             },
///             Command::Meta(meta_cmd) => {
///                 // Execute META command for introspection
///                 todo!("Implement META execution")
///             }
///         }
///     }
/// }
/// ```
pub trait Executor {
    /// Executes a parsed KIP command and returns the result.
    ///
    /// This method takes ownership of a `Command` (which can be a KQL query,
    /// KML statement, or META command) and executes it against the underlying
    /// knowledge system.
    ///
    /// # Arguments
    ///
    /// * `command` - The parsed KIP command to execute
    ///
    /// # Returns
    ///
    /// A `Future` that resolves to:
    /// - `Ok(Response)`: Successful execution with structured response data
    /// - `Err(KipError)`: Execution error with detailed error information
    ///
    /// # Error Handling
    ///
    /// Implementations should return appropriate `KipError` variants for different
    /// failure scenarios:
    /// - `KipError::Execution`: For runtime execution errors
    /// - `KipError::Validation`: For semantic validation failures
    /// - `KipError::NotFound`: For queries that find no matching data
    /// - `KipError::PermissionDenied`: For authorization failures
    ///
    /// # Performance Considerations
    ///
    /// - This method is async to support non-blocking I/O operations
    /// - The returned future is `Send` to enable multi-threaded execution
    /// - Implementations should consider query optimization and caching
    fn execute(&self, command: Command) -> impl Future<Output = Result<Response, KipError>> + Send;
}

/// High-level convenience function for executing KIP commands from string input.
///
/// This function provides a complete pipeline from raw KIP command string to execution result.
/// It handles parsing the input string into a structured command and then delegates execution
/// to the provided executor implementation.
///
/// # Workflow
///
/// 1. **Parse**: Convert the input string into a structured `Command` AST
/// 2. **Execute**: Pass the parsed command to the executor for processing
/// 3. **Return**: Provide the structured response or detailed error information
///
/// # Arguments
///
/// * `executor` - An implementation of the `Executor` trait that will process the command
/// * `command` - The raw KIP command string to parse and execute
///
/// # Returns
///
/// A `Result` containing:
/// - `Ok(Response)`: Successful execution with structured response data
/// - `Err(KipError)`: Either parsing or execution error with detailed information
///
/// # Error Types
///
/// This function can return errors from two sources:
/// - **Parse Errors**: `KipError::Parse` when the input string is malformed
/// - **Execution Errors**: Any `KipError` variant returned by the executor
///
/// # Examples
///
/// ```rust,no_run
/// use anda_kip::{execute_kip, Executor};
///
/// async fn example(my_executor: impl Executor) {
///     // Execute a KQL query
///     let kql_result = execute_kip(
///         &my_executor,
///         "FIND(?drug) WHERE { ?drug(type: \"Drug\") }"
///     ).await;
///
///     // Execute a KML statement
///     let kml_result = execute_kip(
///         &my_executor,
///         "UPSERT { CONCEPT @drug { ON { name: \"Aspirin\" } } }"
///     ).await;
///
///     // Execute a META command
///     let meta_result = execute_kip(
///         &my_executor,
///         "DESCRIBE PRIMER"
///     ).await;
///
///     match kql_result {
///         Ok(response) => println!("Query successful: {:?}", response),
///         Err(error) => eprintln!("Query failed: {:?}", error),
///     }
/// }
/// ```
///
/// # Performance Notes
///
/// - Parsing is performed synchronously before execution
/// - Consider caching parsed commands for repeated execution
/// - The executor implementation determines overall performance characteristics
pub async fn execute_kip(executor: &impl Executor, command: &str) -> Result<Response, KipError> {
    // Parse the raw command string into a structured Command AST
    let cmd = parse_kip(command)?;

    // Delegate execution to the provided executor implementation
    executor.execute(cmd).await
}
