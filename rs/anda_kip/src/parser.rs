use crate::ast::KipCommand;
use nom::{
    IResult,
    branch::alt,
    combinator::{all_consuming, map},
};

// Make sub-modules public within the parser module
mod common;
mod kml;
mod kql;
mod meta;

/// The main entry point for parsing any KIP command.
///
/// It attempts to parse the input as KQL, KML, or META in order.
/// It also ensures the entire input string is consumed.
///
/// # Arguments
///
/// * `input` - The raw KIP command string.
///
/// # Returns
///
/// A `nom::IResult` containing the parsed `KipCommand` or a parsing error.
pub fn parse_kip_command(input: &str) -> IResult<&str, KipCommand> {
    all_consuming(common::ws(alt((
        map(kql::parse_kql_query, KipCommand::Kql),
        map(kml::parse_kml_statement, KipCommand::Kml),
        map(meta::parse_meta_command, KipCommand::Meta),
    ))))(input)
}
