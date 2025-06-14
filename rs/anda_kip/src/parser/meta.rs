use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt, value},
    sequence::preceded,
};

use super::common::*;
use crate::ast::*;

// --- Top Level META Parser ---

pub fn parse_meta_command(input: &str) -> IResult<&str, MetaCommand> {
    alt((
        map(parse_describe_command, MetaCommand::Describe),
        map(parse_search_command, MetaCommand::Search),
    ))
    .parse(input)
}

// --- DESCRIBE ---

fn parse_describe_command(input: &str) -> IResult<&str, DescribeTarget> {
    preceded(
        ws(tag("DESCRIBE ")),
        ws(alt((
            value(DescribeTarget::Primer, tag("PRIMER")),
            value(DescribeTarget::Domains, tag("DOMAINS")),
            value(DescribeTarget::ConceptTypes, tag("CONCEPT TYPES")),
            value(DescribeTarget::PropositionTypes, tag("PROPOSITION TYPES")),
            map(
                preceded(tag("CONCEPT TYPE "), ws(quoted_string)),
                DescribeTarget::ConceptType,
            ),
            map(
                preceded(tag("PROPOSITION TYPE "), ws(quoted_string)),
                DescribeTarget::PropositionType,
            ),
        ))),
    )
    .parse(input)
}

// --- SEARCH ---
fn parse_search_command(input: &str) -> IResult<&str, SearchCommand> {
    map(
        preceded(
            ws(tag("SEARCH ")),
            preceded(
                ws(tag("CONCEPT ")),
                (
                    ws(quoted_string),
                    opt(preceded(tag("WITH TYPE "), ws(quoted_string))),
                    opt(preceded(tag("LIMIT "), ws(nom::character::complete::u64))),
                ),
            ),
        ),
        |(term, in_type, limit)| SearchCommand {
            term,
            in_type,
            limit,
        },
    )
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_meta_command() {
        // Test DESCRIBE commands
        assert_eq!(
            parse_meta_command("DESCRIBE PRIMER"),
            Ok(("", MetaCommand::Describe(DescribeTarget::Primer)))
        );
        assert_eq!(
            parse_meta_command("DESCRIBE DOMAINS"),
            Ok(("", MetaCommand::Describe(DescribeTarget::Domains)))
        );

        // Test SEARCH commands
        assert_eq!(
            parse_meta_command("SEARCH CONCEPT \"aspirin\""),
            Ok((
                "",
                MetaCommand::Search(SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: None,
                    limit: None,
                })
            ))
        );

        // Test with whitespace
        assert_eq!(
            parse_meta_command("  DESCRIBE   PRIMER  "),
            Ok(("", MetaCommand::Describe(DescribeTarget::Primer)))
        );

        // Test invalid command
        assert!(parse_meta_command("INVALID COMMAND").is_err());
    }

    #[test]
    fn test_parse_describe_command() {
        // Test all DESCRIBE targets
        assert_eq!(
            parse_describe_command("DESCRIBE PRIMER"),
            Ok(("", DescribeTarget::Primer))
        );
        assert_eq!(
            parse_describe_command("DESCRIBE DOMAINS"),
            Ok(("", DescribeTarget::Domains))
        );
        assert_eq!(
            parse_describe_command("DESCRIBE CONCEPT TYPES"),
            Ok(("", DescribeTarget::ConceptTypes))
        );
        assert_eq!(
            parse_describe_command("DESCRIBE PROPOSITION TYPES"),
            Ok(("", DescribeTarget::PropositionTypes))
        );
        assert_eq!(
            parse_describe_command("DESCRIBE CONCEPT TYPE \"Drug\""),
            Ok(("", DescribeTarget::ConceptType("Drug".to_string())))
        );
        assert_eq!(
            parse_describe_command("DESCRIBE PROPOSITION TYPE \"treats\""),
            Ok(("", DescribeTarget::PropositionType("treats".to_string())))
        );

        // Test with whitespace
        assert_eq!(
            parse_describe_command("  DESCRIBE   PRIMER  "),
            Ok(("", DescribeTarget::Primer))
        );

        // Test invalid DESCRIBE command
        assert!(parse_describe_command("DESCRIBE INVALID").is_err());
    }

    #[test]
    fn test_parse_search_command() {
        // Basic search
        assert_eq!(
            parse_search_command("SEARCH CONCEPT \"aspirin\""),
            Ok((
                "",
                SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: None,
                    limit: None,
                }
            ))
        );

        // Search with type
        assert_eq!(
            parse_search_command("SEARCH CONCEPT \"aspirin\" \n\n\nWITH TYPE \"Drug\" \nLIMIT  5"),
            Ok((
                "",
                SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: Some("Drug".to_string()),
                    limit: Some(5),
                }
            ))
        );

        // Search with limit
        assert_eq!(
            parse_search_command("SEARCH CONCEPT \"aspirin\" LIMIT 5"),
            Ok((
                "",
                SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: None,
                    limit: Some(5),
                }
            ))
        );

        // Search with type and limit
        assert_eq!(
            parse_search_command("SEARCH CONCEPT \"aspirin\" WITH TYPE \"Drug\" LIMIT 5"),
            Ok((
                "",
                SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: Some("Drug".to_string()),
                    limit: Some(5),
                }
            ))
        );

        // Test with whitespace
        assert_eq!(
            parse_search_command("  SEARCH   CONCEPT   \"aspirin\"  "),
            Ok((
                "",
                SearchCommand {
                    term: "aspirin".to_string(),
                    in_type: None,
                    limit: None,
                }
            ))
        );

        // Test with special characters in search term
        assert_eq!(
            parse_search_command("SEARCH CONCEPT \"阿司匹林\""),
            Ok((
                "",
                SearchCommand {
                    term: "阿司匹林".to_string(),
                    in_type: None,
                    limit: None,
                }
            ))
        );

        // Test invalid search command
        assert!(parse_search_command("SEARCH INVALID").is_err());
    }
}
