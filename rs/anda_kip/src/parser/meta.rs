use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag_no_case,
    combinator::{map, opt},
    sequence::{preceded, tuple},
};

use super::common::*;
use crate::ast::*;

// --- Top Level META Parser ---

pub fn parse_meta_command(input: &str) -> IResult<&str, MetaCommand> {
    alt((
        map(parse_describe_command, MetaCommand::Describe),
        map(parse_search_command, MetaCommand::Search),
    ))(input)
}

// --- DESCRIBE ---

fn parse_describe_command(input: &str) -> IResult<&str, DescribeTarget> {
    preceded(
        ws(tag_no_case("DESCRIBE")),
        alt((
            map(tag_no_case("PRIMER"), |_| DescribeTarget::Primer),
            map(tag_no_case("DOMAINS"), |_| DescribeTarget::Domains),
            map(tag_no_case("CONCEPT TYPES"), |_| {
                DescribeTarget::ConceptTypes
            }),
            map(
                preceded(ws(tag_no_case("CONCEPT TYPE")), quoted_string),
                DescribeTarget::ConceptType,
            ),
            map(tag_no_case("PROPOSITION TYPES"), |_| {
                DescribeTarget::PropositionTypes
            }),
            map(
                preceded(ws(tag_no_case("PROPOSITION TYPE")), quoted_string),
                DescribeTarget::PropositionType,
            ),
        )),
    )(input)
}

// --- SEARCH ---

fn parse_search_command(input: &str) -> IResult<&str, SearchCommand> {
    map(
        preceded(
            ws(tag_no_case("SEARCH CONCEPT")),
            tuple((
                ws(quoted_string),
                opt(ws(preceded(tag_no_case("WITH TYPE"), ws(quoted_string)))),
                opt(ws(preceded(
                    tag_no_case("LIMIT"),
                    nom::character::complete::u64,
                ))),
            )),
        ),
        |(term, in_type, limit)| SearchCommand {
            term,
            in_type,
            limit,
        },
    )(input)
}
