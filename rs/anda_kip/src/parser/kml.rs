use nom::{
    IResult,
    branch::alt,
    bytes::complete::tag_no_case,
    character::complete::char,
    combinator::{map, opt},
    multi::{many1, separated_list1},
    sequence::{preceded, separated_pair, terminated, tuple},
};
use std::collections::HashMap;

use super::common::*;
use super::kql::parse_where_clause_list;
use crate::ast::*;
use crate::nexus::KipValue;

// --- Top Level KML Parser ---

pub fn parse_kml_statement(input: &str) -> IResult<&str, KmlStatement> {
    alt((
        map(parse_upsert_block, KmlStatement::Upsert),
        map(parse_delete_statement, KmlStatement::Delete),
    ))(input)
}

// --- UPSERT ---

fn parse_with_metadata(input: &str) -> IResult<&str, HashMap<String, KipValue>> {
    preceded(ws(tag_no_case("WITH METADATA")), key_value_map)(input)
}

fn parse_upsert_block(input: &str) -> IResult<&str, UpsertBlock> {
    map(
        preceded(
            ws(tag_no_case("UPSERT")),
            tuple((
                braced_block(many1(ws(parse_upsert_item))),
                opt(parse_with_metadata),
            )),
        ),
        |(items, metadata)| UpsertBlock { items, metadata },
    )(input)
}

fn parse_upsert_item(input: &str) -> IResult<&str, UpsertItem> {
    alt((
        map(parse_concept_block, UpsertItem::Concept),
        map(parse_proposition_block, UpsertItem::Proposition),
    ))(input)
}

fn on_clause(input: &str) -> IResult<&str, OnClause> {
    map(
        preceded(
            ws(tag_no_case("ON")),
            braced_block(separated_list1(ws(char(',')), key_value_pair)),
        ),
        |keys| OnClause { keys },
    )(input)
}

fn parse_concept_block(input: &str) -> IResult<&str, ConceptBlock> {
    map(
        tuple((
            preceded(ws(tag_no_case("CONCEPT")), ws(local_handle)),
            braced_block(tuple((
                ws(on_clause),
                opt(ws(preceded(tag_no_case("SET ATTRIBUTES"), key_value_map))),
                opt(ws(preceded(
                    tag_no_case("SET PROPOSITIONS"),
                    braced_block(many1(ws(parse_set_proposition))),
                ))),
                opt(ws(parse_with_metadata)),
            ))),
        )),
        |(handle, (on, set_attributes, set_propositions, metadata))| ConceptBlock {
            handle,
            on,
            set_attributes,
            set_propositions,
            metadata,
        },
    )(input)
}

fn parse_set_proposition(input: &str) -> IResult<&str, SetProposition> {
    map(
        tuple((
            preceded(
                ws(tag_no_case("PROP")),
                parenthesized_block(separated_pair(
                    quoted_string,
                    ws(char(',')),
                    alt((
                        map(local_handle, PropObject::LocalHandle),
                        map(on_clause, PropObject::Node),
                    )),
                )),
            ),
            opt(parse_with_metadata),
        )),
        |((predicate, object), metadata)| SetProposition {
            predicate,
            object,
            metadata,
        },
    )(input)
}

fn parse_proposition_block(input: &str) -> IResult<&str, PropositionBlock> {
    // This parser is a bit complex due to the nested structure
    // PROPOSITION @handle { ( ON {}, "pred", ON {}/@handle ) } WITH METADATA {}
    map(
        tuple((
            preceded(ws(tag_no_case("PROPOSITION")), ws(local_handle)),
            braced_block(tuple((
                parenthesized_block(tuple((
                    ws(on_clause),
                    ws(char(',')),
                    ws(quoted_string),
                    ws(char(',')),
                    ws(alt((
                        map(on_clause, PropObject::Node),
                        map(local_handle, PropObject::LocalHandle),
                    ))),
                ))),
                opt(ws(parse_with_metadata)),
            ))),
        )),
        |(handle, ((subject, _, predicate, _, object), metadata))| PropositionBlock {
            handle,
            subject,
            predicate,
            object,
            metadata,
        },
    )(input)
}

// --- DELETE ---

fn parse_delete_statement(input: &str) -> IResult<&str, DeleteStatement> {
    preceded(
        ws(tag_no_case("DELETE")),
        alt((
            parse_delete_attributes,
            parse_delete_proposition,
            parse_delete_propositions_where,
            parse_delete_concept,
        )),
    )(input)
}

fn parse_delete_attributes(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag_no_case("ATTRIBUTES")),
            tuple((
                braced_block(separated_list1(ws(char(',')), quoted_string)),
                preceded(ws(tag_no_case("FROM")), ws(on_clause)),
            )),
        ),
        |(attributes, from)| DeleteStatement::DeleteAttributes { attributes, from },
    )(input)
}

fn parse_delete_proposition(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag_no_case("PROPOSITION")),
            parenthesized_block(tuple((
                ws(on_clause),
                ws(char(',')),
                ws(quoted_string),
                ws(char(',')),
                ws(on_clause),
            ))),
        ),
        |(subject, _, predicate, _, object)| DeleteStatement::DeleteProposition {
            subject,
            predicate,
            object,
        },
    )(input)
}

fn parse_delete_propositions_where(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag_no_case("PROPOSITIONS")),
            preceded(
                ws(tag_no_case("WHERE")),
                braced_block(parse_where_clause_list),
            ),
        ),
        |where_clauses| DeleteStatement::DeletePropositionsWhere { where_clauses },
    )(input)
}

fn parse_delete_concept(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag_no_case("CONCEPT")),
            terminated(ws(on_clause), ws(tag_no_case("DETACH"))),
        ),
        |on| DeleteStatement::DeleteConcept { on },
    )(input)
}
