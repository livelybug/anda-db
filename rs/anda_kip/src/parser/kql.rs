use nom::{
    IResult,
    branch::alt,
    bytes::complete::{is_not, tag_no_case},
    character::complete::{char, multispace1},
    combinator::{map, opt},
    multi::{many1, separated_list1},
    sequence::{pair, preceded, terminated, tuple},
};

use super::common::*;
use crate::ast::*;

// --- Top Level KQL Parser ---

pub fn parse_kql_query(input: &str) -> IResult<&str, KqlQuery> {
    map(
        tuple((
            ws(parse_find_clause),
            ws(parse_where_block),
            opt(ws(parse_order_by_clause)),
            opt(ws(parse_limit_clause)),
            opt(ws(parse_offset_clause)),
        )),
        |(find_clause, where_clauses, order_by, limit, offset)| KqlQuery {
            find_clause,
            where_clauses,
            order_by,
            limit,
            offset,
        },
    )(input)
}

// --- FIND Clause ---

fn parse_find_clause(input: &str) -> IResult<&str, FindClause> {
    map(
        preceded(
            ws(tag_no_case("FIND")),
            parenthesized_block(separated_list1(ws(char(',')), parse_find_expression)),
        ),
        |expressions| FindClause { expressions },
    )(input)
}

fn parse_find_expression(input: &str) -> IResult<&str, FindExpression> {
    alt((parse_aggregation_expression, parse_find_variable))(input)
}

fn parse_find_variable(input: &str) -> IResult<&str, FindExpression> {
    map(variable, FindExpression::Variable)(input)
}

fn parse_aggregation_expression(input: &str) -> IResult<&str, FindExpression> {
    map(
        tuple((
            parse_aggregation_function,
            parenthesized_block(tuple((
                opt(terminated(tag_no_case("DISTINCT"), multispace1)),
                variable,
            ))),
            preceded(ws(tag_no_case("AS")), variable),
        )),
        |(func, (distinct, var), alias)| FindExpression::Aggregation {
            func,
            var,
            distinct: distinct.is_some(),
            alias,
        },
    )(input)
}

fn parse_aggregation_function(input: &str) -> IResult<&str, AggregationFunction> {
    alt((
        map(tag_no_case("COUNT"), |_| AggregationFunction::Count),
        map(tag_no_case("COLLECT"), |_| AggregationFunction::Collect),
        map(tag_no_case("SUM"), |_| AggregationFunction::Sum),
        map(tag_no_case("AVG"), |_| AggregationFunction::Avg),
        map(tag_no_case("MIN"), |_| AggregationFunction::Min),
        map(tag_no_case("MAX"), |_| AggregationFunction::Max),
    ))(input)
}

// --- WHERE Clause ---

fn parse_where_block(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag_no_case("WHERE")),
        braced_block(parse_where_clause_list),
    )(input)
}

pub fn parse_where_clause_list(input: &str) -> IResult<&str, Vec<WhereClause>> {
    // This handles UNION by separating groups of clauses
    map(
        separated_list1(ws(tag_no_case("UNION")), parse_basic_where_group),
        |groups| {
            groups
                .into_iter()
                .reduce(|left, right| vec![WhereClause::Union { left, right }])
                .unwrap_or_default()
        },
    )(input)
}

fn parse_basic_where_group(input: &str) -> IResult<&str, Vec<WhereClause>> {
    braced_block(many1(ws(parse_single_where_clause)))(input)
}

fn parse_single_where_clause(input: &str) -> IResult<&str, WhereClause> {
    alt((
        map(parse_optional_clause, WhereClause::Optional),
        map(parse_not_clause, WhereClause::Not),
        map(parse_prop_pattern, WhereClause::Proposition),
        map(parse_grounding_clause, WhereClause::Grounding),
        map(parse_attr_clause, WhereClause::Attribute),
        map(parse_filter_clause, WhereClause::Filter),
    ))(input)
}

// --- WHERE Clause Sub-parsers ---

fn parse_grounding_clause(input: &str) -> IResult<&str, Grounding> {
    map(
        tuple((
            variable,
            parenthesized_block(separated_list1(ws(char(',')), key_value_pair)),
        )),
        |(variable, constraints)| Grounding {
            variable,
            constraints,
        },
    )(input)
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

fn parse_prop_term(input: &str) -> IResult<&str, PropTerm> {
    alt((
        map(variable, PropTerm::Variable),
        map(on_clause, PropTerm::Node),
        map(parenthesized_block(parse_prop_pattern), |p| {
            PropTerm::NestedProp(Box::new(p))
        }),
    ))(input)
}

fn parse_prop_pattern(input: &str) -> IResult<&str, PropositionPattern> {
    map(
        tuple((
            preceded(
                ws(tag_no_case("PROP")),
                parenthesized_block(tuple((
                    ws(parse_prop_term),
                    ws(char(',')),
                    ws(map(quoted_string, |s| s.to_string())), // Simplified predicate
                    ws(char(',')),
                    ws(parse_prop_term),
                ))),
            ),
            opt(braced_block(separated_list1(ws(char(',')), key_value_pair))),
        )),
        |((subject, _, predicate, _, object), metadata_constraints)| PropositionPattern {
            subject,
            predicate,
            object,
            metadata_constraints,
        },
    )(input)
}

fn parse_attr_clause(input: &str) -> IResult<&str, AttributePattern> {
    map(
        preceded(
            ws(tag_no_case("ATTR")),
            parenthesized_block(tuple((
                ws(variable),
                ws(char(',')),
                ws(quoted_string),
                ws(char(',')),
                ws(variable),
            ))),
        ),
        |(node_variable, _, attribute_name, _, value_variable)| AttributePattern {
            node_variable,
            attribute_name,
            value_variable,
        },
    )(input)
}

fn parse_filter_clause(input: &str) -> IResult<&str, FilterCondition> {
    map(
        preceded(ws(tag_no_case("FILTER")), parenthesized_block(is_not(")"))),
        |expr| FilterCondition {
            expression: expr.trim().to_string(),
        },
    )(input)
}

fn parse_optional_clause(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag_no_case("OPTIONAL")),
        braced_block(many1(ws(parse_single_where_clause))),
    )(input)
}

fn parse_not_clause(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag_no_case("NOT")),
        braced_block(many1(ws(parse_single_where_clause))),
    )(input)
}

// --- Solution Modifiers ---

fn parse_order_by_clause(input: &str) -> IResult<&str, Vec<OrderByCondition>> {
    preceded(
        ws(tag_no_case("ORDER BY")),
        separated_list1(ws(char(',')), parse_order_by_condition),
    )(input)
}

fn parse_order_by_condition(input: &str) -> IResult<&str, OrderByCondition> {
    map(
        pair(
            variable,
            opt(alt((
                map(ws(tag_no_case("ASC")), |_| OrderDirection::Asc),
                map(ws(tag_no_case("DESC")), |_| OrderDirection::Desc),
            ))),
        ),
        |(variable, direction)| OrderByCondition {
            variable,
            direction: direction.unwrap_or(OrderDirection::Asc),
        },
    )(input)
}

fn parse_limit_clause(input: &str) -> IResult<&str, u64> {
    preceded(
        ws(tag_no_case("LIMIT")),
        map(nom::character::complete::u64, |n| n),
    )(input)
}

fn parse_offset_clause(input: &str) -> IResult<&str, u64> {
    preceded(
        ws(tag_no_case("OFFSET")),
        map(nom::character::complete::u64, |n| n),
    )(input)
}
