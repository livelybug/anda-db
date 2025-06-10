use nom::{
    IResult,
    branch::alt,
    bytes::complete::{escaped_transform, is_not, tag, tag_no_case},
    character::complete::{
        alpha1, alphanumeric1, char, i64 as parse_i64, multispace0, u64 as parse_u64,
    },
    combinator::{map, opt, recognize, value},
    multi::{many0, separated_list1},
    number::complete::double,
    sequence::{delimited, pair, preceded, separated_pair, terminated},
};
use std::collections::HashMap;

use crate::ast::KeyValue;
use crate::nexus::KipValue;

/// Consumes whitespace around a parser.
pub fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

/// Parses a valid identifier (e.g., for variables, types, predicates).
pub fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))(input)
}

/// Parses a KIP variable, like `?my_var`.
pub fn variable(input: &str) -> IResult<&str, String> {
    map(preceded(char('?'), identifier), |s| s.to_string())(input)
}

/// Parses a local handle in KML, like `@my_handle`.
pub fn local_handle(input: &str) -> IResult<&str, String> {
    map(preceded(char('@'), identifier), |s| s.to_string())(input)
}

/// Parses a double-quoted string, handling escaped quotes.
pub fn quoted_string(input: &str) -> IResult<&str, String> {
    map(
        delimited(
            char('"'),
            escaped_transform(
                is_not("\\\""),
                '\\',
                alt((
                    value("\"", tag("\"")),
                    value("\\", tag("\\")),
                    value("\n", tag("n")),
                    value("\t", tag("t")),
                    value("\r", tag("r")),
                )),
            ),
            char('"'),
        ),
        |s| s,
    )(input)
}

// pub fn quoted_string(input: &str) -> IResult<&str, String> {
//     delimited(
//         char('"'),
//         fold_many0(
//             alt((
//                 map(none_of("\\\""), |c| c.to_string()),
//                 map(preceded(char('\\'), alt((
//                     value('"', char('"')),
//                     value('\\', char('\\')),
//                     value('\n', char('n')),
//                     value('\t', char('t')),
//                     value('\r', char('r')),
//                 ))), |c| c.to_string()),
//             )),
//             String::new,
//             |mut acc, item| {
//                 acc.push_str(&item);
//                 acc
//             },
//         ),
//         char('"'),
//     )(input)
// }

/// Parses any KIP value (string, integer, float, boolean, null).
pub fn kip_value(input: &str) -> IResult<&str, KipValue> {
    alt((
        map(double, KipValue::Float),
        map(parse_u64, KipValue::Uint),
        map(parse_i64, KipValue::Int),
        map(quoted_string, KipValue::String),
        map(tag_no_case("true"), |_| KipValue::Bool(true)),
        map(tag_no_case("false"), |_| KipValue::Bool(false)),
        map(tag_no_case("null"), |_| KipValue::Null),
    ))(input)
}

/// Parses a key-value pair, like `name: "Aspirin"`.
pub fn key_value_pair(input: &str) -> IResult<&str, KeyValue> {
    map(
        separated_pair(identifier, ws(char(':')), kip_value),
        |(k, v)| KeyValue {
            key: k.to_string(),
            value: v,
        },
    )(input)
}

/// Parses a list of key-value pairs inside braces, like `{ key1: val1, key2: val2 }`.
pub fn key_value_map(input: &str) -> IResult<&str, HashMap<String, KipValue>> {
    map(
        delimited(
            ws(char('{')),
            opt(terminated(
                separated_list1(ws(char(',')), key_value_pair),
                opt(ws(char(','))), // Allow trailing comma
            )),
            ws(char('}')),
        ),
        |opt_kvs| {
            opt_kvs
                .unwrap_or_default()
                .into_iter()
                .map(|kv| (kv.key, kv.value))
                .collect()
        },
    )(input)
}

/// Parses the contents of a block enclosed in curly braces.
pub fn braced_block<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(ws(char('{')), inner, ws(char('}')))
}

/// Parses the contents of a block enclosed in parentheses.
pub fn parenthesized_block<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(ws(char('(')), inner, ws(char(')')))
}
