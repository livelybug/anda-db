use nom::{
    IResult, Mode, Parser,
    branch::alt,
    bytes::{
        complete::{tag, tag_no_case},
        take,
    },
    character::{
        anychar,
        complete::{alpha1, alphanumeric1, char, multispace0},
        none_of,
    },
    combinator::{map, map_opt, map_res, opt, recognize, value, verify},
    error::{Error, ErrorKind, ParseError},
    multi::{fold, many0, separated_list1},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, separated_pair, terminated},
};
use std::{collections::HashMap, str::FromStr};

use crate::ast::{KeyValue, Number, Value};

/// Consumes whitespace around a parser.
pub fn ws<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = O, Error = E>
where
    E: ParseError<&'a str>,
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(multispace0, f, multispace0)
}

/// Parses the contents of a block enclosed in curly braces.
pub fn braced_block<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = O, Error = E>
where
    E: ParseError<&'a str>,
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(ws(char('{')), f, ws(char('}')))
}

/// Parses the contents of a block enclosed in parentheses.
pub fn parenthesized_block<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = O, Error = E>
where
    E: ParseError<&'a str>,
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(ws(char('(')), f, ws(char(')')))
}

/// Parses a valid identifier (e.g., for variables, types, predicates).
pub fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

/// Parses a KIP variable, like `?my_var`.
pub fn variable(input: &str) -> IResult<&str, String> {
    map(preceded(char('?'), identifier), |s| s.to_string()).parse(input)
}

/// Parses a local handle in KML, like `@my_handle`.
pub fn local_handle(input: &str) -> IResult<&str, String> {
    map(preceded(char('@'), identifier), |s| s.to_string()).parse(input)
}

/// Parses a double-quoted string, handling escaped quotes.
pub fn quoted_string(input: &str) -> IResult<&str, String> {
    // https://github.com/rust-bakery/nom/blob/main/examples/json2.rs#L121
    delimited(
        char('"'),
        fold(0.., character(), String::new, |mut string, c| {
            string.push(c);
            string
        }),
        char('"'),
    )
    .parse(input)
}

/// Parses any KIP value (string, number, boolean, null).
pub fn kip_value(input: &str) -> IResult<&str, Value> {
    alt((
        value(Value::Null, tag_no_case("null")),
        value(Value::Bool(true), tag_no_case("true")),
        value(Value::Bool(false), tag_no_case("false")),
        map(quoted_string, Value::String),
        map(parse_number, Value::Number),
    ))
    .parse(input)
}

/// Parses a key-value pair, like `name: "Aspirin"`.
pub fn key_value_pair(input: &str) -> IResult<&str, KeyValue> {
    map(
        separated_pair(identifier, ws(char(':')), kip_value),
        |(k, v)| KeyValue {
            key: k.to_string(),
            value: v,
        },
    )
    .parse(input)
}

/// Parses a list of key-value pairs inside braces, like `{ key1: val1, key2: val2 }`.
pub fn key_value_map(input: &str) -> IResult<&str, HashMap<String, Value>> {
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
    )
    .parse(input)
}

fn parse_number(input: &str) -> IResult<&str, Number> {
    let (next_input, num_str) = recognize_float(input)?;
    let num = Number::from_str(num_str)
        .map_err(|_| nom::Err::Error(Error::new(num_str, ErrorKind::Digit)))?;
    Ok((next_input, num))
}

fn u16_hex<'a>() -> impl Parser<&'a str, Output = u16, Error = Error<&'a str>> {
    map_res(take(4usize), |s| u16::from_str_radix(s, 16))
}

fn unicode_escape<'a>() -> impl Parser<&'a str, Output = char, Error = Error<&'a str>> {
    map_opt(
        alt((
            // Not a surrogate
            map(
                verify(u16_hex(), |cp| !(0xD800..0xE000).contains(cp)),
                |cp| cp as u32,
            ),
            // See https://en.wikipedia.org/wiki/UTF-16#Code_points_from_U+010000_to_U+10FFFF for details
            map(
                verify(
                    separated_pair(u16_hex(), tag("\\u"), u16_hex()),
                    |(high, low)| (0xD800..0xDC00).contains(high) && (0xDC00..0xE000).contains(low),
                ),
                |(high, low)| {
                    let high_ten = (high as u32) - 0xD800;
                    let low_ten = (low as u32) - 0xDC00;
                    (high_ten << 10) + low_ten + 0x10000
                },
            ),
        )),
        // Could probably be replaced with .unwrap() or _unchecked due to the verify checks
        std::char::from_u32,
    )
}

fn character<'a>() -> impl Parser<&'a str, Output = char, Error = Error<&'a str>> {
    Character
}

struct Character;

impl<'a> Parser<&'a str> for Character {
    type Output = char;

    type Error = Error<&'a str>;

    fn process<OM: nom::OutputMode>(
        &mut self,
        input: &'a str,
    ) -> nom::PResult<OM, &'a str, Self::Output, Self::Error> {
        let (input, c): (&str, char) =
            none_of("\"").process::<nom::OutputM<nom::Emit, OM::Error, OM::Incomplete>>(input)?;
        if c == '\\' {
            alt((
                map_res(anychar, |c| {
                    Ok(match c {
                        '"' | '\\' | '/' => c,
                        'b' => '\x08',
                        'f' => '\x0C',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        _ => return Err(()),
                    })
                }),
                preceded(char('u'), unicode_escape()),
            ))
            .process::<OM>(input)
        } else {
            Ok((input, OM::Output::bind(|| c)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::Error;

    #[test]
    fn test_ws() {
        assert_eq!(
            ws(char::<&str, Error<_>>('a')).parse("  a  "),
            Ok(("", 'a'))
        );
        assert_eq!(ws(char::<&str, Error<_>>('a')).parse("a"), Ok(("", 'a')));
        assert_eq!(
            ws(char::<&str, Error<_>>('a')).parse("\n\t a \r\n"),
            Ok(("", 'a'))
        );
    }

    #[test]
    fn test_identifier() {
        assert_eq!(identifier("hello"), Ok(("", "hello")));
        assert_eq!(identifier("_private"), Ok(("", "_private")));
        assert_eq!(identifier("var123"), Ok(("", "var123")));
        assert_eq!(identifier("hello_world"), Ok(("", "hello_world")));
        assert!(identifier("123invalid").is_err());
        assert!(identifier("").is_err());
    }

    #[test]
    fn test_variable() {
        assert_eq!(variable("?my_var"), Ok(("", "my_var".to_string())));
        assert_eq!(variable("?_private"), Ok(("", "_private".to_string())));
        assert_eq!(variable("?var123"), Ok(("", "var123".to_string())));
        assert!(variable("my_var").is_err());
        assert!(variable("?").is_err());
    }

    #[test]
    fn test_local_handle() {
        assert_eq!(
            local_handle("@my_handle"),
            Ok(("", "my_handle".to_string()))
        );
        assert_eq!(local_handle("@_private"), Ok(("", "_private".to_string())));
        assert_eq!(
            local_handle("@handle123"),
            Ok(("", "handle123".to_string()))
        );
        assert!(local_handle("my_handle").is_err());
        assert!(local_handle("@").is_err());
    }

    #[test]
    fn test_quoted_string() {
        assert_eq!(quoted_string(r#""hello""#), Ok(("", "hello".to_string())));
        assert_eq!(
            quoted_string(r#""hello world""#),
            Ok(("", "hello world".to_string()))
        );
        assert_eq!(
            quoted_string(r#""with \"quotes\"""#),
            Ok(("", r#"with "quotes""#.to_string()))
        );
        assert_eq!(
            quoted_string(r#""with \\backslash""#),
            Ok(("", r#"with \backslash"#.to_string()))
        );
        assert_eq!(
            quoted_string(r#""with \n newline""#),
            Ok(("", "with \n newline".to_string()))
        );
        assert_eq!(
            quoted_string(r#""with \t tab""#),
            Ok(("", "with \t tab".to_string()))
        );
        assert_eq!(
            quoted_string(r#""with \r return""#),
            Ok(("", "with \r return".to_string()))
        );
        assert_eq!(quoted_string(r#""""#), Ok(("", "".to_string())));
        assert!(quoted_string(r#""unclosed"#).is_err());
    }

    #[test]
    fn test_kip_value() {
        assert_eq!(kip_value("42"), Ok(("", Value::Number(Number::from(42)))));
        assert_eq!(kip_value("-42"), Ok(("", Value::Number(Number::from(-42)))));
        assert_eq!(
            kip_value("3.14"),
            Ok(("", Value::Number(Number::from_f64(3.14f64).unwrap())))
        );
        assert_eq!(
            kip_value(r#""hello""#),
            Ok(("", Value::String("hello".to_string())))
        );
        assert_eq!(kip_value("true"), Ok(("", Value::Bool(true))));
        assert_eq!(kip_value("TRUE"), Ok(("", Value::Bool(true))));
        assert_eq!(kip_value("false"), Ok(("", Value::Bool(false))));
        assert_eq!(kip_value("FALSE"), Ok(("", Value::Bool(false))));
        assert_eq!(kip_value("null"), Ok(("", Value::Null)));
        assert_eq!(kip_value("NULL"), Ok(("", Value::Null)));
    }

    #[test]
    fn test_key_value_pair() {
        let result = key_value_pair(r#"name: "John""#);
        assert!(result.is_ok());
        let (_, kv) = result.unwrap();
        assert_eq!(kv.key, "name");
        assert_eq!(kv.value, Value::String("John".to_string()));

        let result = key_value_pair("age: 25");
        assert!(result.is_ok());
        let (_, kv) = result.unwrap();
        assert_eq!(kv.key, "age");
        assert_eq!(kv.value, Value::Number(Number::from(25)));

        let result = key_value_pair("active: true");
        assert!(result.is_ok());
        let (_, kv) = result.unwrap();
        assert_eq!(kv.key, "active");
        assert_eq!(kv.value, Value::Bool(true));
    }

    #[test]
    fn test_key_value_map() {
        let result = key_value_map(r#"{ name: "John", age: 25 }"#);
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("name"), Some(&Value::String("John".to_string())));
        assert_eq!(map.get("age"), Some(&Value::Number(Number::from(25))));

        // Test empty map
        let result = key_value_map("{}");
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 0);

        // Test with trailing comma
        let result = key_value_map(r#"{ name: "John", age: 25, }"#);
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 2);

        // Test with whitespace
        let result = key_value_map(
            r#"{
            name: "John",
            age: 25,
            active: true
        }"#,
        );
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get("active"), Some(&Value::Bool(true)));

        // Test single item
        let result = key_value_map(r#"{ name: "John" }"#);
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_braced_block() {
        let mut parser = braced_block(char::<&str, Error<_>>('a'));
        assert_eq!(parser.parse("{ a }"), Ok(("", 'a')));
        assert_eq!(parser.parse("{a}"), Ok(("", 'a')));
        assert_eq!(parser.parse("{\n  a  \n}"), Ok(("", 'a')));
        assert!(parser.parse("{ b }").is_err());
        assert!(parser.parse("{ a").is_err());
        assert!(parser.parse("a }").is_err());
    }

    #[test]
    fn test_parenthesized_block() {
        let mut parser = parenthesized_block(char::<&str, Error<_>>('a'));
        assert_eq!(parser.parse("( a )"), Ok(("", 'a')));
        assert_eq!(parser.parse("(a)"), Ok(("", 'a')));
        assert_eq!(parser.parse("(\n  a  \n)"), Ok(("", 'a')));
        assert!(parser.parse("( b )").is_err());
        assert!(parser.parse("( a").is_err());
        assert!(parser.parse("a )").is_err());
    }

    #[test]
    fn test_complex_nested_values() {
        let result = kip_value(r#""nested \"quotes\" and \n newlines""#);
        assert!(result.is_ok());
        let (_, value) = result.unwrap();
        assert_eq!(
            value,
            Value::String("nested \"quotes\" and \n newlines".to_string())
        );
    }

    #[test]
    fn test_key_value_map_complex() {
        let input = r#"{
            name: "John Doe",
            age: 30,
            height: 5.9,
            active: true,
            score: null,
            negative: -42,
        }"#;

        let result = key_value_map(input);
        assert!(result.is_ok());
        let (_, map) = result.unwrap();
        assert_eq!(map.len(), 6);
        assert_eq!(
            map.get("name"),
            Some(&Value::String("John Doe".to_string()))
        );
        assert_eq!(map.get("age"), Some(&Value::Number(Number::from(30))));
        assert_eq!(
            map.get("height"),
            Some(&Value::Number(Number::from_f64(5.9).unwrap()))
        );
        assert_eq!(map.get("active"), Some(&Value::Bool(true)));
        assert_eq!(map.get("score"), Some(&Value::Null));
        assert_eq!(map.get("negative"), Some(&Value::Number(Number::from(-42))));
    }

    #[test]
    fn test_edge_cases() {
        // Test identifier edge cases
        assert_eq!(identifier("a"), Ok(("", "a")));
        assert_eq!(identifier("a1"), Ok(("", "a1")));
        assert_eq!(identifier("_"), Ok(("", "_")));
        assert_eq!(identifier("_1"), Ok(("", "_1")));

        // Test that identifier stops at non-alphanumeric/underscore
        assert_eq!(identifier("hello-world"), Ok(("-world", "hello")));
        assert_eq!(identifier("hello world"), Ok((" world", "hello")));
    }

    #[test]
    fn test_kip_value_precedence() {
        // Test that boolean parsing works correctly
        assert_eq!(kip_value("true"), Ok(("", Value::Bool(true))));
        assert_eq!(kip_value("false"), Ok(("", Value::Bool(false))));
        assert_eq!(kip_value("null"), Ok(("", Value::Null)));

        // Test mixed case
        assert_eq!(kip_value("True"), Ok(("", Value::Bool(true))));
        assert_eq!(kip_value("False"), Ok(("", Value::Bool(false))));
        assert_eq!(kip_value("Null"), Ok(("", Value::Null)));

        // Test that numbers are parsed correctly
        assert_eq!(kip_value("0"), Ok(("", Value::Number(Number::from(0)))));
        assert_eq!(
            kip_value("-0"),
            Ok(("", Value::Number(Number::from_f64(-0.0).unwrap())))
        );
        assert_eq!(
            kip_value("0.0"),
            Ok(("", Value::Number(Number::from_f64(0.0).unwrap())))
        );
    }

    #[test]
    fn test_error_handling() {
        // Test malformed inputs
        assert!(quoted_string("'single quotes'").is_err());
        assert!(quoted_string("\"unclosed").is_err());
        assert!(key_value_pair("key: ").is_err());
        assert!(key_value_map("{ key: }").is_err());
        assert!(key_value_map("{ key value }").is_err());
    }
}
