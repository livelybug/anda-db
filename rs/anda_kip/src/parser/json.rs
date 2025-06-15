// https://github.com/rust-bakery/nom/blob/main/examples/json2.rs

use nom::{
    IResult, Mode, Parser,
    branch::alt,
    bytes::{tag, tag_no_case, take},
    character::{anychar, char, complete::multispace0, none_of},
    combinator::{map, map_opt, map_res, value, verify},
    error::{Error, ErrorKind, ParseError},
    multi::{fold, separated_list0},
    number::complete::recognize_float,
    sequence::{delimited, preceded, separated_pair},
};
use std::str::FromStr;

use crate::{Json, Map, Number};

/// Parses a double-quoted string, handling escaped quotes.
pub fn quoted_string(input: &str) -> IResult<&str, String> {
    string().parse(input)
}

pub fn parse_number(input: &str) -> IResult<&str, Number> {
    let (next_input, num_str) = recognize_float(input)?;
    let num = Number::from_str(num_str)
        .map_err(|_| nom::Err::Error(Error::new(num_str, ErrorKind::Digit)))?;
    Ok((next_input, num))
}

pub fn ws<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = O, Error = E>
where
    E: ParseError<&'a str>,
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(multispace0, f, multispace0)
}

pub fn json_value<'a>() -> impl Parser<&'a str, Output = Json, Error = Error<&'a str>> {
    JsonParser
}

fn string<'a>() -> impl Parser<&'a str, Output = String, Error = Error<&'a str>> {
    delimited(
        char('"'),
        fold(0.., character(), String::new, |mut string, c| {
            string.push(c);
            string
        }),
        char('"'),
    )
}

fn array<'a>() -> impl Parser<&'a str, Output = Vec<Json>, Error = Error<&'a str>> {
    delimited(
        char('['),
        ws(separated_list0(ws(char(',')), json_value())),
        char(']'),
    )
}

fn object<'a>() -> impl Parser<&'a str, Output = Map<String, Json>, Error = Error<&'a str>> {
    map(
        delimited(
            char('{'),
            ws(separated_list0(
                ws(char(',')),
                separated_pair(string(), ws(char(':')), json_value()),
            )),
            char('}'),
        ),
        |key_values| key_values.into_iter().collect(),
    )
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

pub fn character<'a>() -> impl Parser<&'a str, Output = char, Error = Error<&'a str>> {
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

struct JsonParser;

impl<'a> Parser<&'a str> for JsonParser {
    type Output = Json;
    type Error = Error<&'a str>;

    fn process<OM: nom::OutputMode>(
        &mut self,
        input: &'a str,
    ) -> nom::PResult<OM, &'a str, Self::Output, Self::Error> {
        let mut parser = alt((
            value(Json::Null, tag_no_case("null")),
            value(Json::Bool(true), tag_no_case("true")),
            value(Json::Bool(false), tag_no_case("false")),
            map(string(), Json::String),
            map(parse_number, Json::Number),
            map(array(), Json::Array),
            map(object(), Json::Object),
        ));

        parser.process::<OM>(input)
    }
}

#[cfg(test)]
mod tests {}
