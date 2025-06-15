use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, multispace1},
    combinator::{map, opt},
    multi::{fold, many1, separated_list1},
    sequence::{pair, preceded, terminated},
};

use super::common::*;
use crate::ast::*;

// --- Top Level KQL Parser ---

pub fn parse_kql_query(input: &str) -> IResult<&str, KqlQuery> {
    map(
        (
            ws(parse_find_clause),
            ws(parse_where_block),
            opt(ws(parse_order_by_clause)),
            opt(ws(parse_limit_clause)),
            opt(ws(parse_offset_clause)),
        ),
        |(find_clause, where_clauses, order_by, limit, offset)| KqlQuery {
            find_clause,
            where_clauses,
            order_by,
            limit,
            offset,
        },
    )
    .parse(input)
}

// --- FIND Clause ---

fn parse_find_clause(input: &str) -> IResult<&str, FindClause> {
    map(
        preceded(
            ws(tag("FIND")),
            parenthesized_block(separated_list1(ws(char(',')), parse_find_expression)),
        ),
        |expressions| FindClause { expressions },
    )
    .parse(input)
}

fn parse_find_expression(input: &str) -> IResult<&str, FindExpression> {
    alt((parse_aggregation_expression, parse_find_variable)).parse(input)
}

fn parse_find_variable(input: &str) -> IResult<&str, FindExpression> {
    map(variable, FindExpression::Variable).parse(input)
}

fn parse_aggregation_expression(input: &str) -> IResult<&str, FindExpression> {
    map(
        (
            parse_aggregation_function,
            parenthesized_block((opt(terminated(tag("DISTINCT"), multispace1)), variable)),
            preceded(ws(tag("AS")), variable),
        ),
        |(func, (distinct, var), alias)| FindExpression::Aggregation {
            func,
            var,
            distinct: distinct.is_some(),
            alias,
        },
    )
    .parse(input)
}

fn parse_aggregation_function(input: &str) -> IResult<&str, AggregationFunction> {
    alt((
        map(tag("COUNT"), |_| AggregationFunction::Count),
        map(tag("COLLECT"), |_| AggregationFunction::Collect),
        map(tag("SUM"), |_| AggregationFunction::Sum),
        map(tag("AVG"), |_| AggregationFunction::Avg),
        map(tag("MIN"), |_| AggregationFunction::Min),
        map(tag("MAX"), |_| AggregationFunction::Max),
    ))
    .parse(input)
}

// --- WHERE Clause ---

fn parse_where_block(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(ws(tag("WHERE")), parse_where_group).parse(input)
}

pub fn parse_where_group(input: &str) -> IResult<&str, Vec<WhereClause>> {
    braced_block(many1(ws(parse_single_where_clause))).parse(input)
}

fn parse_single_where_clause(input: &str) -> IResult<&str, WhereClause> {
    alt((
        map(parse_optional_clause, WhereClause::Optional),
        map(parse_not_clause, WhereClause::Not),
        map(parse_prop_pattern, WhereClause::Proposition),
        map(parse_grounding_clause, WhereClause::Grounding),
        map(parse_attr_clause, WhereClause::Attribute),
        map(parse_filter_clause, WhereClause::Filter),
        map(parse_union_expression, WhereClause::Union),
    ))
    .parse(input)
}

// --- WHERE Clause Sub-parsers ---

fn parse_grounding_clause(input: &str) -> IResult<&str, Grounding> {
    map(
        (
            variable,
            parenthesized_block(separated_list1(ws(char(',')), key_value_pair)),
        ),
        |(variable, constraints)| Grounding {
            variable,
            constraints,
        },
    )
    .parse(input)
}

fn on_clause(input: &str) -> IResult<&str, OnClause> {
    map(
        preceded(
            ws(tag("ON")),
            braced_block(separated_list1(ws(char(',')), key_value_pair)),
        ),
        |keys| OnClause { keys },
    )
    .parse(input)
}

fn parse_prop_term(input: &str) -> IResult<&str, PropTerm> {
    alt((
        map(variable, PropTerm::Variable),
        map(on_clause, PropTerm::Node),
        map(parse_nested_prop_pattern, |p| {
            PropTerm::NestedProp(Box::new(p))
        }),
    ))
    .parse(input)
}

fn parse_pred_term(input: &str) -> IResult<&str, PredTerm> {
    alt((
        map(variable, PredTerm::Variable),
        map(quoted_string, PredTerm::Literal),
    ))
    .parse(input)
}

fn parse_prop_pattern(input: &str) -> IResult<&str, PropositionPattern> {
    map(
        (
            preceded(
                ws(tag("PROP")),
                parenthesized_block((
                    ws(parse_prop_term),
                    ws(char(',')),
                    ws(parse_pred_term),
                    ws(char(',')),
                    ws(parse_prop_term),
                )),
            ),
            opt(braced_block(separated_list1(ws(char(',')), key_value_pair))),
        ),
        |((subject, _, predicate, _, object), metadata_constraints)| PropositionPattern {
            subject,
            predicate,
            object,
            metadata_constraints,
        },
    )
    .parse(input)
}

fn parse_nested_prop_pattern(input: &str) -> IResult<&str, PropositionPattern> {
    map(
        (
            parenthesized_block((
                ws(parse_prop_term),
                ws(char(',')),
                ws(parse_pred_term),
                ws(char(',')),
                ws(parse_prop_term),
            )),
            opt(braced_block(separated_list1(ws(char(',')), key_value_pair))),
        ),
        |((subject, _, predicate, _, object), metadata_constraints)| PropositionPattern {
            subject,
            predicate,
            object,
            metadata_constraints,
        },
    )
    .parse(input)
}

fn parse_attr_clause(input: &str) -> IResult<&str, AttributePattern> {
    map(
        preceded(
            ws(tag("ATTR")),
            parenthesized_block((
                ws(variable),
                ws(char(',')),
                ws(quoted_string),
                ws(char(',')),
                ws(variable),
            )),
        ),
        |(node_variable, _, attribute_name, _, value_variable)| AttributePattern {
            node_variable,
            attribute_name,
            value_variable,
        },
    )
    .parse(input)
}

fn parse_filter_clause(input: &str) -> IResult<&str, FilterCondition> {
    map(
        preceded(
            ws(tag("FILTER")),
            (
                parenthesized_block(parse_filter_expression),
                opt(braced_block(parse_subquery_expression)),
            ),
        ),
        |(expression, subquery)| FilterCondition {
            expression,
            subquery,
        },
    )
    .parse(input)
}

fn parse_filter_expression(input: &str) -> IResult<&str, FilterExpression> {
    parse_logical_or_expression(input)
}

// 解析逻辑 OR 表达式（最低优先级）
fn parse_logical_or_expression(input: &str) -> IResult<&str, FilterExpression> {
    let (input, left) = parse_logical_and_expression(input)?;

    fold(
        0..,
        preceded(ws(tag("||")), parse_logical_and_expression),
        move || left.clone(),
        |acc, right| FilterExpression::Logical {
            left: Box::new(acc),
            operator: LogicalOperator::Or,
            right: Box::new(right),
        },
    )
    .parse(input)
}

// 解析逻辑 AND 表达式
fn parse_logical_and_expression(input: &str) -> IResult<&str, FilterExpression> {
    let (input, left) = parse_unary_expression(input)?;

    fold(
        0..,
        preceded(ws(tag("&&")), parse_unary_expression),
        move || left.clone(),
        |acc, right| FilterExpression::Logical {
            left: Box::new(acc),
            operator: LogicalOperator::And,
            right: Box::new(right),
        },
    )
    .parse(input)
}

// 解析一元表达式（NOT）
fn parse_unary_expression(input: &str) -> IResult<&str, FilterExpression> {
    alt((
        map(preceded(ws(char('!')), parse_primary_expression), |expr| {
            FilterExpression::Not(Box::new(expr))
        }),
        parse_primary_expression,
    ))
    .parse(input)
}

// 解析基本表达式（比较、函数、括号）
fn parse_primary_expression(input: &str) -> IResult<&str, FilterExpression> {
    alt((
        // 括号表达式
        parenthesized_block(parse_filter_expression),
        // 函数调用
        parse_function_expression,
        // 比较表达式
        parse_comparison_expression,
    ))
    .parse(input)
}

// 解析比较表达式
fn parse_comparison_expression(input: &str) -> IResult<&str, FilterExpression> {
    map(
        (
            parse_filter_operand,
            ws(parse_comparison_operator),
            parse_filter_operand,
        ),
        |(left, operator, right)| FilterExpression::Comparison {
            left,
            operator,
            right,
        },
    )
    .parse(input)
}

// 解析函数表达式
fn parse_function_expression(input: &str) -> IResult<&str, FilterExpression> {
    map(
        (
            parse_filter_function,
            parenthesized_block(separated_list1(ws(char(',')), parse_filter_operand)),
        ),
        |(func, args)| FilterExpression::Function { func, args },
    )
    .parse(input)
}

// 解析过滤器操作数
fn parse_filter_operand(input: &str) -> IResult<&str, FilterOperand> {
    alt((
        map(variable, FilterOperand::Variable),
        map(kip_value, FilterOperand::Literal),
    ))
    .parse(input)
}

// 解析比较运算符
fn parse_comparison_operator(input: &str) -> IResult<&str, ComparisonOperator> {
    alt((
        map(tag("=="), |_| ComparisonOperator::Equal),
        map(tag("!="), |_| ComparisonOperator::NotEqual),
        map(tag("<="), |_| ComparisonOperator::LessEqual),
        map(tag(">="), |_| ComparisonOperator::GreaterEqual),
        map(tag("<"), |_| ComparisonOperator::LessThan),
        map(tag(">"), |_| ComparisonOperator::GreaterThan),
    ))
    .parse(input)
}

// 解析过滤器函数
fn parse_filter_function(input: &str) -> IResult<&str, FilterFunction> {
    alt((
        map(tag("CONTAINS"), |_| FilterFunction::Contains),
        map(tag("STARTS_WITH"), |_| FilterFunction::StartsWith),
        map(tag("ENDS_WITH"), |_| FilterFunction::EndsWith),
        map(tag("REGEX"), |_| FilterFunction::Regex),
    ))
    .parse(input)
}

// 解析子查询表达式
fn parse_subquery_expression(input: &str) -> IResult<&str, SubqueryExpression> {
    map(
        (
            map(
                preceded(
                    ws(tag("SELECT")),
                    parenthesized_block(separated_list1(ws(char(',')), parse_find_expression)),
                ),
                |expressions| FindClause { expressions },
            ),
            ws(parse_where_block),
        ),
        |(select_clause, where_clauses)| SubqueryExpression {
            select_clause,
            where_clauses,
        },
    )
    .parse(input)
}

fn parse_optional_clause(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag("OPTIONAL")),
        braced_block(many1(ws(parse_single_where_clause))),
    )
    .parse(input)
}

fn parse_not_clause(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag("NOT")),
        braced_block(many1(ws(parse_single_where_clause))),
    )
    .parse(input)
}

fn parse_union_expression(input: &str) -> IResult<&str, Vec<WhereClause>> {
    preceded(
        ws(tag("UNION")),
        braced_block(many1(ws(parse_single_where_clause))),
    )
    .parse(input)
}

// --- Solution Modifiers ---

fn parse_order_by_clause(input: &str) -> IResult<&str, Vec<OrderByCondition>> {
    preceded(
        ws(tag("ORDER ")),
        preceded(
            ws(tag("BY ")),
            separated_list1(ws(char(',')), parse_order_by_condition),
        ),
    )
    .parse(input)
}

fn parse_order_by_condition(input: &str) -> IResult<&str, OrderByCondition> {
    map(
        pair(
            variable,
            opt(alt((
                map(ws(tag("ASC")), |_| OrderDirection::Asc),
                map(ws(tag("DESC")), |_| OrderDirection::Desc),
            ))),
        ),
        |(variable, direction)| OrderByCondition {
            variable,
            direction: direction.unwrap_or(OrderDirection::Asc),
        },
    )
    .parse(input)
}

fn parse_limit_clause(input: &str) -> IResult<&str, u64> {
    preceded(ws(tag("LIMIT")), map(nom::character::complete::u64, |n| n)).parse(input)
}

fn parse_offset_clause(input: &str) -> IResult<&str, u64> {
    preceded(ws(tag("OFFSET")), map(nom::character::complete::u64, |n| n)).parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_find_query() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.find_clause.expressions.len(), 1);
        match &query.find_clause.expressions[0] {
            FindExpression::Variable(var) => assert_eq!(var, "drug_name"),
            _ => panic!("Expected variable expression"),
        }
        assert_eq!(query.where_clauses.len(), 2);
    }

    #[test]
    fn test_parse_aggregation_find() {
        let input = r#"
            FIND(?drug_class, COUNT(?drug) AS ?drug_count)
            WHERE {
                ?drug(type: "Drug")
                PROP(?drug, "is_class_of", ?drug_class)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.find_clause.expressions.len(), 2);

        match &query.find_clause.expressions[0] {
            FindExpression::Variable(var) => assert_eq!(var, "drug_class"),
            _ => panic!("Expected variable expression"),
        }

        match &query.find_clause.expressions[1] {
            FindExpression::Aggregation {
                func,
                var,
                distinct,
                alias,
            } => {
                assert_eq!(*func, AggregationFunction::Count);
                assert_eq!(var, "drug");
                assert!(!distinct);
                assert_eq!(alias, "drug_count");
            }
            _ => panic!("Expected aggregation expression"),
        }
    }

    #[test]
    fn test_parse_aggregation_with_distinct() {
        let input = r#"
            FIND(COUNT(DISTINCT ?symptom) AS ?unique_symptoms)
            WHERE {
                PROP(?drug, "treats", ?symptom)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.find_clause.expressions[0] {
            FindExpression::Aggregation {
                func,
                var,
                distinct,
                alias,
            } => {
                assert_eq!(*func, AggregationFunction::Count);
                assert_eq!(var, "symptom");
                assert!(*distinct);
                assert_eq!(alias, "unique_symptoms");
            }
            _ => panic!("Expected aggregation expression"),
        }
    }

    #[test]
    fn test_parse_all_aggregation_functions() {
        let functions = vec![
            ("COUNT(?var) AS ?count", AggregationFunction::Count),
            ("COLLECT(?var) AS ?list", AggregationFunction::Collect),
            ("SUM(?var) AS ?sum", AggregationFunction::Sum),
            ("AVG(?var) AS ?avg", AggregationFunction::Avg),
            ("MIN(?var) AS ?min", AggregationFunction::Min),
            ("MAX(?var) AS ?max", AggregationFunction::Max),
        ];

        for (func_str, expected_func) in functions {
            let input = format!("FIND({func_str}) WHERE {{ ?x(type: \"Test\") }}");
            let result = parse_kql_query(&input);
            assert!(result.is_ok(), "Failed to parse: {func_str}");

            let (_, query) = result.unwrap();
            match &query.find_clause.expressions[0] {
                FindExpression::Aggregation { func, .. } => {
                    assert_eq!(*func, expected_func);
                }
                _ => panic!("Expected aggregation for: {func_str}"),
            }
        }
    }

    #[test]
    fn test_parse_grounding_clause() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug", name: "Aspirin")
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Grounding(grounding) => {
                assert_eq!(grounding.variable, "drug");
                assert_eq!(grounding.constraints.len(), 2);
                assert_eq!(grounding.constraints[0].key, "type");
                assert_eq!(grounding.constraints[1].key, "name");
            }
            _ => panic!("Expected grounding clause"),
        }
    }

    #[test]
    fn test_parse_proposition_pattern() {
        let input = r#"
            FIND(?drug_name, ?symptom_name)
            WHERE {
                PROP(?drug, "treats", ?symptom)
                ATTR(?drug, "name", ?drug_name)
                ATTR(?symptom, "name", ?symptom_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Proposition(prop) => {
                assert_eq!(prop.predicate, PredTerm::Literal("treats".to_string()));
                match &prop.subject {
                    PropTerm::Variable(var) => assert_eq!(var, "drug"),
                    _ => panic!("Expected variable subject"),
                }
                match &prop.object {
                    PropTerm::Variable(var) => assert_eq!(var, "symptom"),
                    _ => panic!("Expected variable object"),
                }
            }
            _ => panic!("Expected proposition clause"),
        }
    }

    #[test]
    fn test_parse_proposition_with_on_clause() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                PROP(?drug, "treats", ON { type: "Symptom", name: "Headache" })
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Proposition(prop) => match &prop.object {
                PropTerm::Node(on_clause) => {
                    assert_eq!(on_clause.keys.len(), 2);
                    assert_eq!(on_clause.keys[0].key, "type");
                    assert_eq!(on_clause.keys[1].key, "name");
                }
                _ => panic!("Expected ON clause object"),
            },
            _ => panic!("Expected proposition clause"),
        }
    }

    #[test]
    fn test_parse_proposition_with_metadata() {
        let input = r#"
            FIND(?drug_name, ?symptom_name)
            WHERE {
                PROP(?drug, "treats", ?symptom) { confidence: 0.9, source: "clinical_trial" }
                ATTR(?drug, "name", ?drug_name)
                ATTR(?symptom, "name", ?symptom_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Proposition(prop) => {
                assert!(prop.metadata_constraints.is_some());
                let metadata = prop.metadata_constraints.as_ref().unwrap();
                assert_eq!(metadata.len(), 2);
                assert_eq!(metadata[0].key, "confidence");
                assert_eq!(metadata[1].key, "source");
            }
            _ => panic!("Expected proposition clause"),
        }
    }

    #[test]
    fn test_parse_nested_proposition() {
        let input = r#"
            FIND(?paper_doi, ?drug_name)
            WHERE {
                PROP(?zhangsan, "stated", (?paper, "cites_as_evidence", (?drug, "treats", ?symptom)))
                ATTR(?paper, "doi", ?paper_doi)
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Proposition(prop) => match &prop.object {
                PropTerm::NestedProp(nested) => {
                    assert_eq!(
                        nested.predicate,
                        PredTerm::Literal("cites_as_evidence".to_string())
                    );
                    match &nested.object {
                        PropTerm::NestedProp(inner_nested) => {
                            assert_eq!(
                                inner_nested.predicate,
                                PredTerm::Literal("treats".to_string())
                            );
                        }
                        _ => panic!("Expected nested proposition"),
                    }
                }
                _ => panic!("Expected nested proposition object"),
            },
            _ => panic!("Expected proposition clause"),
        }
    }

    #[test]
    fn test_parse_attribute_pattern() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[1] {
            WhereClause::Attribute(attr) => {
                assert_eq!(attr.node_variable, "drug");
                assert_eq!(attr.attribute_name, "name");
                assert_eq!(attr.value_variable, "drug_name");
            }
            _ => panic!("Expected attribute clause"),
        }
    }

    #[test]
    fn test_parse_simple_comparison_filter() {
        let input = "FILTER(?risk < 3)";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Comparison {
                left,
                operator,
                right,
            } => {
                match left {
                    FilterOperand::Variable(var) => assert_eq!(var, "risk"),
                    _ => panic!("Expected variable operand"),
                }
                assert_eq!(operator, ComparisonOperator::LessThan);
                match right {
                    FilterOperand::Literal(Value::Number(_)) => {}
                    _ => panic!("Expected number literal"),
                }
            }
            _ => panic!("Expected comparison expression"),
        }
        assert!(filter.subquery.is_none());
    }

    #[test]
    fn test_parse_all_comparison_operators() {
        let test_cases = vec![
            ("FILTER(?a == ?b)", ComparisonOperator::Equal),
            ("FILTER(?a != ?b)", ComparisonOperator::NotEqual),
            ("FILTER(?a < ?b)", ComparisonOperator::LessThan),
            ("FILTER(?a > ?b)", ComparisonOperator::GreaterThan),
            ("FILTER(?a <= ?b)", ComparisonOperator::LessEqual),
            ("FILTER(?a >= ?b)", ComparisonOperator::GreaterEqual),
        ];

        for (input, expected_op) in test_cases {
            let result = parse_filter_clause(input);
            assert!(result.is_ok(), "Failed to parse: {input}");

            let (_, filter) = result.unwrap();
            match filter.expression {
                FilterExpression::Comparison { operator, .. } => {
                    assert_eq!(operator, expected_op);
                }
                _ => panic!("Expected comparison expression for: {input}"),
            }
        }
    }

    #[test]
    fn test_parse_logical_and_filter() {
        let input = "FILTER(?risk < 3 && ?score > 0.5)";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical {
                left,
                operator,
                right,
            } => {
                assert_eq!(operator, LogicalOperator::And);
                match left.as_ref() {
                    FilterExpression::Comparison { operator, .. } => {
                        assert_eq!(*operator, ComparisonOperator::LessThan);
                    }
                    _ => panic!("Expected comparison in left side"),
                }
                match right.as_ref() {
                    FilterExpression::Comparison { operator, .. } => {
                        assert_eq!(*operator, ComparisonOperator::GreaterThan);
                    }
                    _ => panic!("Expected comparison in right side"),
                }
            }
            _ => panic!("Expected logical expression"),
        }
    }

    #[test]
    fn test_parse_logical_or_filter() {
        let input = "FILTER(?type == \"Drug\" || ?type == \"Medicine\")";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical { operator, .. } => {
                assert_eq!(operator, LogicalOperator::Or);
            }
            _ => panic!("Expected logical OR expression"),
        }
    }

    #[test]
    fn test_parse_not_filter() {
        let input = "FILTER(!(?risk > 5))";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Not(inner) => match inner.as_ref() {
                FilterExpression::Comparison { operator, .. } => {
                    assert_eq!(*operator, ComparisonOperator::GreaterThan);
                }
                _ => panic!("Expected comparison inside NOT"),
            },
            _ => panic!("Expected NOT expression"),
        }
    }

    #[test]
    fn test_parse_function_filter() {
        let test_cases = vec![
            (
                "FILTER(CONTAINS(?name, \"acid\"))",
                FilterFunction::Contains,
            ),
            (
                "FILTER(STARTS_WITH(?name, \"pre\"))",
                FilterFunction::StartsWith,
            ),
            (
                "FILTER(ENDS_WITH(?name, \"ine\"))",
                FilterFunction::EndsWith,
            ),
            ("FILTER(REGEX(?name, \"[A-Z]+\"))", FilterFunction::Regex),
        ];

        for (input, expected_func) in test_cases {
            let result = parse_filter_clause(input);
            assert!(result.is_ok(), "Failed to parse: {input}");

            let (_, filter) = result.unwrap();
            match filter.expression {
                FilterExpression::Function { func, args } => {
                    assert_eq!(func, expected_func);
                    assert_eq!(args.len(), 2);
                    match &args[0] {
                        FilterOperand::Variable(_) => {}
                        _ => panic!("Expected variable as first argument"),
                    }
                    match &args[1] {
                        FilterOperand::Literal(Value::String(_)) => {}
                        _ => panic!("Expected string literal as second argument"),
                    }
                }
                _ => panic!("Expected function expression for: {input}"),
            }
        }
    }

    #[test]
    fn test_parse_complex_logical_filter() {
        let input = "FILTER(?risk < 3 && (CONTAINS(?name, \"acid\") || ?score > 0.8))";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical {
                left,
                operator,
                right,
            } => {
                assert_eq!(operator, LogicalOperator::And);
                // Left side should be a comparison
                match left.as_ref() {
                    FilterExpression::Comparison { .. } => {}
                    _ => panic!("Expected comparison on left side"),
                }
                // Right side should be a logical OR
                match right.as_ref() {
                    FilterExpression::Logical { operator, .. } => {
                        assert_eq!(*operator, LogicalOperator::Or);
                    }
                    _ => panic!("Expected logical OR on right side"),
                }
            }
            _ => panic!("Expected logical AND expression"),
        }
    }

    #[test]
    fn test_parse_filter_with_different_operand_types() {
        let input = "FILTER(?active == true && ?count != null && ?score >= 3.14)";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        // This tests that we can parse different value types (boolean, null, float)
        match filter.expression {
            FilterExpression::Logical { .. } => {}
            _ => panic!("Expected logical expression"),
        }
    }

    #[test]
    fn test_parse_filter_with_subquery() {
        let input = r#"FILTER(?symptom_count > 3) {
            SELECT(COUNT(?symptom) AS ?symptom_count)
            WHERE { PROP(?drug, "treats", ?symptom) }
        }"#;
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Comparison { .. } => {}
            _ => panic!("Expected comparison expression"),
        }
        assert!(filter.subquery.is_some());

        let subquery = filter.subquery.unwrap();
        assert_eq!(subquery.select_clause.expressions.len(), 1);
        match &subquery.select_clause.expressions[0] {
            FindExpression::Aggregation { func, alias, .. } => {
                assert_eq!(*func, AggregationFunction::Count);
                assert_eq!(alias, "symptom_count");
            }
            _ => panic!("Expected aggregation in subquery"),
        }
        assert_eq!(subquery.where_clauses.len(), 1);
    }

    #[test]
    fn test_parse_operator_precedence() {
        // Test that AND has higher precedence than OR
        let input = "FILTER(?a == 1 || ?b == 2 && ?c == 3)";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical {
                operator,
                left,
                right,
            } => {
                // Should be parsed as: (?a == 1) || (?b == 2 && ?c == 3)
                assert_eq!(operator, LogicalOperator::Or);
                match left.as_ref() {
                    FilterExpression::Comparison { .. } => {}
                    _ => panic!("Expected comparison on left"),
                }
                match right.as_ref() {
                    FilterExpression::Logical { operator, .. } => {
                        assert_eq!(*operator, LogicalOperator::And);
                    }
                    _ => panic!("Expected AND expression on right"),
                }
            }
            _ => panic!("Expected OR expression at top level"),
        }
    }

    #[test]
    fn test_parse_parentheses_override_precedence() {
        // Test that parentheses can override operator precedence
        let input = "FILTER((?a == 1 || ?b == 2) && ?c == 3)";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical {
                operator,
                left,
                right,
            } => {
                // Should be parsed as: (?a == 1 || ?b == 2) && (?c == 3)
                assert_eq!(operator, LogicalOperator::And);
                match left.as_ref() {
                    FilterExpression::Logical { operator, .. } => {
                        assert_eq!(*operator, LogicalOperator::Or);
                    }
                    _ => panic!("Expected OR expression on left"),
                }
                match right.as_ref() {
                    FilterExpression::Comparison { .. } => {}
                    _ => panic!("Expected comparison on right"),
                }
            }
            _ => panic!("Expected AND expression at top level"),
        }
    }

    #[test]
    fn test_parse_filter_error_cases() {
        let invalid_inputs = vec![
            "FILTER()",                 // Empty filter
            "FILTER(?a <)",             // Incomplete comparison
            "FILTER(?a == && ?b)",      // Invalid logical expression
            "FILTER(UNKNOWN_FUNC(?a))", // Unknown function
            "FILTER(?a ===== ?b)",      // Invalid operator
            "FILTER(!)",                // NOT without expression
        ];

        for input in invalid_inputs {
            let result = parse_filter_clause(input);
            assert!(result.is_err(), "Should fail to parse: {input}");
        }
    }

    #[test]
    fn test_parse_filter_whitespace_handling() {
        let input = "FILTER  (  ?risk   <   3   &&   CONTAINS  (  ?name  ,  \"acid\"  )  )";
        let result = parse_filter_clause(input);
        assert!(result.is_ok());

        let (_, filter) = result.unwrap();
        match filter.expression {
            FilterExpression::Logical { operator, .. } => {
                assert_eq!(operator, LogicalOperator::And);
            }
            _ => panic!("Expected logical expression"),
        }
    }

    #[test]
    fn test_parse_filter_clause() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "risk_level", ?risk)
                FILTER(?risk < 3)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[2] {
            WhereClause::Filter(filter) => match &filter.expression {
                FilterExpression::Comparison {
                    left,
                    operator,
                    right,
                } => {
                    match left {
                        FilterOperand::Variable(var) => assert_eq!(var, "risk"),
                        _ => panic!("Expected variable operand"),
                    }
                    assert_eq!(*operator, ComparisonOperator::LessThan);
                    match right {
                        FilterOperand::Literal(Value::Number(_)) => {}
                        _ => panic!("Expected number literal"),
                    }
                }
                _ => panic!("Expected comparison expression"),
            },
            _ => panic!("Expected filter clause"),
        }
    }

    #[test]
    fn test_parse_complex_filter() {
        let input = r#"
        FIND(?drug_name)
        WHERE {
            ?drug(type: "Drug")
            ATTR(?drug, "name", ?drug_name)
            FILTER(CONTAINS(?drug_name, "acid") && ?risk < 3)
        }
    "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[2] {
            WhereClause::Filter(filter) => {
                // 检查这是一个逻辑 AND 表达式
                match &filter.expression {
                    FilterExpression::Logical {
                        left,
                        operator,
                        right,
                    } => {
                        assert_eq!(*operator, LogicalOperator::And);

                        // 左边应该是 CONTAINS 函数
                        match left.as_ref() {
                            FilterExpression::Function { func, args } => {
                                assert_eq!(*func, FilterFunction::Contains);
                                assert_eq!(args.len(), 2);
                                match &args[0] {
                                    FilterOperand::Variable(var) => assert_eq!(var, "drug_name"),
                                    _ => panic!("Expected variable as first argument"),
                                }
                                match &args[1] {
                                    FilterOperand::Literal(Value::String(s)) => {
                                        assert_eq!(s, "acid")
                                    }
                                    _ => panic!("Expected string literal as second argument"),
                                }
                            }
                            _ => panic!("Expected function expression on left side"),
                        }

                        // 右边应该是比较表达式
                        match right.as_ref() {
                            FilterExpression::Comparison {
                                left,
                                operator,
                                right,
                            } => {
                                match left {
                                    FilterOperand::Variable(var) => assert_eq!(var, "risk"),
                                    _ => panic!("Expected variable operand"),
                                }
                                assert_eq!(*operator, ComparisonOperator::LessThan);
                                match right {
                                    FilterOperand::Literal(Value::Number(_)) => {}
                                    _ => panic!("Expected number literal"),
                                }
                            }
                            _ => panic!("Expected comparison expression on right side"),
                        }
                    }
                    _ => panic!("Expected logical AND expression"),
                }

                // 确保没有子查询
                assert!(filter.subquery.is_none());
            }
            _ => panic!("Expected filter clause"),
        }
    }

    #[test]
    fn test_parse_optional_clause() {
        let input = r#"
            FIND(?drug_name, ?side_effect_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
                OPTIONAL {
                    PROP(?drug, "has_side_effect", ?side_effect)
                    ATTR(?side_effect, "name", ?side_effect_name)
                }
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[2] {
            WhereClause::Optional(clauses) => {
                assert_eq!(clauses.len(), 2);
                match &clauses[0] {
                    WhereClause::Proposition(prop) => {
                        assert_eq!(
                            prop.predicate,
                            PredTerm::Literal("has_side_effect".to_string())
                        );
                    }
                    _ => panic!("Expected proposition in optional"),
                }
            }
            _ => panic!("Expected optional clause"),
        }
    }

    #[test]
    fn test_parse_not_clause() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                NOT {
                    ?nsaid_class(name: "NSAID")
                    PROP(?drug, "is_class_of", ?nsaid_class)
                }
                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[1] {
            WhereClause::Not(clauses) => {
                assert_eq!(clauses.len(), 2);
            }
            _ => panic!("Expected NOT clause"),
        }
    }

    #[test]
    fn test_parse_union_clause() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?headache(name: "Headache")
                PROP(?drug, "treats", ?headache)

                UNION {
                    ?fever(name: "Fever")
                    PROP(?drug, "treats", ?fever)
                }

                ATTR(?drug, "name", ?drug_name)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.where_clauses.len(), 4);
        match &query.where_clauses[2] {
            WhereClause::Union(clauses) => {
                assert_eq!(clauses.len(), 2);
            }
            _ => panic!("Expected UNION clause"),
        }
    }

    #[test]
    fn test_parse_order_by_clause() {
        let input = r#"
            FIND(?drug_name, ?risk)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
                ATTR(?drug, "risk_level", ?risk)
            }
            ORDER BY ?risk ASC, ?drug_name DESC
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert!(query.order_by.is_some());
        let order_by = query.order_by.unwrap();
        assert_eq!(order_by.len(), 2);
        assert_eq!(order_by[0].variable, "risk");
        assert_eq!(order_by[0].direction, OrderDirection::Asc);
        assert_eq!(order_by[1].variable, "drug_name");
        assert_eq!(order_by[1].direction, OrderDirection::Desc);
    }

    #[test]
    fn test_parse_order_by_default_asc() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
            }
            ORDER BY ?drug_name
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert!(query.order_by.is_some());
        let order_by = query.order_by.unwrap();
        assert_eq!(order_by.len(), 1);
        assert_eq!(order_by[0].direction, OrderDirection::Asc);
    }

    #[test]
    fn test_parse_limit_and_offset() {
        let input = r#"
            FIND(?drug_name)
            WHERE {
                ?drug(type: "Drug")
                ATTR(?drug, "name", ?drug_name)
            }
            ORDER BY ?drug_name
            LIMIT 20
            OFFSET 10
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.limit, Some(20));
        assert_eq!(query.offset, Some(10));
    }

    #[test]
    fn test_parse_complex_query() {
        let input = r#"
            FIND(?drug_name, ?risk)
            WHERE {
                ?drug(type: "Drug")
                ?headache(name: "Headache")
                ?nsaid_class(name: "NSAID")

                PROP(?drug, "treats", ?headache)

                NOT {
                    PROP(?drug, "is_class_of", ?nsaid_class)
                }

                ATTR(?drug, "name", ?drug_name)
                ATTR(?drug, "risk_level", ?risk)
                FILTER(?risk < 4)
            }
            ORDER BY ?risk ASC
            LIMIT 20
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.find_clause.expressions.len(), 2);
        assert_eq!(query.where_clauses.len(), 8); // 3 groundings + 1 prop + 1 not + 2 attrs + 1 filter
        assert!(query.order_by.is_some());
        assert_eq!(query.limit, Some(20));
        assert!(query.offset.is_none());
    }

    #[test]
    fn test_parse_aggregation_with_grouping() {
        let input = r#"
            FIND(?class_name, COLLECT(?drug_name) AS ?drug_list)
            WHERE {
                ?class(type: "DrugClass")
                ATTR(?class, "name", ?class_name)

                ?drug(type: "Drug")
                PROP(?drug, "is_class_of", ?class)
                ATTR(?drug, "name", ?drug_name)
            }
            ORDER BY ?class_name
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.find_clause.expressions.len(), 2);
        match &query.find_clause.expressions[1] {
            FindExpression::Aggregation {
                func, var, alias, ..
            } => {
                assert_eq!(*func, AggregationFunction::Collect);
                assert_eq!(var, "drug_name");
                assert_eq!(alias, "drug_list");
            }
            _ => panic!("Expected aggregation expression"),
        }
    }

    #[test]
    fn test_parse_error_cases() {
        let invalid_inputs = vec![
            "FIND() WHERE {}",                   // Empty FIND
            "FIND(?var WHERE {}",                // Missing closing parenthesis
            "FIND(?var) WHERE",                  // Missing WHERE block
            "FIND(?var) WHERE { ?var }",         // Invalid grounding syntax
            "FIND(?var) WHERE { PROP(?a, ?b) }", // Missing object in PROP
            "FIND(?var) WHERE { ATTR(?a, ?b) }", // Missing value variable in ATTR
        ];

        for input in invalid_inputs {
            let result = parse_kql_query(input);
            assert!(result.is_err(), "Should fail to parse: {input}");
        }
    }

    #[test]
    fn test_parse_whitespace_handling() {
        let input_with_extra_whitespace = r#"
            FIND  (  ?drug_name  ,  ?symptom_name  )
            WHERE   {
                ?drug  (  type:  "Drug"  )
                PROP  (  ?drug  ,  "treats"  ,  ?symptom  )
                ATTR  (  ?drug  ,  "name"  ,  ?drug_name  )
                ATTR  (  ?symptom  ,  "name"  ,  ?symptom_name  )
            }
            ORDER   BY   ?drug_name   ASC
            LIMIT   10
        "#;

        let result = parse_kql_query(input_with_extra_whitespace);
        println!("Result: {result:?}");

        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        assert_eq!(query.find_clause.expressions.len(), 2);
        assert_eq!(query.where_clauses.len(), 4);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_parse_value_types() {
        let input = r#"
            FIND(?entity)
            WHERE {
                ?entity(type: "Entity", active: true, count: 42, score: 3.14, description: null)
            }
        "#;

        let result = parse_kql_query(input);
        assert!(result.is_ok());

        let (_, query) = result.unwrap();
        match &query.where_clauses[0] {
            WhereClause::Grounding(grounding) => {
                assert_eq!(grounding.constraints.len(), 5);

                // Check different value types
                match &grounding.constraints[1].value {
                    Value::Bool(true) => {}
                    _ => panic!("Expected boolean true"),
                }

                match &grounding.constraints[2].value {
                    Value::Number(_) => {}
                    _ => panic!("Expected number"),
                }

                match &grounding.constraints[4].value {
                    Value::Null => {}
                    _ => panic!("Expected null"),
                }
            }
            _ => panic!("Expected grounding clause"),
        }
    }
}
