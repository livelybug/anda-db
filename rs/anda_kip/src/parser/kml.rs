use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::{many1, separated_list1},
    sequence::{preceded, separated_pair, terminated},
};
use std::collections::HashMap;

use super::common::*;
use super::kql::parse_where_group;
use crate::ast::*;

// --- Top Level KML Parser ---

pub fn parse_kml_statement(input: &str) -> IResult<&str, KmlStatement> {
    alt((
        map(parse_upsert_block, KmlStatement::Upsert),
        map(parse_delete_statement, KmlStatement::Delete),
    ))
    .parse(input)
}

// --- UPSERT ---

fn parse_with_metadata(input: &str) -> IResult<&str, HashMap<String, Value>> {
    preceded(ws(tag("WITH METADATA")), key_value_map).parse(input)
}

fn parse_upsert_block(input: &str) -> IResult<&str, UpsertBlock> {
    map(
        preceded(
            ws(tag("UPSERT")),
            (
                braced_block(many1(ws(parse_upsert_item))),
                opt(parse_with_metadata),
            ),
        ),
        |(items, metadata)| UpsertBlock { items, metadata },
    )
    .parse(input)
}

fn parse_upsert_item(input: &str) -> IResult<&str, UpsertItem> {
    alt((
        map(parse_concept_block, UpsertItem::Concept),
        map(parse_proposition_block, UpsertItem::Proposition),
    ))
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

fn parse_concept_block(input: &str) -> IResult<&str, ConceptBlock> {
    map(
        (
            preceded(ws(tag("CONCEPT")), ws(local_handle)),
            braced_block((
                ws(on_clause),
                opt(ws(preceded(tag("SET ATTRIBUTES"), key_value_map))),
                opt(ws(preceded(
                    tag("SET PROPOSITIONS"),
                    braced_block(many1(ws(parse_set_proposition))),
                ))),
            )),
            opt(ws(parse_with_metadata)),
        ),
        |(handle, (on, set_attributes, set_propositions), metadata)| ConceptBlock {
            handle,
            on,
            set_attributes,
            set_propositions,
            metadata,
        },
    )
    .parse(input)
}

fn parse_set_proposition(input: &str) -> IResult<&str, SetProposition> {
    map(
        (
            preceded(
                ws(tag("PROP")),
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
        ),
        |((predicate, object), metadata)| SetProposition {
            predicate,
            object,
            metadata,
        },
    )
    .parse(input)
}

fn parse_proposition_block(input: &str) -> IResult<&str, PropositionBlock> {
    // This parser is a bit complex due to the nested structure
    // PROPOSITION @handle { ( ON {}, "pred", ON {}/@handle ) } WITH METADATA {}
    map(
        (
            preceded(ws(tag("PROPOSITION")), ws(local_handle)),
            braced_block(parenthesized_block((
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
        ),
        |(handle, (subject, _, predicate, _, object), metadata)| PropositionBlock {
            handle,
            subject,
            predicate,
            object,
            metadata,
        },
    )
    .parse(input)
}

// --- DELETE ---

fn parse_delete_statement(input: &str) -> IResult<&str, DeleteStatement> {
    preceded(
        ws(tag("DELETE ")),
        alt((
            parse_delete_attributes,
            parse_delete_concept,
            parse_delete_propositions_where,
            parse_delete_proposition,
        )),
    )
    .parse(input)
}

fn parse_delete_attributes(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag("ATTRIBUTES")),
            (
                braced_block(separated_list1(ws(char(',')), quoted_string)),
                preceded(ws(tag("FROM ")), ws(on_clause)),
            ),
        ),
        |(attributes, from)| DeleteStatement::DeleteAttributes { attributes, from },
    )
    .parse(input)
}

fn parse_delete_concept(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag("CONCEPT")),
            terminated(ws(on_clause), ws(tag("DETACH"))),
        ),
        |on| DeleteStatement::DeleteConcept { on },
    )
    .parse(input)
}

fn parse_delete_propositions_where(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag("PROPOSITIONS")),
            preceded(ws(tag("WHERE")), parse_where_group),
        ),
        DeleteStatement::DeletePropositionsWhere,
    )
    .parse(input)
}

fn parse_delete_proposition(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag("PROPOSITION ")),
            parenthesized_block((
                ws(on_clause),
                ws(char(',')),
                ws(quoted_string),
                ws(char(',')),
                ws(on_clause),
            )),
        ),
        |(subject, _, predicate, _, object)| DeleteStatement::DeleteProposition {
            subject,
            predicate,
            object,
        },
    )
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Number;

    #[test]
    fn test_parse_simple_upsert_concept() {
        let input = r#"
        UPSERT {
            CONCEPT @drug {
                ON { type: "Drug", name: "Aspirin" }
                SET ATTRIBUTES {
                    molecular_formula: "C9H8O4",
                    risk_level: 2
                }
            }
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert_eq!(upsert.items.len(), 1);
                assert!(upsert.metadata.is_none());

                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "drug");
                        assert_eq!(concept.on.keys.len(), 2);
                        assert_eq!(concept.on.keys[0].key, "type");
                        assert_eq!(concept.on.keys[0].value, Value::String("Drug".to_string()));
                        assert_eq!(concept.on.keys[1].key, "name");
                        assert_eq!(
                            concept.on.keys[1].value,
                            Value::String("Aspirin".to_string())
                        );

                        let attrs = concept.set_attributes.as_ref().unwrap();
                        assert_eq!(attrs.len(), 2);
                        assert_eq!(
                            attrs["molecular_formula"],
                            Value::String("C9H8O4".to_string())
                        );
                        assert_eq!(attrs["risk_level"], Value::Number(Number::from(2)));
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_upsert_with_propositions() {
        let input = r#"
        UPSERT {
            CONCEPT @cognizine {
                ON { type: "Drug", name: "Cognizine" }
                SET ATTRIBUTES {
                    molecular_formula: "C12H15N5O3",
                    risk_level: 2
                }
                SET PROPOSITIONS {
                    PROP("is_class_of", ON { type: "DrugClass", name: "Nootropic" })
                    PROP("treats", ON { type: "Symptom", name: "Brain Fog" })
                    PROP("has_side_effect", @neural_bloom) WITH METADATA {
                        confidence: 0.75,
                        source: "Clinical Trial"
                    }
                }
            }

            CONCEPT @neural_bloom {
                ON { type: "Symptom", name: "Neural Bloom" }
                SET ATTRIBUTES {
                    description: "A rare side effect"
                }
            }
        }
        WITH METADATA {
            source: "KnowledgeCapsule:Nootropics_v1.0",
            author: "LDC Labs",
            confidence: 0.95
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert_eq!(upsert.items.len(), 2);
                assert!(upsert.metadata.is_some());

                let metadata = upsert.metadata.as_ref().unwrap();
                assert_eq!(
                    metadata["source"],
                    Value::String("KnowledgeCapsule:Nootropics_v1.0".to_string())
                );
                assert_eq!(metadata["author"], Value::String("LDC Labs".to_string()));
                assert_eq!(
                    metadata["confidence"],
                    Value::Number(Number::from_f64(0.95).unwrap())
                );

                // Check first concept with propositions
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "cognizine");
                        let props = concept.set_propositions.as_ref().unwrap();
                        assert_eq!(props.len(), 3);

                        // Check first proposition
                        assert_eq!(props[0].predicate, "is_class_of");
                        match &props[0].object {
                            PropObject::Node(on_clause) => {
                                assert_eq!(on_clause.keys.len(), 2);
                                assert_eq!(on_clause.keys[0].key, "type");
                                assert_eq!(
                                    on_clause.keys[0].value,
                                    Value::String("DrugClass".to_string())
                                );
                            }
                            _ => panic!("Expected Node"),
                        }

                        // Check third proposition with metadata
                        assert_eq!(props[2].predicate, "has_side_effect");
                        match &props[2].object {
                            PropObject::LocalHandle(handle) => {
                                assert_eq!(handle, "neural_bloom");
                            }
                            _ => panic!("Expected LocalHandle"),
                        }
                        let prop_metadata = props[2].metadata.as_ref().unwrap();
                        assert_eq!(
                            prop_metadata["confidence"],
                            Value::Number(Number::from_f64(0.75).unwrap())
                        );
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_proposition_block() {
        let input = r#"
        UPSERT {
            PROPOSITION @stmt {
                ( ON { name: "Zhang San" }, "stated", ON { type: "Paper", doi: "10.1000/xyz" } )
            }
            WITH METADATA {
                confidence: 0.9
            }
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert_eq!(upsert.items.len(), 1);

                match &upsert.items[0] {
                    UpsertItem::Proposition(prop) => {
                        assert_eq!(prop.handle, "stmt");
                        assert_eq!(prop.predicate, "stated");

                        // Check subject
                        assert_eq!(prop.subject.keys.len(), 1);
                        assert_eq!(prop.subject.keys[0].key, "name");
                        assert_eq!(
                            prop.subject.keys[0].value,
                            Value::String("Zhang San".to_string())
                        );

                        // Check object
                        match &prop.object {
                            PropObject::Node(on_clause) => {
                                assert_eq!(on_clause.keys.len(), 2);
                                assert_eq!(on_clause.keys[0].key, "type");
                                assert_eq!(
                                    on_clause.keys[0].value,
                                    Value::String("Paper".to_string())
                                );
                                assert_eq!(on_clause.keys[1].key, "doi");
                                assert_eq!(
                                    on_clause.keys[1].value,
                                    Value::String("10.1000/xyz".to_string())
                                );
                            }
                            _ => panic!("Expected Node"),
                        }

                        // Check metadata
                        let metadata = prop.metadata.as_ref().unwrap();
                        assert_eq!(
                            metadata["confidence"],
                            Value::Number(Number::from_f64(0.9).unwrap())
                        );
                    }
                    _ => panic!("Expected PropositionBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_delete_attributes() {
        let input = r#"
        DELETE ATTRIBUTES { "risk_category", "old_name" }
        FROM ON { type: "Drug", name: "Aspirin" }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeleteAttributes { attributes, from }) => {
                assert_eq!(attributes.len(), 2);
                assert_eq!(attributes[0], "risk_category");
                assert_eq!(attributes[1], "old_name");

                assert_eq!(from.keys.len(), 2);
                assert_eq!(from.keys[0].key, "type");
                assert_eq!(from.keys[0].value, Value::String("Drug".to_string()));
                assert_eq!(from.keys[1].key, "name");
                assert_eq!(from.keys[1].value, Value::String("Aspirin".to_string()));
            }
            _ => panic!("Expected DeleteAttributes"),
        }
    }

    #[test]
    fn test_parse_delete_proposition() {
        let input = r#"
        DELETE PROPOSITION (
            ON { type: "Drug", name: "Cognizine" },
            "treats",
            ON { type: "Symptom", name: "Brain Fog" }
        )
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeleteProposition {
                subject,
                predicate,
                object,
            }) => {
                assert_eq!(predicate, "treats");

                // Check subject
                assert_eq!(subject.keys.len(), 2);
                assert_eq!(subject.keys[0].key, "type");
                assert_eq!(subject.keys[0].value, Value::String("Drug".to_string()));
                assert_eq!(subject.keys[1].key, "name");
                assert_eq!(
                    subject.keys[1].value,
                    Value::String("Cognizine".to_string())
                );

                // Check object
                assert_eq!(object.keys.len(), 2);
                assert_eq!(object.keys[0].key, "type");
                assert_eq!(object.keys[0].value, Value::String("Symptom".to_string()));
                assert_eq!(object.keys[1].key, "name");
                assert_eq!(object.keys[1].value, Value::String("Brain Fog".to_string()));
            }
            _ => panic!("Expected DeleteProposition"),
        }
    }

    #[test]
    fn test_parse_delete_propositions_where() {
        let input = r#"
        DELETE PROPOSITIONS
        WHERE {
            PROP(?s, ?p, ?o) { source: "untrusted_source" }
        }
        "#;

        let result = parse_kml_statement(input);
        // println!("Result: {:?}", result);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeletePropositionsWhere(where_clauses)) => {
                assert_eq!(where_clauses.len(), 1);
                // The actual WHERE clause parsing is tested in KQL tests
            }
            _ => panic!("Expected DeletePropositionsWhere"),
        }
    }

    #[test]
    fn test_parse_delete_concept() {
        let input = r#"
        DELETE CONCEPT
        ON { type: "Drug", name: "OutdatedDrug" }
        DETACH
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeleteConcept { on }) => {
                assert_eq!(on.keys.len(), 2);
                assert_eq!(on.keys[0].key, "type");
                assert_eq!(on.keys[0].value, Value::String("Drug".to_string()));
                assert_eq!(on.keys[1].key, "name");
                assert_eq!(on.keys[1].value, Value::String("OutdatedDrug".to_string()));
            }
            _ => panic!("Expected DeleteConcept"),
        }
    }

    #[test]
    fn test_parse_complex_upsert_with_mixed_items() {
        let input = r#"
        UPSERT {
            CONCEPT @drug {
                ON { id: "drug_001" }
                SET ATTRIBUTES {
                    name: "TestDrug",
                    active: true,
                    dosage: null
                }
            }

            PROPOSITION @relation {
                ( ON { id: "drug_001" }, "interacts_with", ON { id: "drug_002" } )
            }
            WITH METADATA {
                interaction_type: "synergistic"
            }

            CONCEPT @target {
                ON { id: "drug_002" }
                SET PROPOSITIONS {
                    PROP("belongs_to", @relation)
                }
            }
        }
        WITH METADATA {
            batch_id: "batch_123",
            timestamp: "2024-01-01T00:00:00Z"
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert_eq!(upsert.items.len(), 3);
                assert!(upsert.metadata.is_some());

                // Check global metadata
                let global_metadata = upsert.metadata.as_ref().unwrap();
                assert_eq!(
                    global_metadata["batch_id"],
                    Value::String("batch_123".to_string())
                );
                assert_eq!(
                    global_metadata["timestamp"],
                    Value::String("2024-01-01T00:00:00Z".to_string())
                );

                // Check first item (concept)
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "drug");
                        let attrs = concept.set_attributes.as_ref().unwrap();
                        assert_eq!(attrs["name"], Value::String("TestDrug".to_string()));
                        assert_eq!(attrs["active"], Value::Bool(true));
                        assert_eq!(attrs["dosage"], Value::Null);
                    }
                    _ => panic!("Expected ConceptBlock"),
                }

                // Check second item (proposition)
                match &upsert.items[1] {
                    UpsertItem::Proposition(prop) => {
                        assert_eq!(prop.handle, "relation");
                        assert_eq!(prop.predicate, "interacts_with");
                        let metadata = prop.metadata.as_ref().unwrap();
                        assert_eq!(
                            metadata["interaction_type"],
                            Value::String("synergistic".to_string())
                        );
                    }
                    _ => panic!("Expected PropositionBlock"),
                }

                // Check third item (concept with proposition referencing local handle)
                match &upsert.items[2] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "target");
                        let props = concept.set_propositions.as_ref().unwrap();
                        assert_eq!(props.len(), 1);
                        assert_eq!(props[0].predicate, "belongs_to");
                        match &props[0].object {
                            PropObject::LocalHandle(handle) => {
                                assert_eq!(handle, "relation");
                            }
                            _ => panic!("Expected LocalHandle"),
                        }
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_minimal_concept() {
        let input = r#"
        UPSERT {
            CONCEPT @minimal {
                ON { id: "test_001" }
            }
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert_eq!(upsert.items.len(), 1);
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "minimal");
                        assert!(concept.set_attributes.is_none());
                        assert!(concept.set_propositions.is_none());
                        assert!(concept.metadata.is_none());
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_error_cases() {
        // Missing DETACH in DELETE CONCEPT
        let input1 = r#"
        DELETE CONCEPT
        ON { type: "Drug", name: "Test" }
        "#;
        assert!(parse_kml_statement(input1).is_err());

        // Invalid local handle (missing @)
        let input2 = r#"
        UPSERT {
            CONCEPT drug {
                ON { id: "test" }
            }
        }
        "#;
        assert!(parse_kml_statement(input2).is_err());

        // Missing ON clause in concept
        let input3 = r#"
        UPSERT {
            CONCEPT @drug {
                SET ATTRIBUTES { name: "Test" }
            }
        }
        "#;
        assert!(parse_kml_statement(input3).is_err());
    }

    #[test]
    fn test_parse_with_metadata_variations() {
        // Test concept with metadata but no global metadata
        let input = r#"
        UPSERT {
            CONCEPT @drug {
                ON { id: "test" }
                SET ATTRIBUTES { name: "Test" }
            }
            WITH METADATA {
                source: "test_source",
                confidence: 1.0
            }
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Upsert(upsert) => {
                assert!(upsert.metadata.is_none());
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        let metadata = concept.metadata.as_ref().unwrap();
                        assert_eq!(metadata["source"], Value::String("test_source".to_string()));
                        assert_eq!(
                            metadata["confidence"],
                            Value::Number(Number::from_f64(1.0).unwrap())
                        );
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }
}
