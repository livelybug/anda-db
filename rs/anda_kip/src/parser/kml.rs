use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    multi::{many1, separated_list1},
    sequence::{preceded, separated_pair, terminated},
};

use super::common::*;
use super::kql::{parse_concept_matcher, parse_prop_mather, parse_target_term, parse_where_block};
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

fn parse_with_metadata(input: &str) -> IResult<&str, Map<String, Json>> {
    preceded(ws(tag("WITH METADATA")), json_value_map).parse(input)
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

fn parse_concept_block(input: &str) -> IResult<&str, ConceptBlock> {
    map(
        (
            preceded(ws(tag("CONCEPT")), ws(variable)),
            braced_block((
                ws(parse_concept_matcher),
                opt(ws(preceded(tag("SET ATTRIBUTES"), json_value_map))),
                opt(ws(preceded(
                    tag("SET PROPOSITIONS"),
                    braced_block(many1(ws(parse_set_proposition))),
                ))),
            )),
            opt(ws(parse_with_metadata)),
        ),
        |(handle, (concept, set_attributes, set_propositions), metadata)| ConceptBlock {
            handle,
            concept,
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
            parenthesized_block(separated_pair(
                quoted_string,
                ws(char(',')),
                parse_target_term,
            )),
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
    map(
        (
            preceded(ws(tag("PROPOSITION")), ws(variable)),
            braced_block((
                ws(parse_prop_mather),
                opt(ws(preceded(tag("SET ATTRIBUTES"), json_value_map))),
            )),
            opt(ws(parse_with_metadata)),
        ),
        |(handle, (proposition, set_attributes), metadata)| PropositionBlock {
            handle,
            proposition,
            set_attributes,
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
            parse_delete_propositions,
            parse_delete_concept,
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
                parse_where_block,
            ),
        ),
        |(attributes, where_clauses)| DeleteStatement::DeleteAttributes {
            attributes,
            where_clauses,
        },
    )
    .parse(input)
}

fn parse_delete_propositions(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(ws(tag("PROPOSITIONS")), parse_where_block),
        DeleteStatement::DeletePropositions,
    )
    .parse(input)
}

fn parse_delete_concept(input: &str) -> IResult<&str, DeleteStatement> {
    map(
        preceded(
            ws(tag("CONCEPT")),
            terminated(parse_concept_matcher, ws(tag("DETACH"))),
        ),
        DeleteStatement::DeleteConcept,
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
            CONCEPT ?drug {
                { type: "Drug", name: "Aspirin" }
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
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: None,
                                r#type: Some("Drug".to_string()),
                                name: Some("Aspirin".to_string()),
                            }
                        );

                        let attrs = concept.set_attributes.as_ref().unwrap();
                        assert_eq!(attrs.len(), 2);
                        assert_eq!(
                            attrs["molecular_formula"],
                            Json::String("C9H8O4".to_string())
                        );
                        assert_eq!(attrs["risk_level"], Json::Number(Number::from(2)));
                    }
                    _ => panic!("Expected ConceptBlock"),
                }
            }
            _ => panic!("Expected UpsertBlock"),
        }
    }

    #[test]
    fn test_parse_simple_upsert_concept_with_metadata() {
        let input = r#"
        UPSERT {
            CONCEPT ?drug {
                {type: "Drug", name: "TestDrug"}
            }
            WITH METADATA {
                "confidence":0.95,
                "source":"clinical_trial"
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
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: None,
                                r#type: Some("Drug".to_string()),
                                name: Some("TestDrug".to_string()),
                            }
                        );

                        let metadata = concept.metadata.as_ref().unwrap();
                        assert_eq!(
                            metadata["confidence"],
                            Json::Number(Number::from_f64(0.95).unwrap())
                        );
                        assert_eq!(
                            metadata["source"],
                            Json::String("clinical_trial".to_string())
                        );
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
            CONCEPT ?cognizine {
                { type: "Drug", name: "Cognizine" }
                SET ATTRIBUTES {
                    molecular_formula: "C12H15N5O3",
                    risk_level: 2
                }
                SET PROPOSITIONS {
                    ("is_class_of", { type: "DrugClass", name: "Nootropic" })
                    ("treats", { type: "Symptom", name: "Brain Fog" })
                    ("has_side_effect", ?neural_bloom) WITH METADATA {
                        confidence: 0.75,
                        source: "Clinical Trial"
                    }
                }
            }

            CONCEPT ?neural_bloom {
                { type: "Symptom", name: "Neural Bloom" }
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
                    Json::String("KnowledgeCapsule:Nootropics_v1.0".to_string())
                );
                assert_eq!(metadata["author"], Json::String("LDC Labs".to_string()));
                assert_eq!(
                    metadata["confidence"],
                    Json::Number(Number::from_f64(0.95).unwrap())
                );

                // Check first concept with propositions
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "cognizine");
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: None,
                                r#type: Some("Drug".to_string()),
                                name: Some("Cognizine".to_string()),
                            }
                        );
                        let props = concept.set_propositions.as_ref().unwrap();
                        assert_eq!(props.len(), 3);

                        assert_eq!(props[0].predicate, "is_class_of");
                        assert_eq!(
                            props[0].object,
                            TargetTerm::Concept(ConceptMatcher {
                                id: None,
                                r#type: Some("DrugClass".to_string()),
                                name: Some("Nootropic".to_string()),
                            })
                        );

                        assert_eq!(props[1].predicate, "treats");
                        assert_eq!(
                            props[1].object,
                            TargetTerm::Concept(ConceptMatcher {
                                id: None,
                                r#type: Some("Symptom".to_string()),
                                name: Some("Brain Fog".to_string()),
                            })
                        );

                        assert_eq!(props[2].predicate, "has_side_effect");
                        assert_eq!(
                            props[2].object,
                            TargetTerm::Variable("neural_bloom".to_string()),
                        );

                        let prop_metadata = props[2].metadata.as_ref().unwrap();
                        assert_eq!(
                            prop_metadata["confidence"],
                            Json::Number(Number::from_f64(0.75).unwrap())
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
            PROPOSITION ?stmt {
                ( { name: "Zhang San" }, "stated", { type: "Paper", name: "paper_doi" } )
                SET ATTRIBUTES {
                    doi: "10.1000/xyz",
                    created_at: "2023-11-10T14:20:10Z"
                }
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
                        assert_eq!(
                            prop.proposition,
                            PropositionMatcher {
                                subject: TargetTerm::Concept(ConceptMatcher {
                                    id: None,
                                    r#type: None,
                                    name: Some("Zhang San".to_string()),
                                }),
                                predicate: PredTerm::Literal("stated".to_string()),
                                object: TargetTerm::Concept(ConceptMatcher {
                                    id: None,
                                    r#type: Some("Paper".to_string()),
                                    name: Some("paper_doi".to_string()),
                                }),
                            }
                        );

                        let set_attributes = prop.set_attributes.as_ref().unwrap();
                        assert_eq!(set_attributes.len(), 2);
                        assert_eq!(set_attributes["doi"], "10.1000/xyz");
                        assert_eq!(set_attributes["created_at"], "2023-11-10T14:20:10Z");

                        // Check metadata
                        let metadata = prop.metadata.as_ref().unwrap();
                        assert_eq!(metadata.len(), 1);
                        assert_eq!(
                            metadata["confidence"],
                            Json::Number(Number::from_f64(0.9).unwrap())
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
        WHERE {{ type: "Drug", name: "Aspirin" }}
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeleteAttributes {
                attributes,
                where_clauses,
            }) => {
                assert_eq!(attributes.len(), 2);
                assert_eq!(attributes[0], "risk_category");
                assert_eq!(attributes[1], "old_name");

                assert_eq!(where_clauses.len(), 1);
                assert_eq!(
                    where_clauses[0],
                    WhereClause::Concept(ConceptClause {
                        matcher: ConceptMatcher {
                            id: None,
                            r#type: Some("Drug".to_string()),
                            name: Some("Aspirin".to_string()),
                        },
                        variable: None,
                    })
                );
            }
            _ => panic!("Expected DeleteAttributes"),
        }
    }

    #[test]
    fn test_parse_delete_propositions_where() {
        let input = r#"
        DELETE PROPOSITIONS
        WHERE {
            META((?s, ?p, ?o), "source", ?source)
            FILTER(?source == "untrusted_source")
        }
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeletePropositions(where_clauses)) => {
                assert_eq!(where_clauses.len(), 2);
                assert_eq!(
                    where_clauses[0],
                    WhereClause::Metadata(MetadataClause {
                        target: TargetTerm::Proposition(Box::new(PropositionMatcher {
                            subject: TargetTerm::Variable("s".to_string()),
                            predicate: PredTerm::Variable("p".to_string()),
                            object: TargetTerm::Variable("o".to_string()),
                        })),
                        key: "source".to_string(),
                        variable: "source".to_string(),
                    })
                );
            }
            _ => panic!("Expected DeletePropositionsWhere"),
        }
    }

    #[test]
    fn test_parse_delete_concept() {
        let input = r#"
        DELETE CONCEPT
        { type: "Drug", name: "OutdatedDrug" }
        DETACH
        "#;

        let result = parse_kml_statement(input);
        assert!(result.is_ok());

        let (_, statement) = result.unwrap();
        match statement {
            KmlStatement::Delete(DeleteStatement::DeleteConcept(concept)) => {
                assert_eq!(
                    concept,
                    ConceptMatcher {
                        id: None,
                        r#type: Some("Drug".to_string()),
                        name: Some("OutdatedDrug".to_string()),
                    }
                );
            }
            _ => panic!("Expected DeleteConcept"),
        }
    }

    #[test]
    fn test_parse_complex_upsert_with_mixed_items() {
        let input = r#"
        UPSERT {
            CONCEPT ?drug {
                { id: "drug_001" }
                SET ATTRIBUTES {
                    name: "TestDrug",
                    active: true,
                    dosage: null
                }
            }

            PROPOSITION ?relation {
                ( { id: "drug_001" }, "interacts_with", { id: "drug_002" } )
            }
            WITH METADATA {
                interaction_type: "synergistic"
            }

            CONCEPT ?target {
                { id: "drug_002" }
                SET PROPOSITIONS {
                    ("belongs_to", ?relation)
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
                    Json::String("batch_123".to_string())
                );
                assert_eq!(
                    global_metadata["timestamp"],
                    Json::String("2024-01-01T00:00:00Z".to_string())
                );

                // Check first item (concept)
                match &upsert.items[0] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "drug");
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: Some("drug_001".to_string()),
                                r#type: None,
                                name: None,
                            }
                        );
                        let attrs = concept.set_attributes.as_ref().unwrap();
                        assert_eq!(attrs["name"], Json::String("TestDrug".to_string()));
                        assert_eq!(attrs["active"], Json::Bool(true));
                        assert_eq!(attrs["dosage"], Json::Null);
                    }
                    _ => panic!("Expected ConceptBlock"),
                }

                // Check second item (proposition)
                match &upsert.items[1] {
                    UpsertItem::Proposition(prop) => {
                        assert_eq!(prop.handle, "relation");
                        assert_eq!(
                            prop.proposition,
                            PropositionMatcher {
                                subject: TargetTerm::Concept(ConceptMatcher {
                                    id: Some("drug_001".to_string()),
                                    r#type: None,
                                    name: None,
                                }),
                                predicate: PredTerm::Literal("interacts_with".to_string()),
                                object: TargetTerm::Concept(ConceptMatcher {
                                    id: Some("drug_002".to_string()),
                                    r#type: None,
                                    name: None,
                                }),
                            }
                        );
                        let metadata = prop.metadata.as_ref().unwrap();
                        assert_eq!(
                            metadata["interaction_type"],
                            Json::String("synergistic".to_string())
                        );
                    }
                    _ => panic!("Expected PropositionBlock"),
                }

                // Check third item (concept with proposition referencing local handle)
                match &upsert.items[2] {
                    UpsertItem::Concept(concept) => {
                        assert_eq!(concept.handle, "target");
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: Some("drug_002".to_string()),
                                r#type: None,
                                name: None,
                            }
                        );
                        let props = concept.set_propositions.as_ref().unwrap();
                        assert_eq!(props.len(), 1);
                        assert_eq!(
                            props[0],
                            SetProposition {
                                predicate: "belongs_to".to_string(),
                                object: TargetTerm::Variable("relation".to_string()),
                                metadata: None,
                            }
                        );
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
            CONCEPT ?minimal {
                { id: "test_001" }
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
                        assert_eq!(
                            concept.concept,
                            ConceptMatcher {
                                id: Some("test_001".to_string()),
                                r#type: None,
                                name: None,
                            }
                        );
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
        { type: "Drug", name: "Test" }
        "#;
        assert!(parse_kml_statement(input1).is_err());

        // Invalid local handle (missing ?)
        let input2 = r#"
        UPSERT {
            CONCEPT drug {
                { id: "test" }
            }
        }
        "#;
        assert!(parse_kml_statement(input2).is_err());

        // Missing concept clause in concept
        let input3 = r#"
        UPSERT {
            CONCEPT @drug {
                SET ATTRIBUTES { name: "Test" }
            }
        }
        "#;
        assert!(parse_kml_statement(input3).is_err());
    }
}
