use anda_kip::{Json, KipError, Map, OrderByCondition, OrderDirection, PredTerm};
use std::borrow::Cow;

use crate::entity::{Concept, EntityID, Properties, Proposition};

pub trait Pipe<T> {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
        Self: Sized;
}

impl<T> Pipe<T> for T {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}

pub fn extract_concept_field_value(concept: &Concept, path: &[String]) -> Result<Json, KipError> {
    if path.is_empty() {
        return Ok(concept.to_concept_node());
    }

    match path[0].as_str() {
        "id" if path.len() == 1 => Ok(concept.entity_id().to_string().into()),
        "type" if path.len() == 1 => Ok(concept.r#type.clone().into()),
        "name" if path.len() == 1 => Ok(concept.name.clone().into()),
        "attributes" if path.len() <= 2 => {
            if path.len() == 1 {
                Ok(concept.attributes.clone().into())
            } else {
                concept
                    .attributes
                    .get(&path[1])
                    .cloned()
                    .unwrap_or(Json::Null)
                    .pipe(Ok)
            }
        }
        "metadata" if path.len() <= 2 => {
            if path.len() == 1 {
                Ok(concept.metadata.clone().into())
            } else {
                concept
                    .metadata
                    .get(&path[1])
                    .cloned()
                    .unwrap_or(Json::Null)
                    .pipe(Ok)
            }
        }
        _ => Err(KipError::InvalidCommand(format!(
            "Invalid field path: {}",
            path.join(".")
        ))),
    }
}

// 从命题中提取字段值
pub fn extract_proposition_field_value(
    proposition: &Proposition,
    predicate: &str,
    path: &[String],
) -> Result<Json, KipError> {
    if !proposition.predicates.contains(predicate) {
        return Err(KipError::Execution(format!(
            "Invalid predicate: {}",
            predicate
        )));
    }

    if path.is_empty() {
        return proposition
            .to_proposition_link(predicate)
            .ok_or_else(|| KipError::InvalidCommand(format!("Invalid predicate: {}", predicate)));
    }

    let prop = proposition
        .properties
        .get(predicate)
        .map(Cow::Borrowed)
        .unwrap_or_else(|| {
            Cow::Owned(Properties {
                attributes: Map::new(),
                metadata: Map::new(),
            })
        });

    match path[0].as_str() {
        "id" if path.len() == 1 => Ok(proposition
            .entity_id(predicate.to_string())
            .to_string()
            .into()),
        "subject" if path.len() == 1 => Ok(proposition.subject.to_string().into()),
        "object" if path.len() == 1 => Ok(proposition.object.to_string().into()),
        "predicate" if path.len() == 1 => Ok(predicate.into()),
        "attributes" if path.len() <= 2 => {
            if path.len() == 1 {
                Ok(prop.attributes.clone().into())
            } else {
                prop.attributes
                    .get(&path[1])
                    .cloned()
                    .unwrap_or(Json::Null)
                    .pipe(Ok)
            }
        }
        "metadata" if path.len() <= 2 => {
            if path.len() == 1 {
                Ok(prop.metadata.clone().into())
            } else {
                prop.metadata
                    .get(&path[1])
                    .cloned()
                    .unwrap_or(Json::Null)
                    .pipe(Ok)
            }
        }
        _ => Err(KipError::InvalidCommand(format!(
            "Invalid field path: {}",
            path.join(".")
        ))),
    }
}

// 应用排序
pub fn apply_order_by(
    mut values: Vec<Json>,
    variable: &str,
    order_by: &[OrderByCondition],
) -> Vec<Json> {
    values.sort_by(|a, b| {
        for cond in order_by {
            if cond.variable.var != variable {
                continue; // 只处理与当前变量相关的排序条件
            }

            let path = format!("/{}", cond.variable.path.join("/"));

            let a_val = a.pointer(&path);
            let b_val = b.pointer(&path);

            let ordering = match (a_val, b_val) {
                (Some(Json::Number(a)), Some(Json::Number(b))) => a
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&b.as_f64().unwrap_or(0.0)),
                (Some(Json::String(a)), Some(Json::String(b))) => Some(a.cmp(b)),
                (Some(Json::Bool(a)), Some(Json::Bool(b))) => Some(a.cmp(b)),
                _ => None,
            };

            if let Some(ord) = ordering {
                let result = match cond.direction {
                    OrderDirection::Asc => ord,
                    OrderDirection::Desc => ord.reverse(),
                };

                if result != std::cmp::Ordering::Equal {
                    return result;
                }
            }
        }
        std::cmp::Ordering::Equal
    });

    values
}

// 处理谓词匹配逻辑
pub fn match_predicate_against_proposition(
    proposition: &Proposition,
    predicate: &PredTerm,
) -> Result<Option<(EntityID, Vec<String>, EntityID)>, KipError> {
    match predicate {
        PredTerm::Literal(pred) => {
            if proposition.predicates.contains(pred) {
                Ok(Some((
                    proposition.subject.clone(),
                    vec![pred.clone()],
                    proposition.object.clone(),
                )))
            } else {
                Ok(None)
            }
        }
        PredTerm::Variable(_) => Ok(Some((
            proposition.subject.clone(),
            proposition.predicates.iter().cloned().collect(),
            proposition.object.clone(),
        ))),
        PredTerm::Alternative(preds) => {
            let matched_preds = proposition
                .predicates
                .iter()
                .filter(|p| preds.contains(p))
                .cloned()
                .collect::<Vec<_>>();
            if !matched_preds.is_empty() {
                Ok(Some((
                    proposition.subject.clone(),
                    matched_preds,
                    proposition.object.clone(),
                )))
            } else {
                Ok(None)
            }
        }
        _ => Err(KipError::InvalidCommand(format!(
            "Predicate must be either Literal or Variable, got: {predicate:?}"
        ))),
    }
}
