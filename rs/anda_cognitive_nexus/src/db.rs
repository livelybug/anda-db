use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    index::{extract_json_text, virtual_field_name, virtual_field_value},
    query::{Filter, Query, RangeQuery, Search},
    unix_ms,
};
use anda_db_schema::Fv;
use anda_db_tfs::jieba_tokenizer;
use anda_kip::*;
use async_trait::async_trait;
use futures::try_join as try_join_await;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    str::FromStr,
    sync::Arc,
};

use crate::{entity::*, helper::*, types::*};

#[derive(Clone, Debug)]
pub struct CognitiveNexus {
    db: Arc<AndaDB>,
    concepts: Arc<Collection>,
    propositions: Arc<Collection>,
}

#[async_trait(?Send)]
impl Executor for CognitiveNexus {
    async fn execute(&self, command: Command, dry_run: bool) -> Result<Json, KipError> {
        match command {
            Command::Kql(command) => self.execute_kql(command, dry_run).await,
            Command::Kml(command) => self.execute_kml(command, dry_run).await,
            Command::Meta(command) => self.execute_meta(command).await,
        }
    }
}

impl CognitiveNexus {
    pub async fn connect(db: Arc<AndaDB>) -> Result<Self, DBError> {
        let schema = Concept::schema()?;
        let concepts = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "concepts".to_string(),
                    description: "Concept nodes".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["type", "name"]).await?;
                    collection.create_btree_index_nx(&["type"]).await?;
                    collection.create_btree_index_nx(&["name"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "attributes", "metadata"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let schema = Proposition::schema()?;
        let propositions = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "propositions".to_string(),
                    description: "Proposition links".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection
                        .create_btree_index_nx(&["subject", "object"])
                        .await?;
                    collection.create_btree_index_nx(&["subject"]).await?;
                    collection.create_btree_index_nx(&["object"]).await?;
                    collection.create_btree_index_nx(&["predicates"]).await?;
                    collection
                        .create_bm25_index_nx(&["predicates", "properties"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;
        Ok(Self {
            db,
            concepts,
            propositions,
        })
    }

    pub async fn close(&self) -> Result<(), DBError> {
        self.db.close().await
    }

    pub fn name(&self) -> &str {
        self.db.name()
    }

    async fn execute_kql(&self, command: KqlQuery, _dry_run: bool) -> Result<Json, KipError> {
        let mut ctx = QueryContext::default();

        // 执行WHERE子句
        for clause in command.where_clauses {
            self.execute_where_clause(&mut ctx, clause).await?;
        }

        // 执行FIND子句
        let mut rt = self
            .execute_find_clause(
                &mut ctx,
                command.find_clause,
                command.order_by,
                command.offset,
                command.limit,
            )
            .await?;

        if rt.len() == 1 {
            Ok(rt.pop().unwrap())
        } else {
            Ok(Json::Array(rt))
        }
    }

    async fn execute_kml(&self, command: KmlStatement, dry_run: bool) -> Result<Json, KipError> {
        match command {
            KmlStatement::Upsert(upsert_blocks) => {
                self.execute_upsert(upsert_blocks, dry_run).await
            }
            KmlStatement::Delete(delete_statement) => {
                self.execute_delete(delete_statement, dry_run).await
            }
        }
    }

    async fn execute_meta(&self, command: MetaCommand) -> Result<Json, KipError> {
        match command {
            MetaCommand::Describe(DescribeTarget::Primer) => self.execute_describe_primer().await,
            MetaCommand::Describe(DescribeTarget::Domains) => self.execute_describe_domains().await,
            MetaCommand::Describe(DescribeTarget::ConceptTypes { limit, offset }) => {
                self.execute_describe_concept_types(limit, offset).await
            }
            MetaCommand::Describe(DescribeTarget::ConceptType(name)) => {
                self.execute_describe_concept_type(name).await
            }
            MetaCommand::Describe(DescribeTarget::PropositionTypes { limit, offset }) => {
                self.execute_describe_proposition_types(limit, offset).await
            }
            MetaCommand::Describe(DescribeTarget::PropositionType(name)) => {
                self.execute_describe_proposition_type(name).await
            }
            MetaCommand::Search(command) => self.execute_search(command).await,
        }
    }

    async fn execute_where_clause(
        &self,
        ctx: &mut QueryContext,
        clause: WhereClause,
    ) -> Result<(), KipError> {
        match clause {
            WhereClause::Concept(clause) => self.execute_concept_clause(ctx, clause).await,
            WhereClause::Proposition(clause) => self.execute_proposition_clause(ctx, clause).await,
            WhereClause::Filter(clause) => self.execute_filter_clause(ctx, clause).await,
            WhereClause::Not(clauses) => self.execute_not_clause(ctx, clauses).await,
            WhereClause::Optional(clauses) => self.execute_optional_clause(ctx, clauses).await,
            WhereClause::Union(clauses) => self.execute_union_clause(ctx, clauses).await,
        }?;

        Ok(())
    }

    async fn execute_concept_clause(
        &self,
        ctx: &mut QueryContext,
        clause: ConceptClause,
    ) -> Result<(), KipError> {
        if ctx.entities.contains_key(&clause.variable) {
            return Err(KipError::InvalidCommand(format!(
                "Variable '{}' already exists in context",
                clause.variable
            )));
        }

        let concept_ids = self.query_concept_ids(&clause.matcher).await?;
        ctx.entities.insert(
            clause.variable,
            concept_ids.into_iter().map(EntityID::Concept).collect(),
        );

        Ok(())
    }

    async fn execute_proposition_clause(
        &self,
        ctx: &mut QueryContext,
        clause: PropositionClause,
    ) -> Result<(), KipError> {
        if let Some(var) = &clause.variable {
            if ctx.entities.contains_key(var) {
                return Err(KipError::InvalidCommand(format!(
                    "Variable '{var}' already exists in context",
                )));
            }
        }

        let result = match clause.matcher {
            PropositionMatcher::ID(id) => {
                let entity_id = EntityID::from_str(&id).map_err(KipError::Parse)?;
                if !matches!(entity_id, EntityID::Proposition(_, _)) {
                    return Err(KipError::InvalidCommand(format!(
                        "Invalid proposition link ID: {id:?}"
                    )));
                }
                TargetEntities::IDs(vec![entity_id])
            }
            PropositionMatcher::Object {
                subject,
                predicate,
                object,
            } => {
                self.match_propositions(ctx, subject, predicate, object)
                    .await?
            }
        };

        if let TargetEntities::IDs(ids) = result {
            if let Some(var) = clause.variable {
                ctx.entities.insert(var, ids.into_iter().collect());
            }
        }

        Ok(())
    }

    async fn execute_filter_clause(
        &self,
        ctx: &mut QueryContext,
        clause: FilterClause,
    ) -> Result<(), KipError> {
        let mut entities: HashMap<String, Vec<EntityID>> = ctx
            .entities
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();

        loop {
            let mut bindings_snapshot = entities.clone();
            let mut bindings_cursor = HashMap::new();
            match self
                .evaluate_filter_expression(
                    ctx,
                    clause.expression.clone(),
                    &mut bindings_snapshot,
                    &mut bindings_cursor,
                )
                .await?
            {
                Some(true) => {
                    // 继续处理剩余绑定
                    entities = bindings_snapshot;
                }
                Some(false) => {
                    // 过滤不通过，移除相关值
                    for (var, id) in bindings_cursor {
                        if let Some(existing) = ctx.entities.get_mut(&var) {
                            existing.remove(&id);
                        }
                    }
                    // 继续处理剩余绑定
                    entities = bindings_snapshot;
                }
                None => {
                    // 没有更多符合条件的绑定可供处理，退出循环
                    return Ok(());
                }
            }
        }
    }

    async fn execute_not_clause(
        &self,
        ctx: &mut QueryContext,
        clauses: Vec<WhereClause>,
    ) -> Result<(), KipError> {
        let mut not_context = ctx.clone();
        for clause in clauses {
            Box::pin(self.execute_where_clause(&mut not_context, clause)).await?;
        }

        for (var, ids) in not_context.entities {
            // 如果NOT子句中有变量绑定，则从当前上下文中移除这些绑定
            if let Some(existing) = ctx.entities.get_mut(&var) {
                existing.retain(|id| !ids.contains(id));
            }
        }

        Ok(())
    }

    async fn execute_optional_clause(
        &self,
        ctx: &mut QueryContext,
        clauses: Vec<WhereClause>,
    ) -> Result<(), KipError> {
        let mut optional_context = ctx.clone();
        for clause in clauses {
            Box::pin(self.execute_where_clause(&mut optional_context, clause)).await?;
        }

        // 合并 OPTIONAL 子句
        for (var, ids) in optional_context.entities {
            if let Some(existing) = ctx.entities.get_mut(&var) {
                existing.extend(ids);
            } else {
                ctx.entities.insert(var, ids);
            }
        }

        Ok(())
    }

    async fn execute_union_clause(
        &self,
        ctx: &mut QueryContext,
        clauses: Vec<WhereClause>,
    ) -> Result<(), KipError> {
        let mut union_context = QueryContext {
            cache: ctx.cache.clone(),
            ..Default::default()
        };

        for clause in clauses {
            Box::pin(self.execute_where_clause(&mut union_context, clause)).await?;
        }

        // 合并 UNION 子句
        for (var, ids) in union_context.entities {
            if let Some(existing) = ctx.entities.get_mut(&var) {
                existing.extend(ids);
            } else {
                ctx.entities.insert(var, ids);
            }
        }

        Ok(())
    }

    async fn execute_find_clause(
        &self,
        ctx: &mut QueryContext,
        clause: FindClause,
        order_by: Option<Vec<OrderByCondition>>,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<Vec<Json>, KipError> {
        let mut result: Vec<Json> = Vec::with_capacity(clause.expressions.len());
        let bindings: HashMap<String, Vec<EntityID>> = ctx
            .entities
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();

        let order_by = order_by.unwrap_or_default();
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(0);

        for expr in clause.expressions {
            match expr {
                FindExpression::Variable(dot_path) => {
                    let mut col = self
                        .resolve_result(&ctx.cache, &bindings, &dot_path, &order_by)
                        .await?;

                    if offset > 0 && offset < col.len() {
                        col = col.into_iter().skip(offset).collect();
                    }
                    if limit > 0 && limit < col.len() {
                        col.truncate(limit);
                    }

                    result.push(Json::Array(col));
                }
                FindExpression::Aggregation {
                    func,
                    var,
                    distinct,
                } => {
                    let col = self
                        .resolve_result(&ctx.cache, &bindings, &var, &[])
                        .await?;

                    result.push(func.calculate(&col, distinct));
                }
            }
        }

        Ok(result)
    }

    async fn execute_describe_primer(&self) -> Result<Json, KipError> {
        let cache = QueryCache::default();
        let matcher = ConceptMatcher::Object {
            r#type: PERSON_TYPE.to_string(),
            name: META_SELF_NAME.to_string(),
        };
        let me = self.query_concept_ids(&matcher).await?;

        let me = me
            .first()
            .ok_or_else(|| KipError::Execution(format!("Concept {matcher} not found")))?;
        let me = self
            .try_get_concept_with(&cache, *me, |concept| {
                extract_concept_field_value(concept, &["attributes".to_string()])
            })
            .await?;

        let domains = self
            .query_concept_ids(&ConceptMatcher::Type(DOMAIN_TYPE.to_string()))
            .await?;
        let mut domain_map: Vec<DomainInfo> = Vec::with_capacity(domains.len());
        for id in domains {
            let mut info = self
                .try_get_concept_with(&cache, id, |concept| Ok(DomainInfo::from(concept)))
                .await?;
            let subjects = self
                .find_propositions(&cache, &EntityID::Concept(id), BELONGS_TO_DOMAIN_TYPE, true)
                .await?;
            let subjects = subjects.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
            for sub in subjects {
                if let EntityID::Concept(id) = sub {
                    let _ = self
                        .try_get_concept_with(&cache, id, |concept| {
                            if concept.r#type == META_CONCEPT_TYPE {
                                info.key_concept_types.push(ConceptTypeInfo::from(concept));
                            } else if concept.r#type == META_PROPOSITION_TYPE {
                                info.key_proposition_types
                                    .push(PropositionTypeInfo::from(concept));
                            }
                            Ok(())
                        })
                        .await;
                }
            }

            domain_map.push(info);
        }

        Ok(json!({
            "identity": me,
            "domain_map": domain_map,
        }))
    }

    async fn execute_describe_domains(&self) -> Result<Json, KipError> {
        let ids = self
            .query_concept_ids(&ConceptMatcher::Type(DOMAIN_TYPE.to_string()))
            .await?;
        let cache = QueryCache::default();
        let mut result: Vec<Json> = Vec::with_capacity(ids.len());
        for id in ids {
            let concept = self
                .try_get_concept_with(&cache, id, |concept| {
                    extract_concept_field_value(concept, &[])
                })
                .await?;
            result.push(concept);
        }
        Ok(json!(result))
    }

    async fn execute_describe_concept_types(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Json, KipError> {
        let index = self.concepts.get_btree_index(&["type"]).map_err(|e| {
            KipError::Execution(format!("Failed to get concept type index: {:?}", e))
        })?;

        let result = index.keys(offset.unwrap_or(0), limit);
        Ok(json!(result))
    }

    async fn execute_describe_concept_type(&self, name: String) -> Result<Json, KipError> {
        let id = self
            .query_concept_ids(&ConceptMatcher::Object {
                r#type: META_CONCEPT_TYPE.to_string(),
                name: name.clone(),
            })
            .await?;

        let id = id
            .first()
            .ok_or_else(|| KipError::Execution(format!("Concept type {name:?} not found")))?;
        let result = self
            .try_get_concept_with(&QueryCache::default(), *id, |concept| {
                extract_concept_field_value(concept, &[])
            })
            .await?;
        Ok(result)
    }

    async fn execute_describe_proposition_types(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Json, KipError> {
        let index = self
            .propositions
            .get_btree_index(&["predicates"])
            .map_err(|e| {
                KipError::Execution(format!(
                    "Failed to get proposition predicates index: {:?}",
                    e
                ))
            })?;

        let result = index.keys(offset.unwrap_or(0), limit);
        Ok(json!(result))
    }

    async fn execute_describe_proposition_type(&self, name: String) -> Result<Json, KipError> {
        let id = self
            .query_concept_ids(&ConceptMatcher::Object {
                r#type: META_PROPOSITION_TYPE.to_string(),
                name: name.clone(),
            })
            .await?;

        let id = id
            .first()
            .ok_or_else(|| KipError::Execution(format!("Proposition type {name:?} not found")))?;
        let result = self
            .try_get_concept_with(&QueryCache::default(), *id, |concept| {
                extract_concept_field_value(concept, &[])
            })
            .await?;
        Ok(result)
    }

    async fn execute_search(&self, command: SearchCommand) -> Result<Json, KipError> {
        match command.target {
            SearchTarget::Concept => {
                let result: Vec<Concept> = self
                    .concepts
                    .search_as(Query {
                        search: Some(Search {
                            text: Some(command.term),
                            logical_search: true,
                            ..Default::default()
                        }),
                        filter: None,
                        limit: command.limit,
                    })
                    .await
                    .map_err(|err| {
                        KipError::Execution(format!("Failed to search concept: {:?}", err))
                    })?;
                Ok(json!(
                    result
                        .into_iter()
                        .map(|c| c.into_concept_node())
                        .collect::<Vec<_>>()
                ))
            }
            SearchTarget::Proposition => {
                let tokens = self.propositions.tokenize(&command.term);
                let ids = self
                    .propositions
                    .search_ids(Query {
                        search: Some(Search {
                            text: Some(command.term),
                            logical_search: true,
                            ..Default::default()
                        }),
                        filter: None,
                        limit: command.limit,
                    })
                    .await
                    .map_err(|err| {
                        KipError::Execution(format!("Failed to search proposition: {:?}", err))
                    })?;
                let cache = QueryCache::default();
                let mut result: Vec<Json> = Vec::with_capacity(ids.len());
                for id in ids {
                    let rt = self
                        .try_get_proposition_with(&cache, id, |proposition| {
                            let mut rt: Vec<Json> = Vec::new();
                            for (predicate, prop) in &proposition.properties {
                                // collect searchable texts
                                let mut texts: Vec<&str> = vec![predicate];
                                for (_, val) in &prop.attributes {
                                    extract_json_text(&mut texts, val);
                                }
                                for (_, val) in &prop.metadata {
                                    extract_json_text(&mut texts, val);
                                }

                                if tokens.iter().any(|t| texts.contains(&t.as_str())) {
                                    if let Ok(val) =
                                        extract_proposition_field_value(proposition, predicate, &[])
                                    {
                                        rt.push(val);
                                    }
                                }
                            }

                            Ok(rt)
                        })
                        .await?;
                    result.extend(rt);
                }
                Ok(json!(result))
            }
        }
    }

    // 处理多跳匹配
    async fn handle_multi_hop_matching(
        &self,
        ctx: &QueryContext,
        subjects: TargetEntities,
        predicate: String,
        min: u16,
        max: Option<u16>,
        objects: TargetEntities,
    ) -> Result<PropositionsMatchResult, KipError> {
        let mut result = PropositionsMatchResult::new();

        if matches!(&subjects, TargetEntities::IDs(_)) {
            let start_nodes = match subjects {
                TargetEntities::IDs(ids) => ids,
                _ => unreachable!(),
            };

            let max_hops = max.unwrap_or(10).min(10);
            for start_node in start_nodes {
                let paths = self
                    .bfs_multi_hop(
                        &ctx.cache,
                        start_node.clone(),
                        &predicate,
                        min,
                        max_hops,
                        &objects,
                        false,
                    )
                    .await?;

                for path in paths {
                    result.matched_subjects.insert(path.start);
                    result.matched_objects.insert(path.end);
                    result.matched_predicates.insert(predicate.clone());
                    result.matched_propositions.extend(path.propositions);
                }
            }
        } else {
            let start_nodes = match objects {
                TargetEntities::IDs(ids) => ids,
                _ => {
                    return Err(KipError::InvalidCommand(
                        "The subject or object cannot both be variables in multi-hop matching"
                            .to_string(),
                    ));
                }
            };

            let max_hops = max.unwrap_or(10).min(10);
            for start_node in start_nodes {
                let paths = self
                    .bfs_multi_hop(
                        &ctx.cache,
                        start_node.clone(),
                        &predicate,
                        min,
                        max_hops,
                        &subjects,
                        true,
                    )
                    .await?;

                for path in paths {
                    result.matched_subjects.insert(path.end);
                    result.matched_objects.insert(path.start);
                    result.matched_predicates.insert(predicate.clone());
                    result.matched_propositions.extend(path.propositions);
                }
            }
        }

        Ok(result)
    }

    // 处理主体和客体都是具体ID的匹配
    async fn handle_subject_object_ids_matching(
        &self,
        ctx: &QueryContext,
        subject_ids: Vec<EntityID>,
        object_ids: Vec<EntityID>,
        predicate: PredTerm,
    ) -> Result<PropositionsMatchResult, KipError> {
        let mut result = PropositionsMatchResult::new();

        for subject_id in &subject_ids {
            for object_id in &object_ids {
                let virtual_name = virtual_field_name(&["subject", "object"]);
                let virtual_val = virtual_field_value(&[
                    Some(&Fv::Text(subject_id.to_string())),
                    Some(&Fv::Text(object_id.to_string())),
                ])
                .unwrap();

                let ids = self
                    .propositions
                    .query_ids(
                        Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                        None,
                    )
                    .await
                    .map_err(|e| {
                        KipError::Execution(format!("Failed to query propositions: {:?}", e))
                    })?;

                for id in ids {
                    if let Some((subj, preds, obj)) = self
                        .try_get_proposition_with(&ctx.cache, id, |proposition| {
                            match_predicate_against_proposition(proposition, &predicate)
                        })
                        .await?
                    {
                        result.add_match(subj, obj, preds, id);
                    }
                }
            }
        }

        Ok(result)
    }

    // 处理主体ID和任意对象的匹配
    async fn handle_subject_ids_any_matching(
        &self,
        ctx: &QueryContext,
        subject_ids: Vec<EntityID>,
        predicate: PredTerm,
        any_propositions: bool,
    ) -> Result<PropositionsMatchResult, KipError> {
        let mut result = PropositionsMatchResult::new();

        for subject_id in &subject_ids {
            let ids = self
                .propositions
                .query_ids(
                    Filter::Field((
                        "subject".to_string(),
                        RangeQuery::Eq(Fv::Text(subject_id.to_string())),
                    )),
                    None,
                )
                .await
                .map_err(|e| {
                    KipError::Execution(format!("Failed to query propositions: {:?}", e))
                })?;

            for id in ids {
                if let Some((subj, preds, obj)) = self
                    .try_get_proposition_with(&ctx.cache, id, |proposition| {
                        if any_propositions && matches!(proposition.object, EntityID::Concept(_)) {
                            return Ok(None);
                        }
                        match_predicate_against_proposition(proposition, &predicate)
                    })
                    .await?
                {
                    result.add_match(subj, obj, preds, id);
                }
            }
        }

        Ok(result)
    }

    // 处理任意主体和对象ID的匹配
    async fn handle_any_to_object_ids_matching(
        &self,
        ctx: &QueryContext,
        object_ids: Vec<EntityID>,
        predicate: PredTerm,
        any_propositions: bool,
    ) -> Result<PropositionsMatchResult, KipError> {
        let mut result = PropositionsMatchResult::new();

        for object_id in &object_ids {
            let ids = self
                .propositions
                .query_ids(
                    Filter::Field((
                        "object".to_string(),
                        RangeQuery::Eq(Fv::Text(object_id.to_string())),
                    )),
                    None,
                )
                .await
                .map_err(|e| {
                    KipError::Execution(format!("Failed to query propositions: {:?}", e))
                })?;

            for id in ids {
                if let Some((subj, preds, obj)) = self
                    .try_get_proposition_with(&ctx.cache, id, |proposition| {
                        if any_propositions && matches!(proposition.subject, EntityID::Concept(_)) {
                            return Ok(None);
                        }
                        match_predicate_against_proposition(proposition, &predicate)
                    })
                    .await?
                {
                    result.add_match(subj, obj, preds, id);
                }
            }
        }

        Ok(result)
    }

    // 处理谓词匹配
    async fn handle_predicate_matching(
        &self,
        ctx: &QueryContext,
        predicate: PredTerm,
    ) -> Result<PropositionsMatchResult, KipError> {
        let mut result = PropositionsMatchResult::new();
        let predicates = match &predicate {
            PredTerm::Literal(pred) => vec![pred.clone()],
            PredTerm::Alternative(preds) => preds.clone(),
            _ => {
                return Err(KipError::InvalidCommand(format!(
                    "Predicate must be either Literal or Alternative, got: {predicate:?}"
                )));
            }
        };

        let ids = self
            .propositions
            .query_ids(
                Filter::Field((
                    "predicates".to_string(),
                    RangeQuery::Or(
                        predicates
                            .into_iter()
                            .map(|v| Box::new(RangeQuery::Eq(v.into())))
                            .collect(),
                    ),
                )),
                None,
            )
            .await
            .map_err(|e| KipError::Execution(format!("Failed to query propositions: {:?}", e)))?;

        for id in ids {
            if let Some((subj, preds, obj)) = self
                .try_get_proposition_with(&ctx.cache, id, |proposition| {
                    match_predicate_against_proposition(proposition, &predicate)
                })
                .await?
            {
                result.add_match(subj, obj, preds, id);
            }
        }

        Ok(result)
    }

    // BFS 路径查找实现
    #[allow(clippy::too_many_arguments)]
    async fn bfs_multi_hop(
        &self,
        cache: &QueryCache,
        start: EntityID,
        predicate: &str,
        min_hops: u16,
        max_hops: u16,
        targets: &TargetEntities,
        reverse: bool,
    ) -> Result<Vec<GraphPath>, KipError> {
        use std::collections::VecDeque;

        let mut queue: VecDeque<GraphPath> = VecDeque::new();
        let mut results: Vec<GraphPath> = Vec::new();
        let mut visited: HashSet<(EntityID, u16)> = HashSet::new(); // (node, depth) 防止循环

        // 初始化队列
        queue.push_back(GraphPath {
            start: start.clone(),
            end: start.clone(),
            propositions: Vec::new(),
            hops: 0,
        });

        while let Some(current_path) = queue.pop_front() {
            // 检查是否已访问过此节点在此深度
            let state = (current_path.end.clone(), current_path.hops);
            if visited.contains(&state) {
                continue;
            }
            visited.insert(state);

            // 如果达到最大跳数，停止扩展此路径
            if current_path.hops >= max_hops {
                if current_path.hops >= min_hops {
                    match targets {
                        TargetEntities::IDs(ids) => {
                            if ids.contains(&current_path.end) {
                                results.push(current_path);
                            }
                        }
                        TargetEntities::AnyPropositions => {
                            if matches!(current_path.end, EntityID::Proposition(_, _)) {
                                results.push(current_path);
                            }
                        }
                        TargetEntities::Any => {
                            results.push(current_path);
                        }
                    }
                }
                continue;
            }

            // 查找从当前节点出发的所有指定谓词的边
            let props = self
                .find_propositions(cache, &current_path.end, predicate, reverse)
                .await?;

            for (prop_id, target_node) in props {
                let mut new_path = current_path.clone();
                new_path.end = target_node;
                new_path.propositions.push(prop_id);
                new_path.hops += 1;

                // 如果满足最小跳数要求，检查是否为有效结果
                if new_path.hops >= min_hops {
                    match targets {
                        TargetEntities::IDs(ids) => {
                            if ids.contains(&new_path.end) {
                                results.push(new_path.clone());
                            }
                        }
                        TargetEntities::AnyPropositions => {
                            if matches!(new_path.end, EntityID::Proposition(_, _)) {
                                results.push(new_path.clone());
                            }
                        }
                        TargetEntities::Any => {
                            results.push(new_path.clone());
                        }
                    }
                }

                // 如果未达到最大跳数，继续扩展
                if new_path.hops < max_hops {
                    queue.push_back(new_path);
                }
            }
        }

        Ok(results)
    }

    async fn execute_upsert(
        &self,
        upsert_blocks: Vec<UpsertBlock>,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        let blocks = upsert_blocks.len();
        let mut concept_nodes: Vec<EntityID> = Vec::new();
        let mut proposition_links: Vec<EntityID> = Vec::new();
        for block in upsert_blocks {
            let mut handle_map: HashMap<String, EntityID> = HashMap::new();
            let mut cached_pks: HashMap<EntityPK, EntityID> = HashMap::new();
            let default_metadata: Map<String, Json> = block.metadata.unwrap_or_default();

            for item in block.items {
                match item {
                    UpsertItem::Concept(concept_block) => {
                        if let Some(entity_id) = self
                            .execute_concept_block(
                                concept_block,
                                &default_metadata,
                                &mut handle_map,
                                &mut cached_pks,
                                dry_run,
                            )
                            .await?
                        {
                            concept_nodes.push(entity_id);
                        }
                    }
                    UpsertItem::Proposition(proposition_block) => {
                        if let Some(entity_id) = self
                            .execute_proposition_block(
                                proposition_block,
                                &default_metadata,
                                &mut handle_map,
                                &mut cached_pks,
                                dry_run,
                            )
                            .await?
                        {
                            proposition_links.push(entity_id);
                        }
                    }
                }
            }
        }

        let now_ms = unix_ms();
        try_join_await!(self.concepts.flush(now_ms), self.propositions.flush(now_ms))
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        Ok(json!({
            "blocks": blocks,
            "concept_nodes": concept_nodes,
            "proposition_links": proposition_links,
        }))
    }

    async fn execute_concept_block(
        &self,
        concept_block: ConceptBlock,
        default_metadata: &Map<String, Json>,
        handle_map: &mut HashMap<String, EntityID>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
        dry_run: bool,
    ) -> Result<Option<EntityID>, KipError> {
        let concept_pk = ConceptPK::try_from(concept_block.concept)?;
        if let Some(propositions) = &concept_block.set_propositions {
            for set_prop in propositions {
                self.check_target_term_for_kml(&set_prop.object, handle_map)?;
            }
        }

        if dry_run {
            return Ok(None);
        }

        let attributes = concept_block
            .set_attributes
            .map(|val| val.into_iter().collect())
            .unwrap_or_default();
        let metadata = concept_block
            .metadata
            .map(|val| val.into_iter().collect())
            .unwrap_or_else(|| default_metadata.clone());

        let entity_id = self
            .upsert_concept(concept_pk, attributes, metadata)
            .await
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        handle_map.insert(concept_block.handle.clone(), entity_id.clone());

        if let Some(propositions) = concept_block.set_propositions {
            for set_prop in propositions {
                self.execute_set_proposition(
                    &entity_id,
                    set_prop,
                    default_metadata,
                    handle_map,
                    cached_pks,
                )
                .await
                .map_err(|err| KipError::Execution(format!("{err:?}")))?;
            }
        }

        Ok(Some(entity_id))
    }

    async fn execute_proposition_block(
        &self,
        proposition_block: PropositionBlock,
        default_metadata: &Map<String, Json>,
        handle_map: &mut HashMap<String, EntityID>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
        dry_run: bool,
    ) -> Result<Option<EntityID>, KipError> {
        let proposition_pk = PropositionPK::try_from(proposition_block.proposition)?;
        if dry_run {
            return Ok(None);
        }

        let attributes = proposition_block
            .set_attributes
            .map(|val| val.into_iter().collect())
            .unwrap_or_default();
        let metadata = proposition_block
            .metadata
            .map(|val| val.into_iter().collect())
            .unwrap_or_else(|| default_metadata.clone());

        let entity_id = self
            .upsert_proposition(proposition_pk, attributes, metadata, cached_pks)
            .await
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        handle_map.insert(proposition_block.handle.clone(), entity_id.clone());

        Ok(Some(entity_id))
    }

    async fn execute_set_proposition(
        &self,
        subject: &EntityID,
        set_prop: SetProposition,
        default_metadata: &Map<String, Json>,
        handle_map: &HashMap<String, EntityID>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<EntityID, KipError> {
        let object_id = self
            .resolve_target_term(set_prop.object, handle_map, cached_pks)
            .await?;

        let proposition_pk = PropositionPK::Object {
            subject: Box::new(subject.clone().into()),
            predicate: set_prop.predicate,
            object: Box::new(object_id.clone().into()),
        };
        let metadata = set_prop
            .metadata
            .map(|val| val.into_iter().collect())
            .unwrap_or_else(|| default_metadata.clone());

        let entity_id = self
            .upsert_proposition(proposition_pk, Map::new(), metadata, cached_pks)
            .await
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        Ok(entity_id)
    }

    async fn execute_delete(
        &self,
        delete_statement: DeleteStatement,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        let result = match delete_statement {
            DeleteStatement::DeleteAttributes {
                attributes,
                target,
                where_clauses,
            } => {
                self.execute_delete_attributes(attributes, target, where_clauses, dry_run)
                    .await
            }
            DeleteStatement::DeleteMetadata {
                keys,
                target,
                where_clauses,
            } => {
                self.execute_delete_metadata(keys, target, where_clauses, dry_run)
                    .await
            }
            DeleteStatement::DeletePropositions {
                target,
                where_clauses,
            } => {
                self.execute_delete_propositions(target, where_clauses, dry_run)
                    .await
            }
            DeleteStatement::DeleteConcept {
                target,
                where_clauses,
            } => {
                self.execute_delete_concepts(target, where_clauses, dry_run)
                    .await
            }
        }?;

        let now_ms = unix_ms();
        try_join_await!(self.concepts.flush(now_ms), self.propositions.flush(now_ms))
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        Ok(result)
    }

    async fn execute_delete_attributes(
        &self,
        attributes: Vec<String>,
        target: String,
        where_clauses: Vec<WhereClause>,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        if dry_run {
            return Ok(json!({
                "updated_concepts": 0,
                "updated_propositions": 0,
            }));
        }

        let mut ctx = QueryContext::default();
        for clause in where_clauses {
            self.execute_where_clause(&mut ctx, clause).await?;
        }

        let target_entities = ctx.entities.get(&target).cloned().ok_or_else(|| {
            KipError::InvalidCommand(format!("Target term '{}' not found in context", target))
        })?;
        let mut updated_concepts: u64 = 0;
        let mut updated_propositions: u64 = 0;
        for entity_id in target_entities {
            match entity_id {
                EntityID::Concept(id) => {
                    if let Ok(mut concept) = self
                        .try_get_concept_with(&ctx.cache, id, |concept| Ok(concept.clone()))
                        .await
                    {
                        let length = concept.attributes.len();
                        for attr in &attributes {
                            concept.attributes.remove(attr);
                        }
                        if concept.attributes.len() < length
                            && self
                                .concepts
                                .update(
                                    id,
                                    BTreeMap::from([(
                                        "attributes".to_string(),
                                        concept.attributes.into(),
                                    )]),
                                )
                                .await
                                .is_ok()
                        {
                            updated_concepts += 1;
                        }
                    }
                }
                EntityID::Proposition(id, predicate) => {
                    if let Ok(mut proposition) = self
                        .try_get_proposition_with(&ctx.cache, id, |prop| Ok(prop.clone()))
                        .await
                    {
                        if let Some(prop) = proposition.properties.get_mut(&predicate) {
                            let length = prop.attributes.len();
                            for attr in &attributes {
                                prop.attributes.remove(attr);
                            }

                            if prop.attributes.len() < length
                                && self
                                    .propositions
                                    .update(
                                        id,
                                        BTreeMap::from([(
                                            "properties".to_string(),
                                            proposition.properties.into(),
                                        )]),
                                    )
                                    .await
                                    .is_ok()
                            {
                                updated_propositions += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(json!({
            "updated_concepts": updated_concepts,
            "updated_propositions": updated_propositions,
        }))
    }

    async fn execute_delete_metadata(
        &self,
        keys: Vec<String>,
        target: String,
        where_clauses: Vec<WhereClause>,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        if dry_run {
            return Ok(json!({
                "updated_concepts": 0,
                "updated_propositions": 0,
            }));
        }

        let mut ctx = QueryContext::default();
        for clause in where_clauses {
            self.execute_where_clause(&mut ctx, clause).await?;
        }

        let target_entities = ctx.entities.get(&target).cloned().ok_or_else(|| {
            KipError::InvalidCommand(format!("Target term '{}' not found in context", target))
        })?;
        let mut updated_concepts: u64 = 0;
        let mut updated_propositions: u64 = 0;
        for entity_id in target_entities {
            match entity_id {
                EntityID::Concept(id) => {
                    if let Ok(mut concept) = self
                        .try_get_concept_with(&ctx.cache, id, |concept| Ok(concept.clone()))
                        .await
                    {
                        let length = concept.metadata.len();
                        for name in &keys {
                            concept.metadata.remove(name);
                        }
                        if concept.attributes.len() < length
                            && self
                                .concepts
                                .update(
                                    id,
                                    BTreeMap::from([(
                                        "metadata".to_string(),
                                        concept.metadata.into(),
                                    )]),
                                )
                                .await
                                .is_ok()
                        {
                            updated_concepts += 1;
                        }
                    }
                }
                EntityID::Proposition(id, predicate) => {
                    if let Ok(mut proposition) = self
                        .try_get_proposition_with(&ctx.cache, id, |prop| Ok(prop.clone()))
                        .await
                    {
                        if let Some(prop) = proposition.properties.get_mut(&predicate) {
                            let length = prop.metadata.len();
                            for name in &keys {
                                prop.metadata.remove(name);
                            }

                            if prop.metadata.len() < length
                                && self
                                    .propositions
                                    .update(
                                        id,
                                        BTreeMap::from([(
                                            "properties".to_string(),
                                            proposition.properties.into(),
                                        )]),
                                    )
                                    .await
                                    .is_ok()
                            {
                                updated_propositions += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(json!({
            "updated_concepts": updated_concepts,
            "updated_propositions": updated_propositions,
        }))
    }

    async fn execute_delete_propositions(
        &self,
        target: String,
        where_clauses: Vec<WhereClause>,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        if dry_run {
            return Ok(json!({
                "deleted_propositions": 0
            }));
        }

        let mut ctx = QueryContext::default();
        for clause in where_clauses {
            self.execute_where_clause(&mut ctx, clause).await?;
        }

        let target_entities = ctx.entities.get(&target).cloned().ok_or_else(|| {
            KipError::InvalidCommand(format!("Target term '{}' not found in context", target))
        })?;

        let mut deleted_propositions: u64 = 0;
        for entity_id in target_entities {
            match entity_id {
                EntityID::Concept(_) => {
                    // ignore
                }
                EntityID::Proposition(id, predicate) => {
                    if let Ok(mut proposition) = self
                        .try_get_proposition_with(&ctx.cache, id, |prop| Ok(prop.clone()))
                        .await
                    {
                        // Remove specified predicates
                        proposition.predicates.remove(&predicate);
                        proposition.properties.remove(&predicate);

                        // If no predicates left, delete the proposition
                        if proposition.predicates.is_empty() {
                            let _ = self.propositions.remove(id).await;
                        } else {
                            // Otherwise, update the proposition with remaining predicates
                            if self
                                .propositions
                                .update(
                                    id,
                                    BTreeMap::from([
                                        ("predicates".to_string(), proposition.predicates.into()),
                                        ("properties".to_string(), proposition.properties.into()),
                                    ]),
                                )
                                .await
                                .is_ok()
                            {
                                deleted_propositions += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(json!({
            "deleted_propositions": deleted_propositions
        }))
    }

    async fn execute_delete_concepts(
        &self,
        target: String,
        where_clauses: Vec<WhereClause>,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        if dry_run {
            return Ok(json!({
                "deleted_propositions": 0,
                "deleted_concepts": 0
            }));
        }
        let mut ctx = QueryContext::default();
        for clause in where_clauses {
            self.execute_where_clause(&mut ctx, clause).await?;
        }

        let target_entities = ctx.entities.get(&target).cloned().ok_or_else(|| {
            KipError::InvalidCommand(format!("Target term '{}' not found in context", target))
        })?;
        let mut deleted_propositions: u64 = 0;
        let mut deleted_concepts: u64 = 0;
        for entity_id in target_entities {
            match &entity_id {
                EntityID::Concept(id) => {
                    let mut propositions_ids: BTreeSet<u64> = BTreeSet::new();
                    let eid: Fv = entity_id.to_string().into();
                    if let Ok(ids) = self
                        .propositions
                        .query_ids(
                            Filter::Or(vec![
                                Box::new(Filter::Field((
                                    "subject".to_string(),
                                    RangeQuery::Eq(eid.clone()),
                                ))),
                                Box::new(Filter::Field((
                                    "object".to_string(),
                                    RangeQuery::Eq(eid),
                                ))),
                            ]),
                            None,
                        )
                        .await
                    {
                        propositions_ids.extend(ids);
                    }

                    deleted_propositions += propositions_ids.len() as u64;

                    for id in propositions_ids {
                        let _ = self.propositions.remove(id).await;
                    }

                    deleted_concepts += 1;
                    let _ = self.concepts.remove(*id).await;
                }
                EntityID::Proposition(_, _) => {
                    // ignore
                }
            }
        }

        Ok(json!({
            "deleted_propositions": deleted_propositions,
            "deleted_concepts": deleted_concepts
        }))
    }

    async fn upsert_concept(
        &self,
        pk: ConceptPK,
        attributes: Map<String, Json>,
        metadata: Map<String, Json>,
    ) -> Result<EntityID, DBError> {
        match pk {
            ConceptPK::ID(id) => {
                self.update_concept(id, attributes, metadata).await?;
                Ok(EntityID::Concept(id))
            }
            ConceptPK::Object { r#type, name } => {
                let virtual_name = virtual_field_name(&["type", "name"]);
                let virtual_val = virtual_field_value(&[
                    Some(&Fv::Text(r#type.clone())),
                    Some(&Fv::Text(name.clone())),
                ])
                .unwrap();
                let ids = self
                    .concepts
                    .query_ids(
                        Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                        None,
                    )
                    .await?;

                if let Some(id) = ids.first() {
                    self.update_concept(*id, attributes, metadata).await?;
                    return Ok(EntityID::Concept(*id));
                }

                let concept = Concept {
                    _id: 0, // Will be set by the database
                    r#type,
                    name,
                    attributes,
                    metadata,
                };
                let id = self.concepts.add_from(&concept).await?;
                Ok(EntityID::Concept(id))
            }
        }
    }

    async fn upsert_proposition(
        &self,
        pk: PropositionPK,
        attributes: Map<String, Json>,
        metadata: Map<String, Json>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<EntityID, DBError> {
        match pk {
            PropositionPK::ID(id, predicate) => {
                self.update_proposition(id, predicate.clone(), attributes, metadata)
                    .await?;
                Ok(EntityID::Proposition(id, predicate))
            }
            PropositionPK::Object {
                subject,
                predicate,
                object,
            } => {
                // Convert EntityPK to EntityID for searching
                let subject = self.resolve_entity_id(subject.as_ref(), cached_pks).await?;
                let object = self.resolve_entity_id(object.as_ref(), cached_pks).await?;

                let virtual_name = virtual_field_name(&["subject", "object"]);
                let virtual_val = virtual_field_value(&[
                    Some(&Fv::Text(subject.to_string())),
                    Some(&Fv::Text(object.to_string())),
                ])
                .unwrap();

                let ids = self
                    .propositions
                    .query_ids(
                        Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                        None,
                    )
                    .await?;

                if let Some(id) = ids.first() {
                    // Proposition exists, update it
                    self.update_proposition(*id, predicate.clone(), attributes, metadata)
                        .await?;
                    return Ok(EntityID::Proposition(*id, predicate));
                }

                // Create new proposition
                let predicates = BTreeSet::from([predicate.clone()]);
                let properties = BTreeMap::from([(
                    predicate.clone(),
                    Properties {
                        attributes,
                        metadata,
                    },
                )]);

                let proposition = Proposition {
                    _id: 0, // Will be set by the database
                    subject,
                    object,
                    predicates,
                    properties,
                };

                let id = self.propositions.add_from(&proposition).await?;
                Ok(EntityID::Proposition(id, predicate))
            }
        }
    }

    async fn update_concept(
        &self,
        id: u64,
        attributes: Map<String, Json>,
        metadata: Map<String, Json>,
    ) -> Result<(), DBError> {
        if !self.concepts.contains(id) {
            return Err(DBError::NotFound {
                name: "concept node".to_string(),
                path: ConceptPK::ID(id).to_string(),
                source: "CognitiveNexus::update_concept".into(),
            });
        }

        // nothing to update
        if attributes.is_empty() && metadata.is_empty() {
            return Ok(());
        }

        let concept: Concept = self.concepts.get_as(id).await?;
        let mut update_fields: BTreeMap<String, Fv> = BTreeMap::new();
        if !attributes.is_empty() {
            let mut fv = concept.attributes;
            fv.extend(attributes);
            update_fields.insert("attributes".to_string(), fv.into());
        }
        if !metadata.is_empty() {
            let mut fv = concept.metadata;
            fv.extend(metadata);
            update_fields.insert("metadata".to_string(), fv.into());
        }
        self.concepts.update(id, update_fields).await?;

        Ok(())
    }

    async fn update_proposition(
        &self,
        id: u64,
        predicate: String,
        attributes: Map<String, Json>,
        metadata: Map<String, Json>,
    ) -> Result<(), DBError> {
        if !self.propositions.contains(id) {
            return Err(DBError::NotFound {
                name: "proposition link".to_string(),
                path: PropositionPK::ID(id, predicate).to_string(),
                source: "CognitiveNexus::update_proposition".into(),
            });
        }

        let proposition: Proposition = self.propositions.get_as(id).await?;
        if proposition.predicates.contains(&predicate)
            && attributes.is_empty()
            && metadata.is_empty()
        {
            return Ok(());
        }

        let mut update_fields: BTreeMap<String, Fv> = BTreeMap::new();
        let mut predicates = proposition.predicates;
        if predicates.insert(predicate.clone()) {
            update_fields.insert("predicates".to_string(), predicates.into());
        }

        if !attributes.is_empty() || !metadata.is_empty() {
            let mut properties = proposition.properties;
            let prop = properties.entry(predicate).or_default();
            prop.attributes.extend(attributes);
            prop.metadata.extend(metadata);

            update_fields.insert("properties".to_string(), properties.into());
        }

        self.propositions.update(id, update_fields).await?;

        Ok(())
    }

    async fn find_propositions(
        &self,
        cache: &QueryCache,
        node: &EntityID,
        predicate: &str,
        reverse: bool,
    ) -> Result<Vec<(EntityID, EntityID)>, KipError> {
        let ids = self
            .propositions
            .query_ids(
                Filter::Field((
                    if reverse {
                        "object".to_string()
                    } else {
                        "subject".to_string()
                    },
                    RangeQuery::Eq(Fv::Text(node.to_string())),
                )),
                None,
            )
            .await
            .map_err(|e| KipError::Execution(format!("Failed to query propositions: {:?}", e)))?;

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let rt = self
                .try_get_proposition_with(cache, id, |proposition| {
                    if proposition.predicates.contains(predicate) {
                        Ok(Some((
                            EntityID::Proposition(id, predicate.to_string()),
                            if reverse {
                                proposition.subject.clone()
                            } else {
                                proposition.object.clone()
                            },
                        )))
                    } else {
                        Ok(None)
                    }
                })
                .await?;

            if let Some(rt) = rt {
                results.push(rt)
            }
        }

        Ok(results)
    }

    // 查询概念节点ID
    async fn query_concept_ids(&self, matcher: &ConceptMatcher) -> Result<Vec<u64>, KipError> {
        match matcher {
            ConceptMatcher::ID(id) => {
                let entity_id = EntityID::from_str(id).map_err(KipError::Parse)?;
                if let EntityID::Concept(concept_id) = entity_id {
                    Ok(vec![concept_id])
                } else {
                    Err(KipError::InvalidCommand(format!(
                        "Invalid concept node ID: {}",
                        id
                    )))
                }
            }
            ConceptMatcher::Type(type_name) => {
                let ids = self
                    .concepts
                    .query_ids(
                        Filter::Field((
                            "type".to_string(),
                            RangeQuery::Eq(Fv::Text(type_name.clone())),
                        )),
                        None,
                    )
                    .await
                    .map_err(|e| KipError::Execution(format!("{:?}", e)))?;
                Ok(ids)
            }
            ConceptMatcher::Name(name) => {
                let ids = self
                    .concepts
                    .query_ids(
                        Filter::Field(("name".to_string(), RangeQuery::Eq(Fv::Text(name.clone())))),
                        None,
                    )
                    .await
                    .map_err(|e| KipError::Execution(format!("{:?}", e)))?;
                Ok(ids)
            }
            ConceptMatcher::Object { r#type, name } => {
                let virtual_name = virtual_field_name(&["type", "name"]);
                let virtual_val = virtual_field_value(&[
                    Some(&Fv::Text(r#type.clone())),
                    Some(&Fv::Text(name.clone())),
                ])
                .unwrap();

                let ids = self
                    .concepts
                    .query_ids(
                        Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                        None,
                    )
                    .await
                    .map_err(|e| KipError::Execution(format!("{:?}", e)))?;
                Ok(ids)
            }
        }
    }

    async fn evaluate_filter_operand(
        &self,
        ctx: &mut QueryContext,
        operand: FilterOperand,
        bindings_snapshot: &mut HashMap<String, Vec<EntityID>>,
        bindings_cursor: &mut HashMap<String, EntityID>,
    ) -> Result<Option<Json>, KipError> {
        match operand {
            FilterOperand::Variable(dot_path) => {
                self.consume_bindings(&ctx.cache, dot_path, bindings_snapshot, bindings_cursor)
                    .await
            }
            FilterOperand::Literal(value) => Ok(Some(value.into())),
        }
    }

    async fn match_propositions(
        &self,
        ctx: &mut QueryContext,
        subject: TargetTerm,
        predicate: PredTerm,
        object: TargetTerm,
    ) -> Result<TargetEntities, KipError> {
        let subject_var = match &subject {
            TargetTerm::Variable(var) => Some(var.clone()),
            _ => None,
        };
        let predicate_var = match &predicate {
            PredTerm::Variable(var) => Some(var.clone()),
            _ => None,
        };
        let object_var = match &object {
            TargetTerm::Variable(var) => Some(var.clone()),
            _ => None,
        };

        let subjects = self.resolve_target_term_ids(ctx, subject).await?;
        let objects = self.resolve_target_term_ids(ctx, object).await?;

        let result = match (subjects, predicate, objects) {
            (
                subjects,
                PredTerm::MultiHop {
                    predicate,
                    min,
                    max,
                },
                objects,
            ) => {
                self.handle_multi_hop_matching(ctx, subjects, predicate, min, max, objects)
                    .await?
            }
            (TargetEntities::IDs(subject_ids), predicate, TargetEntities::IDs(object_ids)) => {
                self.handle_subject_object_ids_matching(ctx, subject_ids, object_ids, predicate)
                    .await?
            }
            (TargetEntities::IDs(subject_ids), predicate, TargetEntities::AnyPropositions) => {
                self.handle_subject_ids_any_matching(ctx, subject_ids, predicate, true)
                    .await?
            }
            (TargetEntities::IDs(subject_ids), predicate, TargetEntities::Any) => {
                self.handle_subject_ids_any_matching(ctx, subject_ids, predicate, false)
                    .await?
            }
            (TargetEntities::AnyPropositions, predicate, TargetEntities::IDs(object_ids)) => {
                self.handle_any_to_object_ids_matching(ctx, object_ids, predicate, true)
                    .await?
            }
            (TargetEntities::Any, predicate, TargetEntities::IDs(object_ids)) => {
                self.handle_any_to_object_ids_matching(ctx, object_ids, predicate, false)
                    .await?
            }
            (_, predicate, _) => {
                if matches!(&predicate, PredTerm::Variable(_)) {
                    return Ok(TargetEntities::AnyPropositions);
                }

                self.handle_predicate_matching(ctx, predicate).await?
            }
        };

        if let Some(var) = subject_var {
            ctx.entities.insert(var, result.matched_subjects.clone());
        }
        if let Some(var) = predicate_var {
            ctx.predicates
                .insert(var, result.matched_predicates.clone());
        }
        if let Some(var) = object_var {
            ctx.entities.insert(var, result.matched_objects.clone());
        }

        Ok(TargetEntities::IDs(
            result.matched_propositions.into_iter().collect(),
        ))
    }

    async fn resolve_result(
        &self,
        cache: &QueryCache,
        bindings: &HashMap<String, Vec<EntityID>>,
        dot_path: &DotPathVar,
        order_by: &[OrderByCondition],
    ) -> Result<Vec<Json>, KipError> {
        let ids = bindings.get(&dot_path.var).ok_or_else(|| {
            KipError::InvalidCommand(format!("Unbound variable: '{}'", dot_path.var))
        })?;

        let mut result = Vec::with_capacity(ids.len());
        for id in ids {
            match id {
                EntityID::Concept(id) => {
                    let rt = self
                        .try_get_concept_with(cache, *id, |concept| {
                            extract_concept_field_value(concept, &[])
                        })
                        .await?;
                    result.push(rt);
                }
                EntityID::Proposition(id, predicate) => {
                    let rt = self
                        .try_get_proposition_with(cache, *id, |prop| {
                            extract_proposition_field_value(prop, predicate, &[])
                        })
                        .await?;
                    result.push(rt);
                }
            };
        }

        result = apply_order_by(result, &dot_path.var, order_by);
        if dot_path.path.is_empty() {
            return Ok(result);
        }

        let path = format!("/{}", dot_path.path.join("/"));
        Ok(result
            .into_iter()
            .map(|val| val.pointer(&path).cloned().unwrap_or(Json::Null))
            .collect())
    }

    // 解析目标项为实体ID列表
    async fn resolve_target_term_ids(
        &self,
        ctx: &mut QueryContext,
        target: TargetTerm,
    ) -> Result<TargetEntities, KipError> {
        match target {
            TargetTerm::Variable(var) => {
                if let Some(ids) = ctx.entities.get(&var) {
                    Ok(TargetEntities::IDs(ids.iter().cloned().collect()))
                } else {
                    Ok(TargetEntities::Any)
                }
            }
            TargetTerm::Concept(concept_matcher) => {
                let ids = self.query_concept_ids(&concept_matcher).await?;
                Ok(TargetEntities::IDs(
                    ids.into_iter().map(EntityID::Concept).collect(),
                ))
            }
            TargetTerm::Proposition(proposition_matcher) => {
                match *proposition_matcher {
                    PropositionMatcher::ID(id) => {
                        let entity_id = EntityID::from_str(&id).map_err(KipError::Parse)?;
                        if !matches!(entity_id, EntityID::Proposition(_, _)) {
                            return Err(KipError::InvalidCommand(format!(
                                "Invalid proposition link ID: {id:?}"
                            )));
                        }
                        Ok(TargetEntities::IDs(vec![entity_id]))
                    }
                    PropositionMatcher::Object {
                        subject: TargetTerm::Variable(_),
                        predicate: PredTerm::Variable(_),
                        object: TargetTerm::Variable(_),
                    } => Ok(TargetEntities::AnyPropositions),
                    PropositionMatcher::Object {
                        subject,
                        predicate,
                        object,
                    } => {
                        // 递归查询命题
                        let result =
                            Box::pin(self.match_propositions(ctx, subject, predicate, object))
                                .await?;
                        Ok(result)
                    }
                }
            }
        }
    }

    async fn consume_bindings(
        &self,
        cache: &QueryCache,
        dot_path: DotPathVar,
        bindings_snapshot: &mut HashMap<String, Vec<EntityID>>,
        bindings_cursor: &mut HashMap<String, EntityID>,
    ) -> Result<Option<Json>, KipError> {
        let entity_id = match bindings_cursor.get(&dot_path.var) {
            Some(id) => id.clone(),
            None => {
                // 如果当前游标没有绑定，尝试从快照中获取
                let ids = bindings_snapshot.get_mut(&dot_path.var).ok_or_else(|| {
                    KipError::InvalidCommand(format!("Unbound variable: '{}'", dot_path.var))
                })?;
                let id = match ids.pop() {
                    Some(id) => id,
                    None => return Ok(None), // 如果没有更多ID，返回None
                };

                if ids.is_empty() {
                    bindings_snapshot.remove(&dot_path.var);
                }

                bindings_cursor.insert(dot_path.var.clone(), id.clone());
                id
            }
        };

        match entity_id {
            EntityID::Concept(id) => {
                let rt = self
                    .try_get_concept_with(cache, id, |concept| {
                        extract_concept_field_value(concept, &dot_path.path)
                    })
                    .await?;

                Ok(Some(rt))
            }
            EntityID::Proposition(id, predicate) => {
                let rt = self
                    .try_get_proposition_with(cache, id, |proposition| {
                        extract_proposition_field_value(proposition, &predicate, &dot_path.path)
                    })
                    .await?;

                Ok(Some(rt))
            }
        }
    }

    async fn try_get_concept_with<F, R>(
        &self,
        cache: &QueryCache,
        id: u64,
        f: F,
    ) -> Result<R, KipError>
    where
        F: FnOnce(&Concept) -> Result<R, KipError>,
    {
        if let Some(concept) = cache.concepts.read().get(&id) {
            return f(concept);
        }
        let concept: Concept = self
            .concepts
            .get_as(id)
            .await
            .map_err(|e| KipError::Execution(format!("{:?}", e)))?;
        let rt = f(&concept)?;
        cache.concepts.write().insert(id, concept);
        Ok(rt)
    }

    async fn try_get_proposition_with<F, R>(
        &self,
        cache: &QueryCache,
        id: u64,
        f: F,
    ) -> Result<R, KipError>
    where
        F: FnOnce(&Proposition) -> Result<R, KipError>,
    {
        if let Some(proposition) = cache.propositions.read().get(&id) {
            return f(proposition);
        }
        let proposition: Proposition = self
            .propositions
            .get_as(id)
            .await
            .map_err(|e| KipError::Execution(format!("{:?}", e)))?;
        let rt = f(&proposition)?;
        cache.propositions.write().insert(id, proposition);
        Ok(rt)
    }

    async fn evaluate_filter_expression(
        &self,
        ctx: &mut QueryContext,
        expr: FilterExpression,
        bindings_snapshot: &mut HashMap<String, Vec<EntityID>>,
        bindings_cursor: &mut HashMap<String, EntityID>,
    ) -> Result<Option<bool>, KipError> {
        match expr {
            FilterExpression::Comparison {
                left,
                operator,
                right,
            } => {
                let left_val = match self
                    .evaluate_filter_operand(ctx, left, bindings_snapshot, bindings_cursor)
                    .await?
                {
                    Some(val) => val,
                    None => return Ok(None),
                };
                let right_val = match self
                    .evaluate_filter_operand(ctx, right, bindings_snapshot, bindings_cursor)
                    .await?
                {
                    Some(val) => val,
                    None => return Ok(None),
                };

                Ok(Some(operator.compare(&left_val, &right_val)))
            }
            FilterExpression::Logical {
                left,
                operator,
                right,
            } => {
                let left_result = match Box::pin(self.evaluate_filter_expression(
                    ctx,
                    *left,
                    bindings_snapshot,
                    bindings_cursor,
                ))
                .await?
                {
                    Some(result) => result,
                    None => return Ok(None),
                };
                let right_result = match Box::pin(self.evaluate_filter_expression(
                    ctx,
                    *right,
                    bindings_snapshot,
                    bindings_cursor,
                ))
                .await?
                {
                    Some(result) => result,
                    None => return Ok(None),
                };

                Ok(match operator {
                    LogicalOperator::And => Some(left_result && right_result),
                    LogicalOperator::Or => Some(left_result || right_result),
                })
            }
            FilterExpression::Not(expr) => {
                let result = Box::pin(self.evaluate_filter_expression(
                    ctx,
                    *expr,
                    bindings_snapshot,
                    bindings_cursor,
                ))
                .await?;
                Ok(result.map(|r| !r))
            }
            FilterExpression::Function { func, args } => {
                self.evaluate_filter_function(ctx, func, args, bindings_snapshot, bindings_cursor)
                    .await
            }
        }
    }

    async fn evaluate_filter_function(
        &self,
        ctx: &mut QueryContext,
        func: FilterFunction,
        mut args: Vec<FilterOperand>,
        bindings_snapshot: &mut HashMap<String, Vec<EntityID>>,
        bindings_cursor: &mut HashMap<String, EntityID>,
    ) -> Result<Option<bool>, KipError> {
        if args.len() != 2 {
            return Err(KipError::InvalidCommand(
                "Filter functions require exactly 2 arguments".to_string(),
            ));
        }

        let pattern_arg = args.pop().unwrap();
        let str_arg = args.pop().unwrap();
        let str_val = match self
            .evaluate_filter_operand(ctx, str_arg, bindings_snapshot, bindings_cursor)
            .await?
        {
            Some(val) => val,
            None => return Ok(None),
        };
        let pattern_val = match self
            .evaluate_filter_operand(ctx, pattern_arg, bindings_snapshot, bindings_cursor)
            .await?
        {
            Some(val) => val,
            None => return Ok(None),
        };

        let string = str_val.as_str().unwrap_or("");
        let pattern = pattern_val.as_str().unwrap_or("");

        match func {
            FilterFunction::Contains => Ok(Some(string.contains(pattern))),
            FilterFunction::StartsWith => Ok(Some(string.starts_with(pattern))),
            FilterFunction::EndsWith => Ok(Some(string.ends_with(pattern))),
            FilterFunction::Regex => {
                // 简单的正则表达式匹配
                let rt = regex::Regex::new(pattern)
                    .map_err(|e| KipError::InvalidCommand(format!("Invalid regex: {}", e)))?
                    .is_match(string);
                Ok(Some(rt))
            }
        }
    }

    fn check_target_term_for_kml(
        &self,
        target: &TargetTerm,
        handle_map: &HashMap<String, EntityID>,
    ) -> Result<(), KipError> {
        match target {
            TargetTerm::Variable(handle) => {
                if !handle_map.contains_key(handle) {
                    return Err(KipError::InvalidCommand(format!(
                        "Undefined handle: {handle}"
                    )));
                }
            }
            TargetTerm::Concept(concept_matcher) => {
                let _ = ConceptPK::try_from(concept_matcher.clone())?;
            }
            TargetTerm::Proposition(proposition_matcher) => {
                let _ = PropositionPK::try_from(*proposition_matcher.clone())?;
            }
        }

        Ok(())
    }

    async fn resolve_target_term(
        &self,
        target: TargetTerm,
        handle_map: &HashMap<String, EntityID>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<EntityID, KipError> {
        match target {
            TargetTerm::Variable(handle) => handle_map
                .get(&handle)
                .cloned()
                .ok_or_else(|| KipError::InvalidCommand(format!("Undefined handle: {handle}"))),
            TargetTerm::Concept(concept_matcher) => {
                let concept_pk = ConceptPK::try_from(concept_matcher)?;
                self.resolve_entity_id(&EntityPK::Concept(concept_pk), cached_pks)
                    .await
                    .map_err(|err| KipError::Execution(format!("{err:?}")))
            }
            TargetTerm::Proposition(proposition_matcher) => {
                let proposition_pk = PropositionPK::try_from(*proposition_matcher)?;
                self.resolve_entity_id(&EntityPK::Proposition(proposition_pk), cached_pks)
                    .await
                    .map_err(|err| KipError::Execution(format!("{err:?}")))
            }
        }
    }

    // Helper method to resolve EntityPK to EntityID
    async fn resolve_entity_id(
        &self,
        entity_pk: &EntityPK,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<EntityID, DBError> {
        {
            if let Some(id) = cached_pks.get(entity_pk) {
                return Ok(id.clone());
            }
        }

        let id = match entity_pk {
            EntityPK::Concept(concept_pk) => match concept_pk {
                ConceptPK::ID(id) => Ok(EntityID::Concept(*id)),
                ConceptPK::Object { r#type, name } => {
                    let virtual_name = virtual_field_name(&["type", "name"]);
                    let virtual_val = virtual_field_value(&[
                        Some(&Fv::Text(r#type.clone())),
                        Some(&Fv::Text(name.clone())),
                    ])
                    .unwrap();

                    let ids = self
                        .concepts
                        .query_ids(
                            Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                            None,
                        )
                        .await?;

                    if let Some(id) = ids.first() {
                        Ok(EntityID::Concept(*id))
                    } else {
                        Err(DBError::NotFound {
                            name: "concept node".to_string(),
                            path: concept_pk.to_string(),
                            source: "CognitiveNexus::resolve_entity_id".into(),
                        })
                    }
                }
            },
            EntityPK::Proposition(proposition_pk) => match proposition_pk {
                PropositionPK::ID(id, predicate) => {
                    Ok(EntityID::Proposition(*id, predicate.clone()))
                }
                PropositionPK::Object {
                    subject,
                    predicate,
                    object,
                } => {
                    // 使用 Box::pin 来处理递归调用
                    let subject_id =
                        Box::pin(self.resolve_entity_id(subject.as_ref(), cached_pks)).await?;

                    let object_id =
                        Box::pin(self.resolve_entity_id(object.as_ref(), cached_pks)).await?;

                    let virtual_name = virtual_field_name(&["subject", "object"]);
                    let virtual_val = virtual_field_value(&[
                        Some(&Fv::Text(subject_id.to_string())),
                        Some(&Fv::Text(object_id.to_string())),
                    ])
                    .unwrap();

                    let ids = self
                        .propositions
                        .query_ids(
                            Filter::Field((virtual_name, RangeQuery::Eq(virtual_val))),
                            None,
                        )
                        .await?;

                    if let Some(id) = ids.first() {
                        Ok(EntityID::Proposition(*id, predicate.clone()))
                    } else {
                        Err(DBError::NotFound {
                            name: "proposition link".to_string(),
                            path: proposition_pk.to_string(),
                            source: "CognitiveNexus::resolve_entity_id".into(),
                        })
                    }
                }
            },
        }?;

        cached_pks.insert(entity_pk.clone(), id.clone());
        Ok(id)
    }
}
