use anda_db::{
    collection::{Collection, CollectionConfig},
    database::{AndaDB, DBMetadata},
    error::DBError,
    index::BTree,
    query::{Filter, RangeQuery},
};
use anda_db_schema::{Document, Fv};
use anda_db_tfs::jieba_tokenizer;
use anda_kip::{
    Command, ConceptBlock, ConceptMatcher, DeleteStatement, Executor, Json, KipError, KmlStatement,
    KqlQuery, MetaCommand, PropositionBlock, SetProposition, TargetTerm, UpsertBlock, UpsertItem,
};
use async_trait::async_trait;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
    time::Duration,
};
use tokio_util::sync::CancellationToken;

use crate::{entity::*, types::*};

#[derive(Clone, Debug)]
pub struct CognitiveNexus {
    db: Arc<AndaDB>,
    concepts: Arc<Collection>,
    propositions: Arc<Collection>,
}

#[async_trait]
impl Executor for CognitiveNexus {
    async fn execute(&self, command: Command, dry_run: bool) -> Result<Json, KipError> {
        match command {
            Command::Kql(command) => self.execute_kql(command, dry_run).await,
            Command::Kml(command) => self.execute_kml(command, dry_run).await,
            Command::Meta(command) => self.execute_meta(command, dry_run).await,
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
                    // TODO: TFS index

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
                    // TODO: TFS index

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

    pub fn name(&self) -> &str {
        &self.db.name()
    }

    pub fn metadata(&self) -> DBMetadata {
        self.db.metadata()
    }

    pub async fn close(&self) -> Result<(), DBError> {
        self.db.close().await
    }

    pub async fn auto_flush(&self, cancel_token: CancellationToken, interval: Duration) {
        self.db.auto_flush(cancel_token, interval).await;
    }

    async fn execute_kql(&self, _command: KqlQuery, _dry_run: bool) -> Result<Json, KipError> {
        unimplemented!("execute_kql is not implemented yet");
    }

    async fn execute_kml(&self, command: KmlStatement, dry_run: bool) -> Result<Json, KipError> {
        match command {
            KmlStatement::Upsert(upsert_block) => self.execute_upsert(upsert_block, dry_run).await,
            KmlStatement::Delete(delete_statement) => {
                self.execute_delete(delete_statement, dry_run).await
            }
        }
    }

    async fn execute_meta(&self, _command: MetaCommand, _dry_run: bool) -> Result<Json, KipError> {
        unimplemented!("execute_meta is not implemented yet");
    }

    async fn execute_upsert(
        &self,
        upsert_block: UpsertBlock,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        let mut handle_map: HashMap<String, EntityID> = HashMap::new();
        let mut cached_pks: HashMap<EntityPK, EntityID> = HashMap::new();
        let mut concept_nodes: Vec<EntityID> = Vec::new();
        let mut proposition_links: Vec<EntityID> = Vec::new();
        let default_metadata: BTreeMap<String, Json> = upsert_block
            .metadata
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
            .unwrap_or_default();

        for item in upsert_block.items {
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

        Ok(json!({
            "concept_nodes": concept_nodes,
            "proposition_links": proposition_links,
        }))
    }

    async fn execute_concept_block(
        &self,
        concept_block: ConceptBlock,
        default_metadata: &BTreeMap<String, Json>,
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
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
            .unwrap_or_default();
        let metadata = concept_block
            .metadata
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
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
        default_metadata: &BTreeMap<String, Json>,
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
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
            .unwrap_or_default();
        let metadata = proposition_block
            .metadata
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
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
        default_metadata: &BTreeMap<String, Json>,
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
            .map(|val| val.into_iter().map(|(k, v)| (k, v.into())).collect())
            .unwrap_or_else(|| default_metadata.clone());

        let entity_id = self
            .upsert_proposition(proposition_pk, BTreeMap::new(), metadata, cached_pks)
            .await
            .map_err(|err| KipError::Execution(format!("{err:?}")))?;

        Ok(entity_id)
    }

    async fn execute_delete(
        &self,
        delete_statement: DeleteStatement,
        dry_run: bool,
    ) -> Result<Json, KipError> {
        if dry_run {
            return Ok(Json::Object(serde_json::Map::from_iter([
                ("operation".to_string(), Json::String("delete".to_string())),
                ("dry_run".to_string(), Json::Bool(true)),
            ])));
        }

        match delete_statement {
            DeleteStatement::DeleteAttributes {
                attributes,
                target: _,
                where_clauses: _,
            } => {
                // 注意：当前实现需要WHERE子句的KQL查询支持
                // 这里提供一个简化版本，实际需要实现KQL查询来获取目标实体
                return Err(KipError::NotImplemented(
                    "DELETE ATTRIBUTES requires KQL query support".to_string(),
                ));
            }
            DeleteStatement::DeletePropositions {
                target: _,
                where_clauses: _,
            } => {
                return Err(KipError::NotImplemented(
                    "DELETE PROPOSITIONS requires KQL query support".to_string(),
                ));
            }
            DeleteStatement::DeleteConcept {
                target: _,
                where_clauses: _,
            } => {
                return Err(KipError::NotImplemented(
                    "DELETE CONCEPT requires KQL query support".to_string(),
                ));
            }
        }
    }

    async fn upsert_concept(
        &self,
        pk: ConceptPK,
        attributes: BTreeMap<String, Json>,
        metadata: BTreeMap<String, Json>,
    ) -> Result<EntityID, DBError> {
        match pk {
            ConceptPK::ID(id) => {
                self.update_concept(id, attributes, metadata).await?;
                Ok(EntityID::Concept(id))
            }
            ConceptPK::Object { r#type, name } => {
                let virtual_name = BTree::virtual_field_name(&["type", "name"]);
                let virtual_val = Document::virtual_field_value(&[
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
        attributes: BTreeMap<String, Json>,
        metadata: BTreeMap<String, Json>,
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

                let virtual_name = BTree::virtual_field_name(&["subject", "object"]);
                let virtual_val = Document::virtual_field_value(&[
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
        attributes: BTreeMap<String, Json>,
        metadata: BTreeMap<String, Json>,
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
        attributes: BTreeMap<String, Json>,
        metadata: BTreeMap<String, Json>,
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

    async fn delete_attributes(
        &self,
        pks: Vec<EntityPK>,
        attrs: Vec<String>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<(), DBError> {
        for pk in pks {
            let id = self.resolve_entity_id(&pk, cached_pks).await?;
            match id {
                EntityID::Concept(id) => {
                    if let Ok(mut concept) = self.concepts.get_as::<Concept>(id).await {
                        let length = concept.attributes.len();
                        for attr in &attrs {
                            concept.attributes.remove(attr);
                        }
                        if concept.attributes.len() < length {
                            self.concepts
                                .update(
                                    id,
                                    BTreeMap::from([(
                                        "attributes".to_string(),
                                        concept.attributes.into(),
                                    )]),
                                )
                                .await?;
                        }
                    }
                }

                EntityID::Proposition(id, predicate) => {
                    if let Ok(mut proposition) = self.propositions.get_as::<Proposition>(id).await {
                        if let Some(prop) = proposition.properties.get_mut(&predicate) {
                            let length = prop.attributes.len();
                            for attr in &attrs {
                                prop.attributes.remove(attr);
                            }

                            if prop.attributes.len() < length {
                                self.propositions
                                    .update(
                                        id,
                                        BTreeMap::from([(
                                            "properties".to_string(),
                                            proposition.properties.into(),
                                        )]),
                                    )
                                    .await?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn delete_metadata(
        &self,
        pks: Vec<EntityPK>,
        keys: Vec<String>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<(), DBError> {
        for pk in pks {
            let id = self.resolve_entity_id(&pk, cached_pks).await?;
            match id {
                EntityID::Concept(id) => {
                    if let Ok(mut concept) = self.concepts.get_as::<Concept>(id).await {
                        let length = concept.metadata.len();
                        for key in &keys {
                            concept.metadata.remove(key);
                        }
                        if concept.metadata.len() < length {
                            self.concepts
                                .update(
                                    id,
                                    BTreeMap::from([(
                                        "metadata".to_string(),
                                        concept.metadata.into(),
                                    )]),
                                )
                                .await?;
                        }
                    }
                }

                EntityID::Proposition(id, predicate) => {
                    if let Ok(mut proposition) = self.propositions.get_as::<Proposition>(id).await {
                        if let Some(prop) = proposition.properties.get_mut(&predicate) {
                            let length = prop.metadata.len();
                            for key in &keys {
                                prop.metadata.remove(key);
                            }

                            if prop.metadata.len() < length {
                                self.propositions
                                    .update(
                                        id,
                                        BTreeMap::from([(
                                            "properties".to_string(),
                                            proposition.properties.into(),
                                        )]),
                                    )
                                    .await?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn delete_propositions(
        &self,
        pks: Vec<PropositionPK>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<(), DBError> {
        // id -> predicates
        let mut proposition_ids: HashMap<u64, BTreeSet<String>> = HashMap::new();
        for pk in pks {
            if let Ok(EntityID::Proposition(id, pred)) = self
                .resolve_entity_id(&EntityPK::Proposition(pk), cached_pks)
                .await
            {
                proposition_ids.entry(id).or_default().insert(pred.clone());
            }
        }

        for (id, predicates) in proposition_ids {
            if let Ok(mut proposition) = self.propositions.get_as::<Proposition>(id).await {
                // Remove specified predicates
                for pred in predicates {
                    proposition.predicates.remove(&pred);
                    proposition.properties.remove(&pred);
                }

                // If no predicates left, delete the proposition
                if proposition.predicates.is_empty() {
                    let _ = self.propositions.remove(id).await;
                } else {
                    // Otherwise, update the proposition with remaining predicates
                    self.propositions
                        .update(
                            id,
                            BTreeMap::from([
                                ("predicates".to_string(), proposition.predicates.into()),
                                ("properties".to_string(), proposition.properties.into()),
                            ]),
                        )
                        .await?;
                }
            }
        }
        Ok(())
    }

    async fn delete_concepts(
        &self,
        pks: Vec<ConceptPK>,
        cached_pks: &mut HashMap<EntityPK, EntityID>,
    ) -> Result<(), DBError> {
        let mut concepts_ids: BTreeSet<u64> = BTreeSet::new();
        let mut propositions_ids: BTreeSet<u64> = BTreeSet::new();
        for pk in pks {
            if let Ok(EntityID::Concept(id)) = self
                .resolve_entity_id(&EntityPK::Concept(pk), cached_pks)
                .await
            {
                concepts_ids.insert(id);
            }
        }

        for id in &concepts_ids {
            let eid: Fv = EntityID::Concept(*id).to_string().into();
            let ids = self
                .propositions
                .query_ids(
                    Filter::Or(vec![
                        Box::new(Filter::Field((
                            "subject".to_string(),
                            RangeQuery::Eq(eid.clone()),
                        ))),
                        Box::new(Filter::Field(("object".to_string(), RangeQuery::Eq(eid)))),
                    ]),
                    None,
                )
                .await?;
            propositions_ids.extend(ids);
        }

        for id in propositions_ids {
            let _ = self.propositions.remove(id).await;
        }

        for id in concepts_ids {
            let _ = self.concepts.remove(id).await;
        }

        Ok(())
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
                    let virtual_name = BTree::virtual_field_name(&["type", "name"]);
                    let virtual_val = Document::virtual_field_value(&[
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
                    let subject_future =
                        Box::pin(self.resolve_entity_id(subject.as_ref(), cached_pks));
                    let subject_id = subject_future.await?;

                    let object_future =
                        Box::pin(self.resolve_entity_id(object.as_ref(), cached_pks));
                    let object_id = object_future.await?;

                    let virtual_name = BTree::virtual_field_name(&["subject", "object"]);
                    let virtual_val = Document::virtual_field_value(&[
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
