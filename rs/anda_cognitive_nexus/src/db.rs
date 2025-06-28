use anda_db::{
    collection::{Collection, CollectionConfig},
    database::{AndaDB, DBMetadata},
    error::DBError,
    index::BTree,
    query::{Filter, Query, RangeQuery},
};
use anda_db_schema::{Document, Fv};
use anda_db_tfs::jieba_tokenizer;
use anda_kip::{Command, Executor, Json, KipError, KmlStatement, KqlQuery, MetaCommand};
use async_trait::async_trait;
use parking_lot::RwLock;
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

    async fn execute_kml(&self, _command: KmlStatement, _dry_run: bool) -> Result<Json, KipError> {
        unimplemented!("execute_kml is not implemented yet");
    }

    async fn execute_meta(&self, _command: MetaCommand, _dry_run: bool) -> Result<Json, KipError> {
        unimplemented!("execute_meta is not implemented yet");
    }

    async fn upsert_concept(
        &self,
        pk: ConceptPK,
        attributes: Option<BTreeMap<String, Json>>,
        metadata: Option<BTreeMap<String, Json>>,
    ) -> Result<EntityID, DBError> {
        let attributes = attributes.unwrap_or_default();
        let metadata = metadata.unwrap_or_default();
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
                    .search_ids(Query {
                        filter: Some(Filter::Field((virtual_name, RangeQuery::Eq(virtual_val)))),
                        ..Default::default()
                    })
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
        attributes: Option<BTreeMap<String, Json>>,
        metadata: Option<BTreeMap<String, Json>>,
    ) -> Result<EntityID, DBError> {
        let attributes = attributes.unwrap_or_default();
        let metadata = metadata.unwrap_or_default();
        let cached_pks: RwLock<HashMap<EntityPK, EntityID>> = RwLock::new(HashMap::new());

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
                let subject = self
                    .resolve_entity_id(subject.as_ref(), &cached_pks)
                    .await?;
                let object = self.resolve_entity_id(object.as_ref(), &cached_pks).await?;

                let virtual_name = BTree::virtual_field_name(&["subject", "object"]);
                let virtual_val = Document::virtual_field_value(&[
                    Some(&Fv::Text(subject.to_string())),
                    Some(&Fv::Text(object.to_string())),
                ])
                .unwrap();

                let ids = self
                    .propositions
                    .search_ids(Query {
                        filter: Some(Filter::Field((virtual_name, RangeQuery::Eq(virtual_val)))),
                        ..Default::default()
                    })
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
                path: id.to_string(),
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
                path: id.to_string(),
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

    // Helper method to resolve EntityPK to EntityID
    async fn resolve_entity_id(
        &self,
        entity_pk: &EntityPK,
        cached_pks: &RwLock<HashMap<EntityPK, EntityID>>,
    ) -> Result<EntityID, DBError> {
        {
            if let Some(id) = cached_pks.read().get(entity_pk) {
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
                        .search_ids(Query {
                            filter: Some(Filter::Field((
                                virtual_name,
                                RangeQuery::Eq(virtual_val),
                            ))),
                            ..Default::default()
                        })
                        .await?;

                    if let Some(id) = ids.first() {
                        Ok(EntityID::Concept(*id))
                    } else {
                        Err(DBError::NotFound {
                            name: "concept node".to_string(),
                            path: format!("type: {}, name: {}", r#type, name),
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
                    let object_future =
                        Box::pin(self.resolve_entity_id(object.as_ref(), cached_pks));

                    let subject_id = subject_future.await?;
                    let object_id = object_future.await?;

                    let virtual_name = BTree::virtual_field_name(&["subject", "object"]);
                    let virtual_val = Document::virtual_field_value(&[
                        Some(&Fv::Text(subject_id.to_string())),
                        Some(&Fv::Text(object_id.to_string())),
                    ])
                    .unwrap();

                    let ids = self
                        .propositions
                        .search_ids(Query {
                            filter: Some(Filter::Field((
                                virtual_name,
                                RangeQuery::Eq(virtual_val),
                            ))),
                            ..Default::default()
                        })
                        .await?;

                    if let Some(id) = ids.first() {
                        Ok(EntityID::Proposition(*id, predicate.clone()))
                    } else {
                        Err(DBError::NotFound {
                            name: "proposition link".to_string(),
                            path: format!(
                                "subject: {}, predicate: {}, object: {}",
                                subject_id, object_id, predicate
                            ),
                            source: "CognitiveNexus::resolve_entity_id".into(),
                        })
                    }
                }
            },
        }?;

        cached_pks.write().insert(entity_pk.clone(), id.clone());
        Ok(id)
    }
}
