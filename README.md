# Anda DB

[![Build Status](https://github.com/ldclabs/anda-db/actions/workflows/test.yml/badge.svg)](https://github.com/ldclabs/anda-db/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ldclabs/anda-db/blob/main/LICENSE)

**Anda DB** is a Rust library designed as a specialized database for AI Agents, focusing on knowledge memory. It supports multimodal data storage, full-text search, and vector similarity search, integrating seamlessly as a local database within AI Agent applications.

At its core, Anda DB powers the **[Anda Cognitive Nexus](https://github.com/ldclabs/anda-db/tree/main/rs/anda_cognitive_nexus)**, a powerful, persistent, and graph-based long-term memory system for AI Agents. It implements the **KIP (Knowledge Interaction Protocol)**, enabling AI to learn, reason, and evolve through a structured, auditable knowledge graph.

## Key Features

-   **Embedded Library:** Functions as a Rust library, not a standalone remote database service, enabling direct integration into AI Agent builds.
-   **Pluggable Storage Backend:** Leverages the [`object_store`](https://docs.rs/object_store) interface, supporting various backends like AWS S3, Google Cloud Storage, Azure Blob Storage, and the local filesystem.
-   **Encrypted & Resilient Storage:** Offers optional, transparent AES-256-GCM encryption for all data at rest, ensuring privacy and security, powered by [`anda_object_store`](https://docs.rs/anda_object_store).
-   **Flexible Schema & ORM:** A document-oriented design with a flexible schema and built-in ORM support via the `AndaDBSchema` derive macro.
-   **Multimodal Data:** Natively handles diverse data types including text, `bfloat16` vectors, binary data, JSON, and more.
-   **Advanced Indexing:**
    -   **BTree Index:** Enables precise matching, range queries, and multi-conditional logical queries on various data types.
    -   **BM25 Index:** Provides efficient, full-text search with customizable tokenizers (including Chinese via Jieba).
    -   **HNSW Index:** Offers high-performance approximate nearest neighbor (ANN) search for vector similarity.
-   **Hybrid Search:** Automatically combines and weights text (BM25) and vector (HNSW) search results using Reciprocal Rank Fusion (RRF) for comprehensive retrieval.
-   **Knowledge Graph Capabilities:** Implements the **KIP (Knowledge Interaction Protocol)** for building and querying a self-evolving knowledge graph of concepts and propositions.
-   **Incremental & Persistent:** Supports efficient incremental index updates and document deletions without requiring costly full rebuilds. The entire database state can be saved and loaded, ensuring data durability.

## Architecture

Anda DB is a modular project composed of several crates, each providing a specific piece of functionality:

-   [`anda_cognitive_nexus`](./rs/anda_cognitive_nexus/): A KIP-based knowledge graph implementation for AI long-term memory.
-   [`anda_db`](./rs/anda_db/): The core database library, integrating storage, indexing, and collection management.
-   [`anda_db_schema`](./rs/anda_db_schema/): Defines the schema structures and types for Anda DB.
-   [`anda_db_derive`](./rs/anda_db_derive/): Provides procedural macros (`AndaDBSchema`, `FieldTyped`) for convenient schema definition.
-   [`anda_object_store`](./rs/anda_object_store/): An extension for `object_store` that adds metadata management and transparent encryption.
-   [`anda_db_btree`](./rs/anda_db_btree/): A high-performance B-tree index implementation.
-   [`anda_db_tfs`](./rs/anda_db_tfs/): A full-text search implementation using the BM25 ranking algorithm.
-   [`anda_db_hnsw`](./rs/anda_db_hnsw/): A high-performance HNSW index for vector similarity search.
-   [`anda_kip`](./rs/anda_kip/): A Rust SDK for the KIP (Knowledge Interaction Protocol), including a parser and executor framework.

## Getting Started

### Basic Usage

Here's a simplified example demonstrating how to connect to a database, define a schema, create a collection, add documents, and perform a query.

Source code: https://github.com/ldclabs/anda-db/blob/main/rs/anda_db/examples/db_demo.rs

### Build Anda Cognitive Nexus

Here's a core snippet of Anda Cognitive Nexus that built on top of Anda DB:
```rs
impl CognitiveNexus {
    pub async fn connect<F>(db: Arc<AndaDB>, f: F) -> Result<Self, KipError>
    where
        F: AsyncFnOnce(&CognitiveNexus) -> Result<(), KipError>,
    {
        let schema = Concept::schema().map_err(KipError::parse)?;
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
            .await
            .map_err(db_to_kip_error)?;

        let schema = Proposition::schema().map_err(KipError::parse)?;
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
            .await
            .map_err(db_to_kip_error)?;
        let this = Self {
            db,
            concepts,
            propositions,
        };

        if !this
            .has_concept(&ConceptPK::Object {
                r#type: META_CONCEPT_TYPE.to_string(),
                name: META_CONCEPT_TYPE.to_string(),
            })
            .await
        {
            this.execute_kml(parse_kml(GENESIS_KIP)?, false).await?;
        }

        if !this
            .has_concept(&ConceptPK::Object {
                r#type: META_CONCEPT_TYPE.to_string(),
                name: PERSON_TYPE.to_string(),
            })
            .await
        {
            this.execute_kml(parse_kml(PERSON_KIP)?, false).await?;
        }

        f(&this).await?;
        Ok(this)
    }

    // ...
}
```

### Anda Cognitive Nexus Example

Here's a brief example of how to initialize the nexus, insert knowledge using KML, and retrieve it with KQL.

```rust
use anda_cognitive_nexus::{CognitiveNexus, KipError};
use anda_db::{database::{AndaDB, DBConfig}, storage::StorageConfig};
use anda_kip::{parse_kml, parse_kql};
use anda_object_store::MetaStoreBuilder;
use object_store::local::LocalFileSystem;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), KipError> {
    // 1. Set up storage and database
    let object_store = MetaStoreBuilder::new(LocalFileSystem::new_with_prefix("./db")?, 10000).build();
    let db_config = DBConfig::default();
    let db = AndaDB::connect(Arc::new(object_store), db_config).await?;

    // 2. Connect to the Cognitive Nexus
    let nexus = CognitiveNexus::connect(Arc::new(db), |_| async { Ok(()) }).await?;
    println!("Connected to Anda Cognitive Nexus: {}", nexus.name());

    // 3. Manipulate Knowledge with KML (Knowledge Manipulation Language)
    let kml_string = r#"
    UPSERT {
        // Define concept types
        CONCEPT ?drug_type {
            {type: "$ConceptType", name: "Drug"}
            SET ATTRIBUTES {
                description: "Pharmaceutical drug concept type"
            }
        }

        CONCEPT ?symptom_type {
            {type: "$ConceptType", name: "Symptom"}
            SET ATTRIBUTES {
                description: "Medical symptom concept type"
            }
        }

        // Define relation types
        CONCEPT ?treats_relation {
            {type: "$PropositionType", name: "treats"}
            SET ATTRIBUTES {
                description: "Drug treats symptom relationship"
            }
        }

        // Create symptom concepts
        CONCEPT ?headache {
            {type: "Symptom", name: "Headache"}
            SET ATTRIBUTES {
                severity_scale: "1-10",
                description: "Pain in the head or neck area"
            }
        }

        // Create a drug and the symptom it treats
        CONCEPT ?aspirin {
            {type: "Drug", name: "Aspirin"}
            SET ATTRIBUTES { molecular_formula: "C9H8O4", risk_level: 1 }
            SET PROPOSITIONS {
                ("treats", {type: "Symptom", name: "Headache"})
            }
        }
    }
    WITH METADATA { source: "Basic Medical Knowledge" }
    "#;

    let kml_commands = parse_kml(kml_string)?;
    let kml_result = nexus.execute_kml(kml_commands, false).await?;
    println!("KML Execution Result: {:#?}", kml_result);

    // 4. Query Knowledge with KQL (Knowledge Query Language)
    let kql_query = r#"
    FIND(?drug.name, ?drug.attributes.risk_level)
    WHERE {
        ?drug {type: "Drug"}
        (?drug, "treats", {type: "Symptom", name: "Headache"})
    }
    "#;

    let (kql_result, _) = nexus.execute_kql(parse_kql(kql_query)?).await?;
    println!("KQL Query Result: {:#?}", kql_result);

    nexus.close().await?;
    Ok(())
}
```

## License

Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.
