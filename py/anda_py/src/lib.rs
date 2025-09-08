use anda_cognitive_nexus::CognitiveNexus;
use anda_core::BoxError;
use anda_db::database::{AndaDB, DBConfig};
use anda_engine::store::LocalFileSystem;
use anda_kip::{CommandType, KipError, Request, Response};
use object_store::memory::InMemory;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use anda_object_store::{MetaStore, MetaStoreBuilder};
use anda_kip::Json;
use anda_kip::executor::Executor;

/// Formats the sum of two numbers as a string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn anda_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum StoreLocationType {
    InMem,
    LocalFile,
}

#[derive(Debug, Deserialize)]
pub struct AndaDbConfig {
    pub store_location_type: StoreLocationType,
    pub store_location: String, // changed from Option<String> to String
    pub DB_name: String,
    pub DB_desc: Option<String>,
    pub meta_cache_capacity: Option<u64>,
}

impl AndaDbConfig {
    /// Verifies the configuration for AndaDbConfig.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `store_location_type` is `LocalFile` and `store_location` is empty.
    /// - `store_location` does not exist on the filesystem.
    pub fn verify_config(&self) -> Result<(), String> {
        if let StoreLocationType::LocalFile = self.store_location_type {
            if self.store_location.trim().is_empty() {
                return Err("store_location is required when store_location_type is LocalFile".to_string());
            }
            use std::path::Path;
            if !Path::new(&self.store_location).exists() {
                return Err(format!("store_location path does not exist: {}", self.store_location));
            }
        }
        Ok(())
    }
}

/// Create a CognitiveNexus instance from AndaDbConfig.
/// Returns an Arc-wrapped Nexus for use in KIP execution.
/// * `db_config` - Database configuration as an `AndaDbConfig` struct.
///     - `store_location_type`: `"InMem"` for in-memory DB, `"LocalFile"` for file-backed DB.
///     - `store_location`: Required if `store_location_type` is `"LocalFile"`.
///     - `DB_name`: Name of the database.
///     - `DB_desc`: Optional description of the database.
///     - `meta_cache_capacity`: Optional cache capacity for metadata (default: 10000).
///
///
/// # Errors
/// Returns an error if the config is invalid or DB/Nexus creation fails.
pub async fn create_kip_db(
    db_config: AndaDbConfig,
) -> Result<Arc<CognitiveNexus>, BoxError> {
    db_config.verify_config()
        .map_err(|e| KipError::Execution(e))?;

    let db_name = db_config.DB_name.as_str();
    let db_desc = db_config.DB_desc.as_deref().unwrap_or_default();
    let meta_cache_capacity = db_config.meta_cache_capacity.unwrap_or(10000);

    let object_store: Arc<dyn object_store::ObjectStore> = match db_config.store_location_type {
        StoreLocationType::InMem => Arc::new(InMemory::new()),
        StoreLocationType::LocalFile => {
            let local_file = MetaStoreBuilder::new(
                LocalFileSystem::new_with_prefix(&db_config.store_location)
                    .map_err(|err| KipError::Execution(err.to_string()))?,
                meta_cache_capacity,
            ).build();
            Arc::new(local_file)
        }
    };

    let db_config = DBConfig {
        name: db_name.to_string(),
        description: db_desc.to_string(),
        ..Default::default()
    };

    let db = Arc::new(AndaDB::connect(object_store, db_config).await?);
    let nexus = Arc::new(CognitiveNexus::connect(db, async |_| Ok(())).await?);
    Ok(nexus)
}

/// Executes a KIP command using an existing Executor instance.
///
/// # Arguments
///
/// * `nexus` - Reference to an Executor instance (`&(impl Executor + Sync)`).
/// * `command` - The KIP command string to execute (KML/KQL/META).
/// * `parameters` - An optional map of command parameters (`Option<Map<String, Json>>`). If `None`, treated as empty.
/// * `dry_run` - If true, performs a dry run without committing changes.
///
/// # Returns
///
/// Returns a tuple of the command type and the response on success, or a boxed error on failure.
///
/// # Errors
///
/// Returns an error if the KIP command execution fails.
///
/// # Example
/// 
/// Refer to tools/anda_py/examples directory
pub async fn execute_kip(
    nexus: &(impl Executor + Sync),
    command: String,
    parameters: Option<Map<String, Json>>,
    dry_run: bool,
) -> Result<(CommandType, Response), BoxError> {
    let params_map = parameters.unwrap_or_default();

    let request = Request {
        command,
        parameters: params_map,
        dry_run,
    };

    Ok(request.execute(nexus).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anda_kip::{Json, Map};
    use std::future::Future;

    // Helper to run async code in tests
    fn block_on<F: Future<Output = T>, T>(fut: F) -> T {
        tokio::runtime::Runtime::new().unwrap().block_on(fut)
    }

    // Create basic concept types and medical knowledge capsule
    static MEDICAL_KNOWLEDGE_KML: &str = r#"
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

            CONCEPT ?has_side_effect_relation {
                {type: "$PropositionType", name: "has_side_effect"}
                SET ATTRIBUTES {
                    description: "Drug has side effect relationship"
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

            CONCEPT ?fever {
                {type: "Symptom", name: "Fever"}
                SET ATTRIBUTES {
                    normal_temp: "98.6°F (37°C)",
                    description: "Elevated body temperature"
                }
            }

            CONCEPT ?stomach_irritation {
                {type: "Symptom", name: "Stomach Irritation"}
                SET ATTRIBUTES {
                    severity: "mild to moderate",
                    description: "Gastrointestinal discomfort"
                }
            }

            // Create specific drug concepts
            CONCEPT ?aspirin {
                {type: "Drug", name: "Aspirin"}
                SET ATTRIBUTES {
                    molecular_formula: "C9H8O4",
                    risk_level: 1,
                    description: "Common pain reliever and anti-inflammatory drug"
                }
                SET PROPOSITIONS {
                    ("treats", ?headache)
                    ("treats", ?fever)
                    ("has_side_effect", ?stomach_irritation)
                }
            }
        }
        WITH METADATA {
            source: "KIP Demo Medical Knowledge",
            author: "Demo System",
            confidence: 0.95,
            created_at: "2025-07-01T00:00:00Z"
        }
        "#;

    // Create a new hypothetical drug
    static NEW_DRUG_KML: &str = r#"
        UPSERT {
            CONCEPT ?brain_fog {
                {type: "Symptom", name: $symptom_name}
                SET ATTRIBUTES {
                    description: "Mental fatigue and lack of clarity",
                    cognitive_impact: "high"
                }
            }

            CONCEPT ?neural_bloom {
                {type: "Symptom", name: "Neural Bloom"}
                SET ATTRIBUTES {
                    description: "A rare side effect characterized by temporary burst of creative thoughts",
                    frequency: "rare",
                    severity: "mild"
                }
            }

            CONCEPT ?cognizine {
                {type: "Drug", name: "Cognizine"}
                SET ATTRIBUTES {
                    molecular_formula: "C12H15N5O3",
                    risk_level: 2,
                    description: "A novel nootropic drug designed to enhance cognitive functions",
                    status: "experimental"
                }
                SET PROPOSITIONS {
                    ("treats", {type: "Symptom", name: "Brain Fog"})
                    ("has_side_effect", ?neural_bloom)
                }
            }
        }
        WITH METADATA {
            source: "Experimental Drug Research",
            confidence: 0.75,
            status: "under_review"
        }
        "#;

    #[test]
    fn test_execute_kip_in_mem() {
        // 1. Execute the first KML command from the demo to set up schema and initial data
        println!("\n1. Executing Medical Knowledge KML...");

        // Add db_config for in-memory DB (as AndaDbConfig struct expects)
        let db_config_in_mem = AndaDbConfig {
            store_location_type: StoreLocationType::InMem,
            store_location: "".to_owned(),
            DB_name: "test_medical_db".to_string(),
            DB_desc: Some("Ephemeral DB for medical KIP test".to_string()),
            meta_cache_capacity: Some(10000),
        };

        // Create Nexus instance for in-memory DB
        let nexus_in_mem = block_on(create_kip_db(db_config_in_mem)).expect("Failed to create in_mem Nexus");

        // Use empty Map for parameters
        let empty_params: Map<String, Json> = Map::new();

        let (_, response1) = block_on(execute_kip(
            nexus_in_mem.as_ref(),
            MEDICAL_KNOWLEDGE_KML.to_string(),
            Some(empty_params.clone()),
            false,
        ))
        .expect("Execution of medical_knowledge_kml failed");
        assert!(matches!(response1, Response::Ok { .. }), "Expected first KML execution to be Ok, but got {:?}", response1);
        println!("Medical Knowledge KML executed successfully (in_mem DB).");

        // 2. Execute the second KML command from the demo to add more data
        println!("\n2. Executing New Drug KML...");

        let mut ql_parameters = Map::new();
        ql_parameters.insert(
            "symptom_name".to_string(),
            Json::String("Brain Fog".to_string()),
        );

        let (_, response2) = block_on(execute_kip(
            nexus_in_mem.as_ref(),
            NEW_DRUG_KML.to_string(),
            Some(ql_parameters.clone()),
            false,
        ))
        .expect("Execution of new_drug_kml failed");
        assert!(matches!(response2, Response::Ok { .. }), "Expected third KML execution to be Ok, but got {:?}", response2);
        println!("New Drug KML executed successfully (in_mem DB).");
        // 3. Execute a KQL query from the demo to verify the data
        println!("\n3. Executing KQL Query to find all drugs...");
        let query = r#"
        FIND(?drug.name, ?drug.attributes.molecular_formula, ?drug.attributes.risk_level)
        WHERE {
            ?drug {type: "Drug"}
        }
        ORDER BY ?drug.attributes.risk_level ASC
        "#;

        let (_, query_response) = block_on(execute_kip(
            nexus_in_mem.as_ref(),
            query.to_string(),
            None,
            false,
        ))
        .expect("Execution of KQL query failed");

        println!("Query Response: {:#?}", query_response);

        // 4. Assert that the query was successful and returned the correct data
        assert!(matches!(query_response, Response::Ok { .. }), "Expected KQL query to be Ok, but got {:?}", query_response);

        if let Response::Ok { result, .. } = query_response {
            let result_array = result.as_array().expect("Result should be an array");
            assert_eq!(result_array.len(), 2, "Expected to find 2 drugs, but found {}", result_array.len());
            println!("Successfully found 2 drugs as expected.");
        } else {
            panic!("Query failed, expected Ok response");
        }

        println!("\n--- Full Stateful KIP Execution Test Passed ---");        
    }
}
