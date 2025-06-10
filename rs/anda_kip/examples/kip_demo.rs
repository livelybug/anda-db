use anda_kip::CognitiveNexus;
use anda_kip::execute_kip;
use serde_json;

fn main() {
    // 1. 初始化一个空的认知中枢
    let mut nexus = CognitiveNexus::new();
    println!("--- Initializing Cognitive Nexus ---\n");

    // 2. 使用 KML UPSERT 命令填充初始知识（来自规范中的知识胶囊）
    let kml_upsert_command = r#"
        // Knowledge Capsule: cognizine.v1.0
        UPSERT {
          // Define the main drug concept: Cognizine
          CONCEPT @cognizine {
            ON { type: "Drug", name: "Cognizine" }
            SET ATTRIBUTES {
              molecular_formula: "C12H15N5O3",
              risk_level: 2,
              description: "A novel nootropic drug designed to enhance cognitive functions."
            }
            SET PROPOSITIONS {
              PROP("is_class_of", ON { type: "DrugClass", name: "Nootropic" })
              PROP("treats", ON { type: "Symptom", name: "Brain Fog" })
              PROP("has_side_effect", @neural_bloom) WITH METADATA {
                confidence: 0.75,
                source: "Preliminary Clinical Trial NCT012345"
              }
            }
          }

          // Define the new side effect concept: Neural Bloom
          CONCEPT @neural_bloom {
            ON { type: "Symptom", name: "Neural Bloom" }
            SET ATTRIBUTES {
              description: "A rare side effect characterized by a temporary burst of creative thoughts."
            }
          }

          // Define a pre-existing concept for the relation
          CONCEPT @nootropic_class {
             ON { type: "DrugClass", name: "Nootropic" }
          }
          CONCEPT @brain_fog_symptom {
             ON { type: "Symptom", name: "Brain Fog" }
          }
        }
        WITH METADATA {
          source: "KnowledgeCapsule:Nootropics_v1.0",
          author: "LDC Labs Research Team",
          confidence: 0.95
        }
    "#;

    println!(
        "--- Executing KML UPSERT ---\nCMD:\n{}\n",
        kml_upsert_command
    );
    let response = execute_kip(&mut nexus, kml_upsert_command).unwrap();
    println!(
        "RES:\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );

    // 3. 使用 META 命令探索知识图谱
    let meta_describe_types = r#"DESCRIBE CONCEPT TYPES"#;
    println!(
        "--- Executing META DESCRIBE ---\nCMD:\n{}\n",
        meta_describe_types
    );
    let response = execute_kip(&mut nexus, meta_describe_types).unwrap();
    println!(
        "RES:\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );

    let meta_search = r#"SEARCH CONCEPT "Cognizine""#;
    println!("--- Executing META SEARCH ---\nCMD:\n{}\n", meta_search);
    let response = execute_kip(&mut nexus, meta_search).unwrap();
    println!(
        "RES:\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );

    // 4. 使用 KQL 查询知识
    let kql_find_drug = r#"
        FIND(?drug_name, ?desc)
        WHERE {
            ?drug(type: "Drug", name: "Cognizine")
            ATTR(?drug, "name", ?drug_name)
            ATTR(?drug, "description", ?desc)
        }
    "#;
    println!("--- Executing KQL FIND ---\nCMD:\n{}\n", kql_find_drug);
    let response = execute_kip(&mut nexus, kql_find_drug).unwrap();
    println!(
        "RES:\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );

    let kql_find_effects = r#"
        FIND(?drug_name, ?relation, ?effect_name)
        WHERE {
            ?drug(name: "Cognizine")
            ATTR(?drug, "name", ?drug_name)
            {
                PROP(?drug, "treats", ?effect)
                BIND("treats" AS ?relation)
            } UNION {
                PROP(?drug, "has_side_effect", ?effect)
                BIND("has_side_effect" AS ?relation)
            }
            ATTR(?effect, "name", ?effect_name)
        }
    "#;
    // 注：为演示 UNION，此处添加了一个未在 AST 中完全实现的 BIND 伪指令。
    // 在真实执行器中，可以在 UNION 的每个分支内处理 ?relation 的绑定。
    println!(
        "--- Executing KQL FIND with UNION ---\nCMD:\n(Simplified for demo)\n{}\n",
        kql_find_effects
    );
    // 假设执行器能够处理这种模式
    // let response = execute_kip(&mut nexus, kql_find_effects).unwrap();
    // println!("RES:\n{}\n", serde_json::to_string_pretty(&response).unwrap());

    // 5. 使用 KML DELETE 删除知识
    let kml_delete_prop = r#"
        DELETE PROPOSITION (
            ON { type: "Drug", name: "Cognizine" },
            "treats",
            ON { type: "Symptom", name: "Brain Fog" }
        )
    "#;
    println!("--- Executing KML DELETE ---\nCMD:\n{}\n", kml_delete_prop);
    let response = execute_kip(&mut nexus, kml_delete_prop).unwrap();
    println!(
        "RES:\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );

    // 6. 验证删除
    let kql_verify_delete = r#"
        FIND(?drug_name, ?symptom_name)
        WHERE {
            ?symptom(name: "Brain Fog")
            PROP(?drug, "treats", ?symptom)
            ATTR(?drug, "name", ?drug_name)
            ATTR(?symptom, "name", ?symptom_name)
        }
    "#;
    println!(
        "--- Verifying Deletion with KQL ---\nCMD:\n{}\n",
        kql_verify_delete
    );
    let response = execute_kip(&mut nexus, kql_verify_delete).unwrap();
    println!(
        "RES (should be empty):\n{}\n",
        serde_json::to_string_pretty(&response).unwrap()
    );
}
