pub mod db;
pub mod entity;

mod helper;
mod types;

pub use db::*;
pub use entity::*;
pub use helper::*;
pub use types::ConceptPK;

#[cfg(test)]
mod tests {
    use super::*;
    use anda_db::{
        collection::CollectionConfig,
        database::{AndaDB, DBConfig},
        error::DBError,
    };
    use object_store::memory::InMemory;
    use std::sync::Arc;

    async fn build_future() {
        let db = AndaDB::connect(Arc::new(InMemory::new()), DBConfig::default())
            .await
            .unwrap();

        let schema = Concept::schema().unwrap();
        let _concepts = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "concepts".to_string(),
                    description: "Concept nodes".to_string(),
                },
                async |_collection| Ok::<(), DBError>(()),
            )
            .await
            .unwrap();

        let _nexus = CognitiveNexus::connect(Arc::new(db), async |_nexus| Ok(()))
            .await
            .unwrap();
    }
    fn assert_send<T: Send>(_: &T) {}

    #[tokio::test]
    #[ignore = "test is used for compilation errors"]
    async fn test_async_send_lifetime() {
        let fut = build_future();
        assert_send(&fut); // 编译报错信息会更聚焦
        fut.await;
    }
}
