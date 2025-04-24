use anda_db::{
    collection::{Collection, CollectionConfig},
    database::{AndaDB, DBConfig},
    error::DBError,
    index::HnswConfig,
    query::{Filter, Query, RangeQuery, Search},
    schema::{Document, Fe, Ft, Fv, Json, Resource, Schema, Segment},
    storage::StorageConfig,
};
use anda_db_tfs::jieba_tokenizer;
use ic_auth_types::Xid;
use object_store::local::LocalFileSystem;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};
use structured_logger::unix_ms;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Knowledge {
    pub id: u64,
    pub thread: Xid,
    pub title: String,
    pub authors: Vec<String>,
    pub metadata: BTreeMap<String, Json>,
    pub segments: Vec<Segment>,
    pub source: Option<Resource>,
}

// cargo run --example db_demo --features=full
#[tokio::main]
async fn main() -> Result<(), DBError> {
    // init structured logger
    structured_logger::init();

    // create an in-memory object store
    // It's a simple in-memory storage for testing purposes.
    // In a real application, you would use a persistent storage backend.
    let object_store = LocalFileSystem::new_with_prefix("./debug")?;

    let db_config = DBConfig {
        name: "anda_db_demo".to_string(),
        description: "Anda DB demo".to_string(),
        storage: StorageConfig {
            compress_level: 0, // no compression
            ..Default::default()
        },
    };

    // connect to the database (create if it doesn't exist)
    let db = AndaDB::connect(Arc::new(object_store), db_config).await?;
    log::info!(
        action = "connect",
        database = db.name();
        "connected to database"
    );

    // 创建用户集合的模式
    let mut schema = Schema::builder();
    schema
        .with_xid("thread", false)?
        .add_field(
            Fe::new("title".to_string(), Ft::Text)?
                .with_required()
                .with_description("knowledge title".to_string()),
        )?
        .add_field(
            Fe::new("authors".to_string(), Ft::Array(vec![Ft::Text]))?
                .with_description("knowledge authors".to_string()),
        )?
        .add_field(
            Fe::new("metadata".to_string(), Ft::Map(BTreeMap::new()))?
                .with_description("knowledge metadata".to_string()),
        )?
        .with_segments("segments", true)?
        .with_resource("source", false)?;
    let schema = schema.build()?;

    // 创建用户集合配置
    let collection_config = CollectionConfig {
        name: "knowledges".to_string(),
        description: "My knowledges".to_string(),
    };

    let collection = db
        .open_or_create_collection(schema, collection_config, async |collection| {
            collection.set_tokenizer(jieba_tokenizer());
            collection
                .create_btree_index_nx("btree_thread", "thread")
                .await?;
            collection
                .create_search_index_nx(
                    "search_segments",
                    "segments",
                    HnswConfig {
                        dimension: 10,
                        ..Default::default()
                    },
                )
                .await?;
            Ok::<(), DBError>(())
        })
        .await?;
    log::info!(
        action = "open_or_create_collection",
        collection = collection.name();
        "opened or created collection"
    );

    add_knowledges_and_query(&collection).await?;

    db.close().await?;

    Ok(())
}

// ...existing code...

async fn add_knowledges_and_query(collection: &Arc<Collection>) -> Result<(), DBError> {
    println!("-----> 开始添加示例知识");
    let mut thread = Xid::new();

    // 创建一些示例知识
    let knowledges = vec![
        Knowledge {
            id: 0,
            thread: thread.clone(),
            title: "Rust 编程语言入门".to_string(),
            authors: vec!["Anda".to_string(), "Bill".to_string()],
            metadata: BTreeMap::new(),
            segments: vec![
                Segment::new(
                    "Rust 是一门系统级编程语言，专注于安全性、并发性和性能。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                Segment::new(
                    "Rust 的所有权系统是其最独特的特性之一，它在编译时确保内存安全。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]),
            ],
            source: None,
        },
        Knowledge {
            id: 0,
            thread: thread.clone(),
            title: "向量数据库简介".to_string(),
            authors: vec!["Charlie".to_string()],
            metadata: BTreeMap::new(),
            segments: vec![
                Segment::new(
                    "向量数据库是一种特殊类型的数据库，专门用于存储和检索向量嵌入。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                Segment::new(
                    "与传统数据库相比，向量数据库能够高效地进行相似性搜索。".to_string(),
                    None,
                )
                .with_vec_f32(vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
            ],
            source: None,
        },
    ];

    let stats = collection.stats();
    println!("-----> 集合统计信息: {:?}", stats);

    // 插入知识到集合中
    if stats.num_documents == 0 {
        for mut knowledge in knowledges {
            collection.obtain_segment_ids(&mut knowledge.segments);
            let doc = Document::try_from(collection.schema(), &knowledge)?;
            let id = collection.add(doc).await?;
            println!("插入知识成功, id: {id}");
        }
        collection.flush(unix_ms()).await?;
    }

    println!("-----> 开始执行查询: id = 1");
    // 1. 基于 ID 查询单个文档
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            filter: Some(Filter::Field((
                "id".to_string(),
                RangeQuery::Eq(Fv::U64(1)),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    thread = result[0].thread.clone();
    for doc in &result {
        println!("查询到知识: {:?}\n", doc);
    }

    // 2. 使用等值条件查询
    println!("-----> 开始执行查询: thread = xxx");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            filter: Some(Filter::Field((
                "thread".to_string(),
                RangeQuery::Eq(Fv::Bytes(thread.as_slice().into())),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 2);
    for doc in &result {
        println!("查询到知识: {:?}\n", doc);
    }

    // 3. 使用文本搜索
    println!("-----> 开始执行查询: text = Rust");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                text: Some("rust".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    for doc in &result {
        println!("查询到知识: {:?}\n", doc);
    }

    // 4. 使用向量搜索（相似性搜索）
    println!("-----> 开始执行查询: vector_search");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                vector: Some(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 2);
    for doc in &result {
        println!("查询到知识: {:?}\n", doc);
    }

    // 5. 复合查询 - 结合多个条件
    println!("-----> 开始执行查询: compound_query");
    let result: Vec<Knowledge> = collection
        .search_as(Query {
            search: Some(Search {
                field: "segments".to_string(),
                text: Some("数据库".to_string()),
                vector: Some(vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                ..Default::default()
            }),
            filter: Some(Filter::Field((
                "id".to_string(),
                RangeQuery::Gt(Fv::U64(1)),
            ))),
            ..Default::default()
        })
        .await?;
    assert_eq!(result.len(), 1);
    for doc in &result {
        println!("查询到知识: {:?}\n", doc);
    }

    Ok(())
}
