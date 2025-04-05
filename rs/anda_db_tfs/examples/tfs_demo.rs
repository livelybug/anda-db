use anda_db_tfs::{BM25Index, jieba_tokenizer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    id: u64,
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    structured_logger::Builder::new().init();

    // 创建索引
    let index = BM25Index::new(jieba_tokenizer(), None);

    // 批量添加文档
    let docs = vec![
        Document {
            id: 0,
            text: "Rust is a systems programming language".into(),
        },
        Document {
            id: 1,
            text: "Rust is fast and memory efficient, 安全、并发、实用".into(),
        },
        Document {
            id: 2,
            text: "Python is a dynamic language".into(),
        },
    ];

    let rs = index.insert_batch(
        docs.iter()
            .into_iter()
            .map(|doc| (doc.id, doc.text.clone()))
            .collect(),
        0,
    );
    assert_eq!(rs.len(), 3);

    // 搜索
    let results = index.search("rust memory", 2);
    for (doc_id, score) in results {
        println!("Found doc {}, score: {:.2}", doc_id, score);
    }

    let results = index.search("安全", 2);
    for (doc_id, score) in results {
        println!("Found doc {}, score: {:.2}", doc_id, score);
    }

    // 删除文档
    index.remove(docs[2].id, &docs[2].text, 0);
    println!("Total documents after removal: {}", index.len());

    // 保存和加载
    {
        let file = tokio::fs::File::create("tfs_demo.cbor").await?;
        index.store_all(file, 0).await?;
    }

    let file = tokio::fs::File::open("tfs_demo.cbor").await?;
    let loaded_index = BM25Index::load(file, jieba_tokenizer()).await?;
    println!("Loaded index with {} documents", loaded_index.len());

    Ok(())
}
