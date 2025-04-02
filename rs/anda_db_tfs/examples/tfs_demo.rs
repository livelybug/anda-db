use anda_db_tfs::{BM25Index, jieba_tokenizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    id: u64,
    text: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    structured_logger::Builder::new().init();

    // 创建索引
    let index = BM25Index::new(jieba_tokenizer());

    // 批量添加文档
    let docs: HashMap<u64, Document> = vec![
        Document {
            id: 1,
            text: "Rust is a systems programming language".into(),
        },
        Document {
            id: 2,
            text: "Rust is fast and memory efficient, 安全、并发、实用".into(),
        },
        Document {
            id: 3,
            text: "Python is a dynamic language".into(),
        },
    ]
    .into_iter()
    .map(|doc| (doc.id, doc))
    .collect();

    for (_, doc) in &docs {
        index.add_document(doc.id, &doc.text).unwrap();
    }

    // 搜索
    let results = index.search("rust memory", 2);
    for (doc_id, score) in results {
        println!(
            "Found doc {} (score: {:.2}): {:?}",
            doc_id,
            score,
            docs.get(&doc_id)
        );
    }

    let results = index.search("安全", 2);
    for (doc_id, score) in results {
        println!(
            "安全: found doc {} (score: {:.2}): {:?}",
            doc_id,
            score,
            docs.get(&doc_id)
        );
    }

    // 删除文档
    let doc = docs.get(&3).unwrap();
    index.remove_document(doc.id, &doc.text);
    println!("Total documents after removal: {}", index.len());

    // 保存和加载
    {
        let mut file = std::fs::File::create("tfs_demo.cbor")?;
        index.save(&mut file)?;
    }

    let file = std::fs::File::open("tfs_demo.cbor")?;
    let loaded_index = BM25Index::load(&file, jieba_tokenizer())?;
    println!("Loaded index with {} documents", loaded_index.len());

    Ok(())
}
