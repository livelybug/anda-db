use anda_db_hnsw::{HnswConfig, HnswIndex};
use rand::Rng;
use tokio::time;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

pub fn unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    ts.as_millis() as u64
}

// cargo build --example hnsw_demo --release
// ./target/release/examples/hnsw_demo
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    structured_logger::Builder::new().init();

    const DIM: usize = 384;

    // 创建索引 (384维向量，如BERT嵌入)
    let config = HnswConfig {
        dimension: DIM,
        max_nodes: Some(1_000_000),
        ..Default::default()
    };
    // 39900 inserted 100 vectors in 2.482874333s
    // 39900 Search returned 10 results in 4.483417ms
    // 40000 inserted 100 vectors in 2.736496208s
    // 40000 Search returned 10 results in 3.011458ms
    // 40000 Removed vector 21574 in 13.337416ms
    // config.select_neighbors_strategy = SelectNeighborsStrategy::Simple;
    // 39900 inserted 100 vectors in 631.205083ms
    // 39900 Search returned 10 results in 2.442875ms
    // 40000 inserted 100 vectors in 637.636791ms
    // 40000 Search returned 10 results in 2.136208ms
    // 40000 Removed vector 13432 in 12.864834ms
    let index = HnswIndex::new("anda_db_hnsw".to_string(), Some(config));

    // 模拟数据流
    let mut rng = rand::rng();

    let mut inert_start = time::Instant::now();
    for i in 0..50_000 {
        let vector: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
        let _ = index.insert_f32(i as u64, vector, unix_ms())?;
        // println!("{} inserted vector {}", i, i);

        // 模拟搜索查询
        if i % 100 == 0 {
            println!("{} inserted 100 vectors in {:?}", i, inert_start.elapsed());
            inert_start = time::Instant::now();

            let query: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
            let query_start = time::Instant::now();
            let results = index.search_f32(&query, 10)?;
            println!(
                "{} Search returned {} results in {:?}",
                i,
                results.len(),
                query_start.elapsed()
            );
        }

        // 模拟删除
        if i % 1000 == 0 && i > 0 {
            let to_remove = rng.random_range(0..i);
            let remove_start = time::Instant::now();
            index.remove(to_remove, unix_ms());
            println!(
                "{} Removed vector {} in {:?}",
                i,
                to_remove,
                remove_start.elapsed()
            );
        }
    }

    // 打印统计信息
    let stats = index.stats();
    println!("Index statistics:");
    println!("- Total vectors: {}", stats.num_nodes);
    println!("- Max layer: {}", stats.max_layer);
    println!("- Avg connections: {:.2}", stats.avg_connections);
    println!("- Search operations: {}", stats.search_count);
    println!("- Insert operations: {}", stats.insert_count);
    println!("- Delete operations: {}", stats.delete_count);

    // 最终保存
    {
        let file = tokio::fs::File::create("hnsw_demo.cbor")
            .await?
            .compat_write();
        let store_start = time::Instant::now();
        index.store_all(file, unix_ms()).await?;
        println!("Stored index with nodes in {:?}", store_start.elapsed());
    }

    let file = tokio::fs::File::open("hnsw_demo.cbor").await?.compat();
    let load_start = time::Instant::now();
    let index = HnswIndex::load(file).await?;
    println!("Load index in {:?}", load_start.elapsed());
    let query: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>()).collect();
    let query_start = time::Instant::now();
    let results = index.search_f32(&query, 10)?;
    println!(
        "Search returned {} results in {:?}",
        results.len(),
        query_start.elapsed()
    );

    Ok(())
}
