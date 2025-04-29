# Anda DB

Anda DB is a Rust library designed as a specialized database for AI Agents, focusing on knowledge memory. It supports multimodal data storage, full-text search, and vector similarity search, integrating seamlessly as a local database within AI Agent applications.

## Key Features

-   **Embedded Library:** Functions as a Rust library, not a standalone remote database service, enabling direct integration into AI Agent builds.
-   **Object Store Backend:** Leverages an [Object Store](https://docs.rs/object_store) interface, supporting various backends like AWS S3, Google Cloud Storage, Azure Blob Storage, local filesystem, and even the [ICP blockchain](https://internetcomputer.org/).
-   **Encrypted Storage:** Offers optional encrypted storage, writing all data as ciphertext to the Object Store (currently supported for the ICP backend) to ensure data privacy.
-   **Multimodal Data:** Natively handles storage and retrieval of diverse data types including text, images, audio, video, and arbitrary binary data within a flexible document structure.
-   **Flexible Schema & ORM:** Document-oriented design with a flexible schema supporting various field types like `bfloat16` vectors, binary data, JSON, etc. Includes built-in ORM support via procedural macros.
-   **Advanced Indexing:**
    -   **BTree Index:** Enables precise matching, range queries (including timestamps), and multi-conditional logical queries on `U64`, `I64`, `String`, `Bytes`, `Array<T>`, `Option<T>` fields, powered by [`anda_db_btree`](https://docs.rs/anda_db_btree).
    -   **BM25 Index:** Provides efficient full-text search capabilities with multi-conditional logic and powerful tokenizer, powered by [`anda_db_tfs`](https://docs.rs/anda_db_tfs).
    -   **HNSW Index:** Offers high-performance approximate nearest neighbor (ANN) search for vector similarity, powered by [`anda_db_btree`](https://docs.rs/anda_db_hnsw).
-   **Hybrid Search:** Automatically combines and weights text (BM25) and vector (HNSW) search results using Reciprocal Rank Fusion (RRF) for comprehensive retrieval.
-   **Incremental Updates & Persistence:** Supports efficient incremental index updates and document deletions without requiring costly full index rebuilds. Capably saves and loads the entire database state, ensuring data durability.
-   **Efficient Serialization:** Uses CBOR (Concise Binary Object Representation) and Zstd for compact and efficient data serialization.
-   **Collection Management:** Organizes documents into distinct collections, each with its own schema and indexes.

## License

Copyright © 2025 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda-db` is licensed under the MIT License. See [LICENSE](./LICENSE-MIT) for the full license text.