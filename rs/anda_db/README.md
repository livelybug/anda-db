# Anda DB

A Rust library for AI knowledge storage with multimodal data support, full-text search, and vector similarity search.

## Features

- **Multimodal Data Support**: Store and retrieve text, images, audio, video, and binary data
- **Full-Text Search**: BM25-based text search using Tantivy
- **Vector Similarity Search**: Fast vector search using IVF-PQ algorithm
- **Hybrid Search**: Combine text and vector search with configurable weights
- **Object Store Integration**: Store data in various backend systems
- **CBOR Serialization**: Efficient binary serialization format
- **Document-Oriented**: Flexible schema with fields of different types
- **Collection Management**: Organize documents into collections
- **Persistence**: Save and load database state
