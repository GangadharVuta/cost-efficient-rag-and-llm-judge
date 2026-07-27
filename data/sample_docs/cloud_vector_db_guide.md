# Comprehensive Guide to Vector Databases and Infra Costs

## 1. Managed Vector DB Pricing Dynamics
On fully managed cloud vector databases (such as Pinecone, Zilliz Cloud, or Databricks Vector Search), infrastructure costs are primarily driven by always-on compute pods and index memory allocation. When hosting large document corpora (e.g. 1 million to 10 million vectors), an index that receives light query volume still incurs continuous hourly charges ranging from $70 to over $1,200 per month per pod cluster.

## 2. Embedded and Low-Cost Vector Stores
Embedded vector stores like LanceDB, SQLite-vec, and ChromaDB operate in-process without requiring always-on background server instances. By storing vector embeddings in columnar disk files (e.g., Parquet format in LanceDB) or SQLite database tables, infrastructure storage cost is reduced strictly to object storage or persistent disk costs ($0.02 per GB per month).

## 3. Grounded Retrieval and Citation Tracking
A reliable RAG system must retrieve top-k semantically relevant text chunks using cosine similarity or distance thresholds. Every generated answer must include explicit inline citations referencing the source document ID and chunk index. If the maximum similarity score falls below the relevance threshold (e.g., 0.35), the RAG pipeline must return a deterministic 'no relevant context' message to prevent hallucination.

## 4. Chunking Strategies and Idempotency
Document ingestion pipelines split raw documents into overlapping chunks (e.g., 512 tokens per chunk with 64 token overlap). Idempotency is maintained by computing a SHA-256 hash of the chunk content and document identifier. Re-ingesting an existing document updates matching chunk IDs without producing duplicate vectors.
