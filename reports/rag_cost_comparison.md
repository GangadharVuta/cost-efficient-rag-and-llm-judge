# RAG Cost Comparison and Scale Analysis

This document provides a realistic, assumption-stated cost comparison comparing **Embedded Vector Stores (LanceDB / SQLite-vec)** against **Managed Vector Databases (Pinecone / Zilliz / Databricks Vector Search)** at **100K**, **1M**, and **10M** vector scales.

---

## 1. Core Model & Infrastructure Assumptions

- **Vector Dimensionality**: 384-dimensional dense vectors (`all-MiniLM-L6-v2`, float32 precision).
- **Raw Vector Size**: $384 \times 4\text{ bytes} = 1,536\text{ bytes} \approx 1.54\text{ KB}$ per vector.
- **Metadata Overhead**: 0.5 KB per document chunk (doc_id, source_path, text, category).
- **Total Storage per Vector**: ~2.0 KB per stored vector record.
- **Storage Pricing**: AWS S3 / EBS Persistent Disk at **$0.023 per GB per month**.
- **Managed Vector DB Pricing Model**:
  - Based on Pinecone / Zilliz Cloud standard pod pricing ($0.096/pod-hour \approx $70/month per pod).
  - 100K vectors requires 1 pod (~$70/mo).
  - 1M vectors requires 2 pods for RAM indexing (~$140/mo).
  - 10M vectors requires 16 pods or high-memory cluster (~$1,120/mo).

---

## 2. Monthly Infrastructure Cost Comparison Table

| Vector Scale | Data Storage Volume | Managed Vector DB (Pods) | Embedded LanceDB / SQLite-vec | Monthly Cost Savings (%) |
| :--- | :--- | :--- | :--- | :--- |
| **100,000 (100K)** | ~0.20 GB | **$70.00 / month** | **$0.01 / month** | **99.98%** |
| **1,000,000 (1M)** | ~2.00 GB | **$140.00 / month** | **$0.05 / month** | **99.96%** |
| **10,000,000 (10M)**| ~20.00 GB | **$1,120.00 / month** | **$0.46 / month** | **99.95%** |

---

## 3. Engineering Discussion & Trade-Off Analysis

### When to Use Low-Cost Embedded Stores (LanceDB / SQLite-vec):
1. **Light to Moderate Query Traffic**: When query volume is infrequent (e.g. internal enterprise search, batch QA, or low RPM APIs), paying $70-$1,120/mo for idle pod memory is wasteful.
2. **Serverless & Edge Deployments**: Ideal for AWS Lambda, Cloudflare Workers, desktop CLI tools, or single-container microservices.
3. **Data Privacy & Zero Egress**: Vectors reside entirely on local disk or private cloud storage without third-party network transmission.

### When to Switch Back to Managed Vector DBs:
1. **High Concurrent Writes & Global Scale**: When write throughput exceeds thousands of upserts/sec across distributed multi-region writers.
2. **Ultra-High Availability (99.999% SLA)**: When managed multi-AZ replication and automated failover are strict enterprise requirements.
3. **Distributed Sharding Beyond Single Node**: When index size exceeds single-machine RAM/NVMe storage limits (> 100M vectors).
