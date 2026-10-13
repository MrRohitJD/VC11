# Gmail RAG (Retrieval-Augmented Generation) System

Standalone AI Retrieval and Context Synthesis service for Gmail Mailbox Intelligence.
Kept completely outside the `gmail-ai-dashboard` repository to maintain modularity, service isolation, and clean separation of concerns.

---

## 🏛️ Architecture Overview

```
User Query (e.g. "Find interview emails from last 3 months")
                   │
                   ▼
       [ Assistant Frontend UI ]
        (localhost:5173/assistant)
                   │
                   ▼
      [ RAG Pipeline Orchestrator ] (rag_pipeline.py)
      ┌────────────┴────────────┐
      ▼                         ▼
[ Query Embedding ]     [ Intent Classifier ]
  (embedder.py)           (schemas.py)
      │
      ▼
[ Vector Store & Hybrid Search ] (vector_store.py / retriever.py)
  - Cosine Dense Vector Matching
  - Metadata Filtering (Dates, AI Labels, Senders)
  - Cross-Chunk Deduplication & Grounding
      │
      ▼
[ Context Assembler & Synthesizer ]
  - Metrics Extraction (KPIs, Active Processes)
  - Entity Extraction (Dates, Upcoming Interviews, Bills)
  - Source Citations & References
      │
      ▼
   Grounded Answer with Sources & Metrics
```

---

## 📂 Directory Structure

| File | Description |
|---|---|
| `schemas.py` | Pydantic data contracts for chunks, queries, citations, and responses |
| `config.py` | Embedding dimensions, chunk size, similarity thresholds, and LLM providers |
| `chunker.py` | Smart hierarchical email chunker preserving headers, metadata, and body |
| `embedder.py` | Dense vector embedder with caching and robust deterministic fallbacks |
| `vector_store.py` | Vector database with cosine similarity search and metadata filtering |
| `retriever.py` | Hybrid retriever combining semantic search and citation grounding |
| `rag_pipeline.py` | End-to-end RAG orchestrator for query analysis, retrieval, and synthesis |
| `api_server.py` | Standalone FastAPI microservice exposing `/rag/query`, `/rag/index`, `/health` |
| `requirements.txt` | Python dependencies for the standalone service |

---

## 🚀 Running the Standalone RAG Service

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the standalone API server
python api_server.py
# Or:
uvicorn api_server:app --port 8001 --reload
```
