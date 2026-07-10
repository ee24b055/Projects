# Projects

SmartFin-RAG: Asynchronous Hybrid Financial Inference Engine

An institutional-grade Retrieval-Augmented Generation (RAG) platform built with FastAPI and LangChain to extract deterministic structural features and sentiment arrays from unstructured alternative financial datasets (Earnings Call Transcripts, SEC filings).

## System Architecture & Optimization Pillars

* **Multi-Tenant Metadata Isolation:** Designed an API ingestion routing engine that isolates corporate text matrices inside ChromaDB using dynamic metadata filters, preventing cross-ticker data bleed during concurrent multi-user execution sessions.
* **Hybrid Structural Retrieval:** Combines a dense vector embedding matrix (`gemini-embedding-2-preview`) with a sparse token-matching index layer to preserve strict numerical values, acronym signals, and specific margin trends (e.g., basis points metrics).
* **Cross-Encoder Reranking Integration:** Implemented a localized `mxbai-rerank-base-v1` cross-encoder layer to score context fragments before inference, completely bypassing the LLM "Lost-in-the-Middle" context window limitation.
* **Asynchronous Execution Bounds:** Utilizes FastAPI `BackgroundTasks` to stream and process heavy document ingestion tasks asynchronously, preventing event-loop degradation.
