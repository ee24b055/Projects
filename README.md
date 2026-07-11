# Projects

1)SmartFin-RAG: Asynchronous Hybrid Financial Inference Engine

An institutional-grade Retrieval-Augmented Generation (RAG) platform built with FastAPI and LangChain to extract deterministic structural features and sentiment arrays from unstructured alternative financial datasets (Earnings Call Transcripts, SEC filings).

## System Architecture & Optimization Pillars

* **Multi-Tenant Metadata Isolation:** Designed an API ingestion routing engine that isolates corporate text matrices inside ChromaDB using dynamic metadata filters, preventing cross-ticker data bleed during concurrent multi-user execution sessions.
* **Hybrid Structural Retrieval:** Combines a dense vector embedding matrix (`gemini-embedding-2-preview`) with a sparse token-matching index layer to preserve strict numerical values, acronym signals, and specific margin trends (e.g., basis points metrics).
* **Cross-Encoder Reranking Integration:** Implemented a localized `mxbai-rerank-base-v1` cross-encoder layer to score context fragments before inference, completely bypassing the LLM "Lost-in-the-Middle" context window limitation.
* **Asynchronous Execution Bounds:** Utilizes FastAPI `BackgroundTasks` to stream and process heavy document ingestion tasks asynchronously, preventing event-loop degradation.
**Automated Test Coverage (`pytest`):** Implemented an end-to-end integration test suite using FastAPI's `TestClient`. It dynamically generates layout-valid PDF binary streams in memory to fully verify the robustness of the background ingestion pipeline without manual file uploads.
* **HTTP & Web User Interface:** Integrated a clean, production-ready single-page web terminal directly into the application. Users can now easily dispatch HTTP multipart form requests (PDF payloads + Query string parameters) to the backend tracking engine.
* **High-Signal LLM Extraction:** Upgraded the downstream pipeline to pass text blocks through a Cross-Encoder reranking model to eliminate text noise, optimizing what is fed to the Gemini LLM for structured financial sentiment scoring.

2) Multi-Factor Constrained Statistical Arbitrage Engine
An enterprise-grade numerical computing pipeline built in Python. This framework isolates pure idiosyncratic alpha by estimating rolling time-varying multi-factor risk parameters via the Fama-French 3-Factor Model. It features an automated multi-threaded yfinance data acquisition pipeline, a rolling OLS risk engine, and a high-dimensional convex optimization layer (SLSQP matrix solver) that dynamically neutralizes Market, Size, and Value betas simultaneously while enforcing strict dollar neutrality, position caps, and transaction cost turnover drag constraints.

3) ### GraphCache-LLM
A high-performance, dependency-free context routing and memory management engine designed to optimize multi-agent LLM systems. By modeling agent workflows as an active Directed Acyclic Graph (DAG), the engine orchestrates non-conflicting task sequences via topological sorting and enforces strict memory bounds using a custom $O(1)$ Least Recently Used (LRU) cache. This architecture prevents context redundancy, lowers API token consumption, and provides cascading downstream cache invalidations via graph traversals without relying on external frameworks.
