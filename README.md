# Projects

1)SmartFin-RAG: Asynchronous Hybrid Financial Inference Engine

An institutional-grade Retrieval-Augmented Generation (RAG) platform built with FastAPI and LangChain to extract deterministic structural features and sentiment arrays from unstructured alternative financial datasets (Earnings Call Transcripts, SEC filings).

## System Architecture & Optimization Pillars

* **Multi-Tenant Metadata Isolation:** Designed an API ingestion routing engine that isolates corporate text matrices inside ChromaDB using dynamic metadata filters, preventing cross-ticker data bleed during concurrent multi-user execution sessions.
* **Hybrid Structural Retrieval:** Combines a dense vector embedding matrix (`gemini-embedding-2-preview`) with a sparse token-matching index layer to preserve strict numerical values, acronym signals, and specific margin trends (e.g., basis points metrics).
* **Cross-Encoder Reranking Integration:** Implemented a localized `mxbai-rerank-base-v1` cross-encoder layer to score context fragments before inference, completely bypassing the LLM "Lost-in-the-Middle" context window limitation.
* **Asynchronous Execution Bounds:** Utilizes FastAPI `BackgroundTasks` to stream and process heavy document ingestion tasks asynchronously, preventing event-loop degradation.

2) Multi-Factor Constrained Statistical Arbitrage Engine
An enterprise-grade numerical computing pipeline built in Python. This framework isolates pure idiosyncratic alpha by estimating rolling time-varying multi-factor risk parameters via the Fama-French 3-Factor Model. It features an automated multi-threaded yfinance data acquisition pipeline, a rolling OLS risk engine, and a high-dimensional convex optimization layer (SLSQP matrix solver) that dynamically neutralizes Market, Size, and Value betas simultaneously while enforcing strict dollar neutrality, position caps, and transaction cost turnover drag constraints.
