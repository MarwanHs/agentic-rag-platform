# agentic-rag-platform

**An agentic RAG and multi-agent orchestration platform, built to demonstrate production-grade patterns — not just a proof of concept.**

This platform combines iterative, self-correcting retrieval with a pluggable multi-agent orchestration layer, served through a FastAPI backend designed to scale: async job processing, caching, observability, and load-tested throughput numbers included.

Most RAG demos stop at "embed, retrieve, generate." This platform goes further:

- **Agentic retrieval** — query decomposition for multi-hop questions, iterative retrieval with confidence checks, and citation-grounded self-critique (not single-shot prompt-and-respond)
- **Real orchestration** — a typed, pluggable agent/tool architecture (Pydantic contracts, dependency injection) so new tools can be added without touching core logic
- **Production concerns, not afterthoughts** — async ingestion pipelines, Redis caching, structured logging, distributed tracing across multi-step agent calls, and load-tested p50/p95 latency numbers

Built with **Claude** (Anthropic) for reasoning/tool-use and **Voyage AI** for embeddings, behind a FastAPI service designed for horizontal scale.

## Why this exists

Most agentic RAG projects on GitHub demonstrate that retrieval works. This one is built to demonstrate that it *keeps* working — under load, with observability, with measured tradeoffs, and with an architecture another engineer could extend without reading every line first.

## Progress

- [x] Repo scaffolding
- [ ] architecture design
- [ ] Core retrieval — embeddings + hybrid search (Voyage)
- [ ] Agentic retrieval loop — query decomposition, iterative retrieval, self-critique
- [ ] Evaluation harness — golden dataset, retrieval + faithfulness metrics
- [ ] FastAPI service — endpoints, streaming responses
- [ ] Multi-agent orchestration layer — pluggable tools, agent loop
- [ ] Async job queue for ingestion pipelines
- [ ] Caching layer (Redis)
- [ ] Observability — structured logging, tracing, metrics
- [ ] Load testing & published benchmark numbers
- [ ] Docker Compose / deployment manifests