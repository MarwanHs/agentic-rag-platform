# agentic-rag-platform

**A multi-agent system that helps engineers understand large, unfamiliar codebases — built to demonstrate production-grade agentic RAG and orchestration patterns, not just a proof of concept.**

Ask a question about a codebase — *"why does the retry logic in the ingestion pipeline sometimes double-process a job?"* — and get a grounded, multi-hop answer. Naive RAG breaks on questions like this: the answer requires pulling together a function, its callers, a config file, and a related test, none of which are adjacent in embedding space. This system decomposes the question, retrieves iteratively, falls back to exact structural search when semantic search isn't enough, and synthesizes a cited answer — instead of a single embed-and-generate pass.

Served through a FastAPI backend designed to scale: async ingestion pipelines, caching, observability, and load-tested throughput numbers included.

## How it works

- **Planner agent** (Claude) — interprets the question, decomposes it, and decides which agents are needed to answer it
- **Retriever agent** (Claude + Voyage) — hybrid semantic search over code, docs, and commit history
- **Code-navigation agent** — deterministic structural search (symbol lookup, call-graph traversal); implementation has no dependency on Claude or Voyage, though in the running system it's invoked by the planner rather than called standalone
- **Critic-synthesizer agent** (Claude) — verifies the gathered evidence is sufficient, cites sources, and either answers or refuses rather than filling gaps

See [`docs/architecture.md`](docs/architecture.md) for the full design, including diagrams and the reasoning behind each major decision.

## Requirements

Requires your own **Anthropic** and **Voyage AI** API keys — see [Getting Started](#getting-started). All agentic reasoning and routing goes through Claude, so the system needs both keys to run end to end.

## Why this exists

Most agentic RAG projects on GitHub demonstrate that retrieval works. This one is built to demonstrate that it *keeps* working — under load, with observability, with measured tradeoffs, and with an architecture another engineer could extend without reading every line first. An indexed, reusable retrieval layer also avoids re-reading an entire repository on every session, and scales to codebases larger than any single context window.

## Progress

Checkboxes track implementation, not design — several unchecked items below already have a settled design in `docs/architecture.md`, with implementation in progress or pending.

- [x] Architecture design
- [x] Repo scaffolding
- [x] Core retrieval — embeddings + hybrid search (Voyage) — shared tree-sitter parsing layer, `voyage-code-3` embeddings, Qdrant hybrid (dense + native BM25) search with RRF fusion and reranking
- [x] Code-navigation tooling — symbol search, call-graph lookups — Postgres symbols/references schema, `find_definition`/`find_references` lookups
- [x] Async job queue for ingestion pipelines — Postgres-native queue (`SELECT ... FOR UPDATE SKIP LOCKED`, decision #37), standalone worker executing the full clone → parse → embed/index → code-navigation-index pipeline; verified end-to-end against this repo's own source
- [ ] Agentic retrieval loop — planner, iterative retrieval, critic-synthesizer — *planner (forced-tool-use routing, decision #42), retriever-as-agent (confidence-gated hybrid search with one reformulation retry, decision #45), and code-navigation-as-agent (decision #44) are all implemented; critic-synthesizer and the orchestrating function that wires them together are not, so `POST /codebases/{id}/query` still returns `501 Not Implemented` rather than a fabricated `refused: true` — that field is specifically the critic's sufficiency verdict, and since no critic exists yet, returning it would be indistinguishable from a real refusal*
- [ ] Multi-turn conversation support (LangGraph + Postgres checkpointing) — *design settled (decision #29), not yet implemented — new `POST /codebases/{id}/conversations` and `POST /conversations/{id}/messages` endpoints, planner reasons over prior turns*
- [ ] Evaluation harness — golden dataset, LLM-as-judge grading — *approach decided, details not yet designed*
- [x] FastAPI service — endpoints, async ingestion, blocking query endpoint — *endpoints live, query endpoint stubbed pending orchestration: `POST /jobs`, `GET /jobs/{id}`, and `GET /codebases` are fully functional, but `POST /codebases/{id}/query` returns `501 Not Implemented` until the critic-synthesizer exists*
- [ ] Multi-agent orchestration layer — *planner + both evidence-gathering agents implemented and independently tested; the function that routes between them and calls the (not-yet-built) critic-synthesizer is the remaining piece*
- [ ] Caching layer (Redis) — *deferred to a future version, see architecture.md*
- [ ] Observability — structured logging, tracing, metrics — *basic structured logging exists today (per-batch ingestion progress, Voyage token usage, planner/retriever routing decisions); the Prometheus/Grafana + OpenTelemetry/Tempo stack from decision #27 isn't wired up yet*
- [ ] Load testing & published benchmark numbers
- [ ] Docker Compose / deployment manifests — *a dev-only compose file exists for local Postgres/Qdrant (with named volumes, so data survives `docker compose down`); app containerization and deployment manifests not yet built*

## Getting Started

Requirements: [Docker](https://www.docker.com/) (for local Postgres + Qdrant), [uv](https://docs.astral.sh/uv/), a [Voyage AI](https://www.voyageai.com/) API key, and an [Anthropic](https://console.anthropic.com/) API key.

```bash
git clone https://github.com/MarwanHs/agentic-rag-platform
cd agentic-rag-platform
uv sync

cp .env.example .env
# fill in DATABASE_URL, QDRANT_URL, VOYAGE_API_KEY, ANTHROPIC_API_KEY -- the
# first two can stay as the docker-compose defaults shown in .env.example

docker compose up -d   # starts local Postgres + Qdrant

# run the API
uv run --env-file .env uvicorn api.main:app --reload --app-dir services/api/src

# run the ingestion worker in a separate terminal -- polls for queued jobs
uv run --env-file .env python -m worker.main
```

Submit a codebase for ingestion and poll until it's ready:

```bash
curl -X POST localhost:8000/jobs -d '{"url": "https://github.com/<owner>/<repo>"}'
curl localhost:8000/jobs/<job_id>   # poll until status is "ready"
```

Querying an ingested codebase (`POST /codebases/{id}/query`) isn't live yet — the agentic query pipeline is still in progress; see Progress above for exactly what's built and what's left.
