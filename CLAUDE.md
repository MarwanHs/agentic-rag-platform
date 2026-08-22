# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hosted, multi-agent RAG service that answers natural-language questions about a codebase with grounded, multi-hop, cited answers. A Claude-based planner routes each question to a semantic retriever (Voyage embeddings + Qdrant hybrid search), a deterministic code-navigation tool (Postgres symbol/reference lookups), or both, then a critic-synthesizer verifies sufficiency and either answers or refuses.

Read `docs/architecture.md` before making any non-trivial change. It's a decision log (40 numbered decisions) covering *why* each major choice was made and what alternatives were rejected — routing logic, schema design, datastore choice, model selection, etc. Don't reverse or contradict a numbered decision without flagging it; if new work touches one, reference the decision number in commit messages/comments the way existing code does (e.g. `# decisions #9, #11`).

**Current implementation state** (see README.md progress checklist for the authoritative list): ingestion job lifecycle, retrieval indexing, and code-navigation indexing are built. The agentic query pipeline (planner, retriever-as-agent, critic-synthesizer) and LangGraph conversation orchestration are **not implemented yet** — `POST /codebases/{id}/query` and `POST /conversations/{id}/messages` intentionally return `501` rather than fabricate a response, because `refused` in the response shape is specifically the critic's sufficiency verdict and no critic exists yet. Don't "fix" these 501s by stubbing a fake answer/refusal.

## Commands

This is a `uv` workspace with four members: `packages/shared`, `packages/rag_core`, `packages/agent_orchestrator`, `services/api`.

```bash
# install / sync all workspace members
uv sync

# run all tests
uv run pytest

# run a single package's tests
uv run pytest packages/rag_core
uv run pytest services/api

# run a single test
uv run pytest services/api/tests/test_jobs.py::test_create_and_get_job

# run the API locally
uv run uvicorn api.main:app --reload --app-dir services/api/src

# start local Postgres + Qdrant (required for most tests)
docker compose up -d
```

There is no configured lint/format/type-check tool (no ruff/mypy config in the repo) — don't assume one silently.

### Test datastore dependency

Most tests need Postgres (`services/api`) or Postgres + Qdrant (`packages/rag_core`) reachable. Both test suites probe reachability in `conftest.py` and **skip** (not fail) if the relevant service isn't up — if you see a batch of skips, run `docker compose up -d` first rather than assuming the tests are broken. Fake Voyage embedding/rerank clients (deterministic hash-based embeddings, token-overlap reranking) are used in tests instead of real API calls — see `FakeEmbeddingClient`/`FakeRerankClient` in the two `conftest.py` files — so tests never require `VOYAGE_API_KEY`.

Env vars (see `.env.example`): `DATABASE_URL`, `QDRANT_URL`, `VOYAGE_API_KEY`. Tests use `TEST_DATABASE_URL`/`TEST_QDRANT_URL` overrides if set, else the same defaults as `docker-compose.yml` (`postgresql://rag:rag@localhost:5432/rag`, `http://localhost:6333`).

## Architecture

### Workspace layout

- **`packages/shared`** — common types/config used across packages (currently minimal).
- **`packages/rag_core`** — ingestion and retrieval engine, no FastAPI/HTTP dependency:
  - `parsing/` — tree-sitter based structural parsing (Python only for v1), shared by both retrieval chunking and code-navigation indexing rather than built twice. Attaches docstrings/comments to their owning function/class node via tree structure, not proximity heuristics.
  - `retrieval/` — Voyage `voyage-code-3` embeddings, Qdrant hybrid search (dense + native BM25 sparse vectors, fused via Reciprocal Rank Fusion), Voyage reranking as a final pass.
  - `code_navigation/` — deterministic symbol/reference indexing and lookup (`find_definition`, `find_references`) over a Postgres relational schema (`symbols`, `symbol_references`). No LLM/embedding dependency by design — this is what naive RAG can't do (exact structural lookups), so it must work standalone even though in the running system it's only reachable via the planner.
  - `jobs/` — ingestion job row lifecycle (`create_job`, `get_job`, `update_status`, `list_ready_codebases`) against a Postgres `jobs` table. A job's own `id` doubles as the codebase id once `status = 'ready'` — there's no separate codebases table; a codebase is defined as "a job that succeeded."
  - `conversations/` — conversation row lifecycle (which codebase a conversation is scoped to); turn-by-turn history will live in LangGraph's Postgres checkpointer once that's implemented, not in this table.
- **`packages/agent_orchestrator`** — planner, retriever-as-agent, code-navigation-as-agent, critic-synthesizer, and the routing between them. Depends on `shared` + `rag_core`. Currently just a package skeleton — this is where the 501-stubbed query pipeline gets implemented.
- **`services/api`** — FastAPI HTTP layer: routers per concern (`jobs`, `codebases`, `conversations`), Pydantic request/response models in `api/models.py`, shared dependency wiring (Postgres connection, Qdrant client, Voyage clients) in `api/deps.py`. Schema migrations run at app startup via `lifespan` in `api/main.py` — **order matters**: jobs schema must run before conversations schema (FK dependency), see the comment in `main.py`.

### Key design invariants (from `docs/architecture.md`)

- **Explicit IDs, never inferred.** Codebase and conversation identification is always a passed ID, never guessed from question text.
- **All-or-nothing job readiness.** A job is `ready` or `failed`, never partially queryable — this is deliberate (avoids ever having one referenced file indexed and another not) and should not be relaxed casually.
- **Critic never re-triggers retrieval.** Retry/reformulation logic belongs to the retriever and code-navigation tool during the collection phase (decision #12, #32); the critic-synthesizer only evaluates a closed evidence set and either answers or refuses — it's a gate, not an orchestrator.
- **Planner is one structured-output call, not a tool-use loop** (decision #33) — it decides `agents_needed` + retriever query + code-navigation symbol name(s) up front; it never calls agents mid-reasoning itself. Code-navigation performs pure lookup with no reasoning of its own (decision #34) — all NL-to-symbol translation happens in the planner.
- **Single-shot query vs. conversation are separate endpoints, deliberately not unified** (decision #31) — `POST /codebases/{id}/query` stays plain-Python/stateless; `POST /conversations/{id}/messages` is the LangGraph-backed path. Don't route single-shot queries through LangGraph "for consistency."
- **Unified evidence schema** (`EvidenceItem`, decision #35) is shared between retriever and code-navigation results before reaching the critic-synthesizer, and maps directly onto the citation format in API responses — check this shape before adding new fields to either agent's output.
- **Model split** (decision #36): Haiku for planner + retriever-reformulation calls, Sonnet for the critic-synthesizer (highest-stakes call, worth the stronger model).
- **No LangChain, no standing graph DB, no Kafka** — each was deliberately evaluated and rejected (decisions #21, #37, #40) because the system's actual query/throughput patterns don't need them; don't reach for them without re-reading the relevant decision first.
- Vector storage is one Qdrant collection per codebase (physical isolation, not metadata filtering) — see decision #14 before changing this.
