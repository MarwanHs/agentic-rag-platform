# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hosted, multi-agent RAG service that answers natural-language questions about a codebase with grounded, multi-hop, cited answers. A Claude-based planner routes each question to a semantic retriever (Voyage embeddings + Qdrant hybrid search), a deterministic code-navigation tool (Postgres symbol/reference lookups), or both, then a critic-synthesizer verifies sufficiency and either answers or refuses.

Read `docs/architecture.md` before making any non-trivial change. It's a decision log (47 numbered decisions) covering *why* each major choice was made and what alternatives were rejected — routing logic, schema design, datastore choice, model selection, etc. Don't reverse or contradict a numbered decision without flagging it; if new work touches one, reference the decision number in commit messages/comments the way existing code does (e.g. `# decisions #9, #11`).

**Current implementation state** (see README.md progress checklist, though it lags this file — trust this file and the code over it): ingestion is fully built end-to-end — job lifecycle, a Postgres-native job queue, the standalone worker that executes clone/parse/embed/index, retrieval indexing, and code-navigation indexing. Of the agentic query pipeline, the **planner** (`agent_orchestrator/planner.py`, a forced-tool-use Claude call, decision #42), **code-navigation-as-agent** (`agent_orchestrator/code_navigation_agent.py`, decisions #43, #44), and **retriever-as-agent** (`agent_orchestrator/retriever_agent.py`, decision #45) are implemented; the critic-synthesizer is not, so `POST /codebases/{id}/query` and `POST /conversations/{id}/messages` still intentionally return `501` rather than fabricate a response — `refused` in the response shape is specifically the critic's sufficiency verdict and no critic exists yet. LangGraph conversation orchestration is also not implemented yet. Don't "fix" these 501s by stubbing a fake answer/refusal.

## Commands

This is a `uv` workspace with five members: `packages/shared`, `packages/rag_core`, `packages/agent_orchestrator`, `services/api`, `services/worker`.

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

# run the API locally (--env-file loads .env into the process before Python
# starts -- see decision #47 for why this isn't a load_dotenv() call instead)
uv run --env-file .env uvicorn api.main:app --reload --app-dir services/api/src

# run the ingestion worker locally (polls the jobs table and executes clone/parse/embed/index)
uv run --env-file .env python -m worker.main

# start local Postgres + Qdrant (required for most tests)
docker compose up -d
```

Copy `.env.example` to `.env` and fill in `DATABASE_URL`, `QDRANT_URL`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY` before running the commands above. `.env` is gitignored; `.env.example` should only ever contain blank placeholders -- never fill in real values there.

There is no configured lint/format/type-check tool (no ruff/mypy config in the repo) — don't assume one silently.

### Test datastore dependency

Most tests need Postgres (`services/api`) or Postgres + Qdrant (`packages/rag_core`) reachable. Both test suites probe reachability in `conftest.py` and **skip** (not fail) if the relevant service isn't up — if you see a batch of skips, run `docker compose up -d` first rather than assuming the tests are broken. Fake Voyage embedding/rerank clients (deterministic hash-based embeddings, token-overlap reranking) are used in tests instead of real API calls — see `FakeEmbeddingClient`/`FakeRerankClient` in the two `conftest.py` files — so tests never require `VOYAGE_API_KEY`.

Env vars (see `.env.example`): `DATABASE_URL`, `QDRANT_URL`, `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY` (the last is required once any agent_orchestrator code runs -- planner and retriever-as-agent both call Claude). Tests use `TEST_DATABASE_URL`/`TEST_QDRANT_URL` overrides if set, else the same defaults as `docker-compose.yml` (`postgresql://rag:rag@localhost:5432/rag`, `http://localhost:6333`), and never need `VOYAGE_API_KEY`/`ANTHROPIC_API_KEY` -- both are mocked at the client boundary in tests (`FakeEmbeddingClient`/`FakeRerankClient`, and the mocked `anthropic.Anthropic` in `test_planner.py`/`test_retriever_agent.py`). `RERANK_CONFIDENCE_THRESHOLD` (default `0.5`) tunes the retriever-as-agent's confidence gate (decision #45); `EMBED_BATCH_SIZE` (default `64`) tunes ingestion's Voyage `embed_documents` batch size (decision #46). Loading: `uv run --env-file .env <command>`, not `python-dotenv` in application code (decision #47).

## Architecture

### Workspace layout

- **`packages/shared`** — common types/config used across packages: currently just `evidence.py`, the `EvidenceItem` schema (decision #35) that unifies retriever and code-navigation output before it reaches the critic-synthesizer.
- **`packages/rag_core`** — ingestion and retrieval engine, no FastAPI/HTTP dependency:
  - `parsing/` — tree-sitter based structural parsing (Python only for v1), shared by both retrieval chunking and code-navigation indexing rather than built twice. Attaches docstrings/comments to their owning function/class node via tree structure, not proximity heuristics.
  - `retrieval/` — Voyage `voyage-code-3` embeddings, Qdrant hybrid search (dense + native BM25 sparse vectors, fused via Reciprocal Rank Fusion), Voyage reranking as a final pass.
  - `code_navigation/` — deterministic symbol/reference indexing and lookup (`find_definition`, `find_references`) over a Postgres relational schema (`symbols`, `symbol_references`, the latter carrying a `source_text` column populated at ingestion time, decision #43). No LLM/embedding dependency by design — this is what naive RAG can't do (exact structural lookups), so it must work standalone even though in the running system it's only reachable via the planner.
  - `jobs/` — ingestion job row lifecycle (`create_job`, `get_job`, `update_status`, `list_ready_codebases`, `dequeue_next_job`) against a Postgres `jobs` table. A job's own `id` doubles as the codebase id once `status = 'ready'` — there's no separate codebases table; a codebase is defined as "a job that succeeded." Per-job execution state (batch list, position, attempt counts) lives in a `pipeline_state JSONB` column, not a normalized table (decision #39).
  - `ingestion/` — the actual clone → parse → embed/index → code-navigation-index pipeline (`pipeline.py::run_pipeline`) executed by `services/worker` for one dequeued job at a time (decisions #3, #6, #7, #37–39).
  - `conversations/` — conversation row lifecycle (which codebase a conversation is scoped to); turn-by-turn history will live in LangGraph's Postgres checkpointer once that's implemented, not in this table.
- **`packages/agent_orchestrator`** — planner, retriever-as-agent, code-navigation-as-agent, critic-synthesizer, and the routing between them. Depends on `shared` + `rag_core`. Implemented so far: the **planner** (`planner.py` — a forced-tool-use Claude call producing a `PlannerDecision`, decisions #33, #34, #36, #41, #42), **code-navigation-as-agent** (`code_navigation_agent.py` — a plain function with no LLM/API dependency that calls both `find_definition` and `find_references` for every planner-selected symbol name and maps results onto `EvidenceItem`, decisions #43, #44), and **retriever-as-agent** (`retriever_agent.py` — wraps `rag_core`'s `hybrid_search` with decision #32's two-stage mechanism: a top-1-rerank-score confidence gate (`RERANK_CONFIDENCE_THRESHOLD`, env-configurable, default `0.5`), and if it fails, exactly one forced-tool-use Claude reformulation call (mirroring the planner's `PlannerClient`/`AnthropicPlannerClient` shape) before returning whatever the retry produces, unconditionally — decision #45). The critic-synthesizer is not yet built, which is why the query endpoints still return `501`.
- **`services/api`** — FastAPI HTTP layer: routers per concern (`jobs`, `codebases`, `conversations`), Pydantic request/response models in `api/models.py`, shared dependency wiring (Postgres connection, Qdrant client, Voyage clients) in `api/deps.py`. Schema migrations run at app startup via `lifespan` in `api/main.py` — **order matters**: jobs schema must run before conversations schema (FK dependency), see the comment in `main.py`.
- **`services/worker`** — standalone ingestion worker process (`worker/main.py::run_forever`): polls the `jobs` table via `dequeue_next_job`'s `SELECT ... FOR UPDATE SKIP LOCKED` (decision #37 — Postgres-native queue, no separate broker) and runs `rag_core.ingestion.pipeline.run_pipeline` to completion for one job before dequeuing the next. Default concurrency is one job at a time (`MAX_CONCURRENT_JOBS`, decision #38 — not yet wired to actually run jobs concurrently). No crash recovery: a worker that dies mid-job leaves that job stuck in a non-`queued` status, deliberately unaddressed for now (see the module docstring).

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
- **The job queue is Postgres itself** (`FOR UPDATE SKIP LOCKED` on `jobs`, decision #37), not a message broker — don't introduce arq/Celery/Kafka for ingestion work.
- **`agents_needed` from the planner can legitimately be empty** (decision #41) — either the question is off-topic (single-shot) or prior conversation evidence already suffices (follow-up), distinguished by the separate `existing_context_sufficient` field (decision #42) since "prior evidence exists" doesn't imply "prior evidence is relevant to this question." The critic must still run in both cases and must be prompted to refuse on empty evidence rather than answer from pretraining — this is required, not incidental.
- **The planner uses forced tool-use** (`tool_choice` pinned to `route_query`, decision #42), not prompted free-text JSON — preserves the "one structured-output call" invariant (decision #33) without a parse-and-retry loop. Follow the same pattern for the critic-synthesizer when it's built.
