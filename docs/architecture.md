# Architecture

This document describes the system design for `agentic-rag-platform`, including the reasoning behind key decisions and the alternatives that were considered and deferred.

## What the system does

A hosted, multi-agent service that helps engineers understand a codebase. A user submits a public GitHub URL for ingestion; once indexed, they can ask natural-language questions and get grounded, multi-hop, cited answers — routed through a Claude-based planner to a semantic retriever, a deterministic code-navigation tool, or both, then verified and synthesized into a final response.

## Ingestion flow

![Ingestion pipeline architecture](./architecture/ingestion_pipeline.svg)

A client submits a codebase URL. The API acknowledges immediately with a job ID rather than blocking, because cloning, parsing, and embedding a real repository can take longer than a single HTTP request should stay open. A job queue processes the work in batches — not per-file, not as one giant stage — moving through clone, parse, embed, and index. Status is tracked per step in a job status table, which the client polls to check progress. A job is only marked `ready` once every step has fully succeeded; there is no partial or "ready with gaps" state. Once ready, the codebase's vectors live in their own isolated collection in the vector store.

## Query flow

![Query answering architecture](./architecture/query_answering_flow.svg)

A client sends a question along with an explicit codebase ID (never inferred from the question text). A Claude-based planner reads the question and decides which agents are needed: a semantic retriever (embeds the query with Voyage, searches the codebase's vector collection), a deterministic code-navigation agent (symbol lookup, call-graph traversal, no LLM dependency), or both — branching rather than looping, with no path permanently closed off. Each agent owns its own bounded retry logic internally before reporting a final result. Once collection is complete, a single critic-synthesizer call reviews the gathered evidence, verifies it's sufficient, and either produces a cited answer or refuses — collection is never re-triggered after this point. The whole request is a single blocking HTTP call; given the bounded, few-second latency of two Claude calls plus fast lookups, there is no async job pattern or streaming on the query path.

## Design decisions

Each decision below states what was chosen, and — where relevant — what was considered and deferred instead, and why.

1. **Hosted service, not a local tool.** Chosen deliberately over a CLI/localhost tool because it forces real architectural problems — async ingestion, durable job state, multi-tenant isolation — that make the scaling and clean-code goals of this project credible rather than theoretical.

2. **v1 scope: public GitHub URLs only, no private-repo authentication.** An explicit, stated limitation rather than an oversight — avoids taking on credential handling and secrets management before the core pipeline is proven.

3. **Ingestion is asynchronous.** The client receives an ack and a job ID immediately; the actual clone/parse/embed/index work happens out of band, because it can't reliably fit inside a single request/response cycle.

4. **Job status is discovered via polling, not push (webhook/SSE/websocket).** The client's runtime is unknown and unconstrained — a CLI, a script, anything — and push mechanisms assume things about the client (a reachable endpoint, a held-open connection) that don't hold universally. Polling also avoids fighting the stateless-server scaling goal.

5. **Job status is tracked per step, not as a single binary flag.** Gives the client visibility into pipeline progress (clone → parse → embed → index) rather than an opaque wait.

6. **Job readiness is all-or-nothing.** A job is either fully `ready` or `failed` — never partially queryable. Incomplete grounding is treated as worse than no answer at all; this is central to the system's trust story and later resolves a class of cross-file-reference concerns for free (a query can never hit a state where one referenced file is indexed and another isn't).

7. **Ingestion is processed in batches, not per-file streamed.** Per-file streaming was considered for maximum retry granularity and pipeline throughput, but dropped due to the complexity of concurrent rate-limit handling against the embedding API. Batching keeps most of the benefit — partial-progress retry, no full-pipeline restart on failure — with bounded, controllable concurrency.

8. **Codebase identification is always an explicit ID, never inferred from question text.** Deterministic and unambiguous. Implies a `GET /codebases` listing endpoint is needed so clients can discover valid IDs.

9. **Query routing is agent-driven (the Claude-based planner decides), not rule-based/regex classification.** Avoids building and maintaining a brittle keyword/pattern classification layer to cover open-ended user intent.

10. **The code-navigation agent is deterministic and has no hard Claude/Voyage dependency in its implementation** — but in the running system it is only reachable via the Claude-based planner, since routing itself requires a planner call. It is decoupled by design and testable in isolation, not reachable standalone in deployment.

11. **Critic and synthesizer are combined into a single Claude call**, not split into separate agents. Splitting was considered for independent judgment — avoiding the bias of a generation-focused prompt toward completing the task — but the cost of a guaranteed extra call per query outweighed the benefit, given disciplined prompting (explicit refuse-over-fill instructions, mandatory per-claim citation) can achieve similar grounding at lower cost. Accepted as a known, documented tradeoff rather than a fully solved problem.

12. **Retry logic lives inside the retriever and code-navigation agents, in the collection phase — not at the critic.** The critic never triggers additional data collection; it only evaluates a closed, final set of evidence. This keeps its role a simple gate (answer or refuse) rather than an orchestrator.

13. **Query answering is a single blocking HTTP request — no async job pattern, no token/progress streaming.** Unlike ingestion, a query's critical path (two bounded Claude calls plus fast index/graph lookups) has a predictable, few-second latency well within HTTP timeout limits. Progress streaming (à la Claude Code) was considered but rejected: that pattern solves open-ended, unpredictable-duration waits, which this system doesn't have. Deferred as a future item if p95 latency ever grows materially.

14. **Vector storage: a separate collection per codebase, not a shared index with metadata filtering.** Metadata filtering is a legitimate, widely used multi-tenant pattern and would likely be preferable at large tenant scale (thousands+), where per-tenant physical isolation becomes an operational burden. At this project's scale, separate collections were chosen because clean reingest/delete of a single codebase is a first-class requirement, and structural isolation removes reliance on every future code path correctly applying a filter. This is a scale-bounded decision, not a universal best practice, and would need revisiting if tenant count grew significantly.

15. **The raw cloned repository is discarded after ingestion**, not retained in object storage. Ephemeral local disk is used as scratch space during cloning/parsing only; nothing downstream (retrieval, code-navigation) queries the filesystem directly, and since v1 sources are always public GitHub URLs, the origin is always re-fetchable on demand. Deferred — would become necessary if private or non-GitHub sources are added later, where the origin isn't guaranteed re-fetchable.

16. **Evaluation uses a golden dataset built on a known repository (the project's own)**, with hand-defined question/expected-evidence pairs, graded by an LLM-as-judge rather than exact-match scoring, since answers are prose.

17. **A semantic query/answer cache was considered and deferred to a future version.** The cost argument favors it (embedding a query via Voyage is near-free; the expensive part is the Claude reasoning pipeline, which a cache hit would skip entirely) — but it was deferred because a false-positive cache hit means silently serving a wrong answer for a different question, directly undermining the system's grounding and trust goals, and because there isn't yet real usage data to responsibly tune a similarity threshold against.

18. **Observability captures per-query, per-step data across the full pipeline**: planner reasoning and routing decisions, retriever results and similarity scores, code-navigation results (including not-found cases), critic-synthesizer sufficiency judgment and final output, and per-step/total timing. This serves two distinct purposes — debugging low eval scores, and substantiating the project's scaling and latency claims with real measured numbers rather than assumptions.

## Not yet resolved

- Observability stack (tracing/logging tooling choice)
- Detailed API surface (full endpoint list, request/response schemas)
- Deployment topology (containerization, orchestration)
