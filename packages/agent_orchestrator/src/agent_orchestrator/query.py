"""Orchestrating function for single-shot query answering (decision #50).

Wires the planner, retriever-as-agent, code-navigation-as-agent, and the
critic-synthesizer together for `POST /codebases/{id}/query` (decision #31
keeps this separate from the future LangGraph conversation flow). Calls the
planner, branches on `agents_needed` to invoke whichever agents were
requested (sequentially, not in parallel -- deferred to v2), concatenates
their evidence, calls the critic over the combined list, then maps
`cited_evidence_indices` back into citations by indexing into that same
list.

Returns this module's own dataclasses, not `services/api`'s pydantic
models -- `agent_orchestrator` must not depend on `services/api` (the
workspace's dependency direction is services -> packages, never the
reverse). `services/api`'s endpoint handler does the trivial field-by-field
mapping into its own response model.

PlannerValidationError/CriticValidationError propagate uncaught: a
malformed model output is an anomaly, not a legitimate insufficient-evidence
case, and disguising it as a normal refusal response would hide a real bug
behind the same shape as an intended one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import psycopg
from qdrant_client import QdrantClient

from agent_orchestrator.code_navigation_agent import gather_code_navigation_evidence
from agent_orchestrator.critic import CriticClient
from agent_orchestrator.planner import PlannerClient
from agent_orchestrator.retriever_agent import ReformulationClient, gather_retriever_evidence
from rag_core.retrieval.embeddings import EmbeddingClient
from rag_core.retrieval.reranker import RerankClient
from shared.evidence import EvidenceItem


@dataclass(frozen=True, slots=True)
class QueryCitation:
    source: Literal["retriever", "code_navigation"]
    file_path: str
    line_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str | None
    refused: bool
    reason: str | None
    citations: list[QueryCitation]
    sources_used: list[Literal["retriever", "code_navigation"]]


def answer_query(
    question: str,
    codebase_id: str,
    conn: psycopg.Connection,
    qdrant_client: QdrantClient,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
    planner_client: PlannerClient,
    reformulation_client: ReformulationClient,
    critic_client: CriticClient,
    prior_evidence: list[EvidenceItem] | None = None,
    gather_retriever_evidence_fn=gather_retriever_evidence,
    gather_code_navigation_evidence_fn=gather_code_navigation_evidence,
) -> QueryResult:
    prior_evidence = prior_evidence or []

    decision = planner_client.plan(question, prior_evidence)

    evidence: list[EvidenceItem] = list(prior_evidence)
    if "retriever" in decision.agents_needed:
        evidence.extend(
            gather_retriever_evidence_fn(
                qdrant_client, codebase_id, decision.retriever_query, embedding_client, rerank_client, reformulation_client
            )
        )
    if "code_navigation" in decision.agents_needed:
        evidence.extend(
            gather_code_navigation_evidence_fn(conn, codebase_id, decision.code_navigation_symbols)
        )

    critic_decision = critic_client.synthesize(question, evidence)

    citations = [
        QueryCitation(
            source=evidence[i].source,
            file_path=evidence[i].file_path,
            line_range=evidence[i].line_range,
        )
        for i in critic_decision.cited_evidence_indices
    ]

    return QueryResult(
        answer=critic_decision.answer,
        refused=critic_decision.refused,
        reason=critic_decision.reason,
        citations=citations,
        sources_used=decision.agents_needed,
    )
