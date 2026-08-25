"""Retriever-as-agent: confidence-gated semantic search evidence gathering (decision #45).

Wraps rag_core's existing hybrid_search (decisions #14, #19, #20) with the
two-stage mechanism decision #32 requires: a top-1-rerank-score confidence
gate, then at most one forced-tool-use reformulation call if the gate fails.
Whatever the second pass returns is final -- no re-gating, no loop.

This agent never sees the original user question or conversation history,
only the planner's self-contained retriever_query string (decision #42).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
from qdrant_client import QdrantClient

from rag_core.retrieval.embeddings import EmbeddingClient
from rag_core.retrieval.qdrant_index import SearchResult, hybrid_search
from rag_core.retrieval.reranker import RerankClient
from shared.evidence import EvidenceItem

DEFAULT_REFORMULATION_MODEL = "claude-haiku-4-5-20251001"

# Decision #45: env-configurable, not a hardcoded constant like
# DEFAULT_EMBEDDING_MODEL/DEFAULT_RERANK_MODEL/DEFAULT_PLANNER_MODEL. This is
# a hosted service with potentially multiple independent deployers (decision
# #1) and this threshold is a real cost/latency/quality dial -- a stricter
# operator pays for more reformulation calls in exchange for fewer mediocre
# first-pass results slipping through. The 0.5 default is an explicit
# placeholder anchored to one real observed score (a known-good query in this
# project's own smoke testing scored 0.746), not a tuned value -- pending
# decision #16's eval harness.
RERANK_CONFIDENCE_THRESHOLD = float(os.environ.get("RERANK_CONFIDENCE_THRESHOLD", "0.5"))

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You help a codebase question-answering system's retriever recover from a \
weak first-pass search. The retriever searches a codebase via hybrid \
semantic (embedding) + lexical (BM25) search over code chunks. The first \
attempt returned weak matches.

Given the first-pass query and its top results (file path, qualified name, \
snippet, score), produce a better query. Consider: more specific \
terminology, different phrasing closer to how the concept might actually \
appear in code or docstrings, or a narrower or broader scope.

Always call the reformulate_query tool with your decision.\
"""

REFORMULATE_QUERY_TOOL: dict = {
    "name": "reformulate_query",
    "description": "Reformulate a semantic search query that returned weak first-pass results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why the first pass was weak and what changed.",
            },
            "reformulated_query": {
                "type": "string",
                "description": "The new search query to try.",
            },
        },
        "required": ["reasoning", "reformulated_query"],
    },
}


@dataclass(frozen=True, slots=True)
class ReformulationDecision:
    reasoning: str
    reformulated_query: str


class ReformulationClient(Protocol):
    def reformulate(self, first_pass_query: str, first_pass_results: list[SearchResult]) -> ReformulationDecision: ...


def _passes_confidence_gate(results: list[SearchResult]) -> bool:
    """Decision #45: gate on the top-1 rerank score only, not an average --
    the real question is "is there at least one strong piece of evidence,"
    which an average would dilute with weaker lower-ranked results that
    don't matter if the top hit clears the bar.
    """
    if not results:
        return False
    return results[0].score >= RERANK_CONFIDENCE_THRESHOLD


def _format_search_results(results: list[SearchResult]) -> str:
    lines = []
    for result in results:
        payload = result.payload
        location = f"{payload['file_path']}:{payload['start_line']}-{payload['end_line']}"
        snippet = payload["text"].strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(
            f"- {location} qualified_name={payload['qualified_name']} score={result.score}: {snippet}"
        )
    return "\n".join(lines)


def _build_user_message(first_pass_query: str, first_pass_results: list[SearchResult]) -> str:
    if not first_pass_results:
        return f"First-pass query: {first_pass_query}\n\nFirst-pass results: none."
    return (
        f"First-pass query: {first_pass_query}\n\n"
        "First-pass results:\n"
        f"{_format_search_results(first_pass_results)}"
    )


class AnthropicReformulationClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_REFORMULATION_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def reformulate(self, first_pass_query: str, first_pass_results: list[SearchResult]) -> ReformulationDecision:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[REFORMULATE_QUERY_TOOL],
            tool_choice={"type": "tool", "name": "reformulate_query"},
            messages=[{"role": "user", "content": _build_user_message(first_pass_query, first_pass_results)}],
        )

        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_input: dict[str, Any] = tool_use.input

        logger.info("retriever reformulation decision: %s", tool_input)

        return ReformulationDecision(
            reasoning=tool_input["reasoning"],
            reformulated_query=tool_input["reformulated_query"],
        )


def _to_evidence(results: list[SearchResult]) -> list[EvidenceItem]:
    evidence = []
    for result in results:
        payload = result.payload
        evidence.append(
            EvidenceItem(
                source="retriever",
                file_path=payload["file_path"],
                line_range=(payload["start_line"], payload["end_line"]),
                content=payload["text"],
                score=result.score,
            )
        )
    return evidence


def gather_retriever_evidence(
    qdrant_client: QdrantClient,
    codebase_id: str,
    query: str,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
    reformulation_client: ReformulationClient,
) -> list[EvidenceItem]:
    first_pass = hybrid_search(qdrant_client, codebase_id, query, embedding_client, rerank_client)

    if _passes_confidence_gate(first_pass):
        logger.info(
            "retriever confidence gate passed: codebase=%s query=%r top_score=%s",
            codebase_id,
            query,
            first_pass[0].score,
        )
        return _to_evidence(first_pass)

    logger.info(
        "retriever confidence gate failed: codebase=%s query=%r top_score=%s",
        codebase_id,
        query,
        first_pass[0].score if first_pass else "no results",
    )
    decision = reformulation_client.reformulate(query, first_pass)
    logger.info("retriever reformulated query: codebase=%s reformulated_query=%r", codebase_id, decision.reformulated_query)

    # Decision #32/#45's explicit bound: exactly one retry. Whatever comes
    # back is final regardless of its score -- the gate is not re-checked.
    second_pass = hybrid_search(
        qdrant_client, codebase_id, decision.reformulated_query, embedding_client, rerank_client
    )
    return _to_evidence(second_pass)
