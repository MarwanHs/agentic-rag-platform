from __future__ import annotations

import socket
from collections.abc import Callable
from urllib.parse import urlparse

import psycopg
import pytest
from fastapi.testclient import TestClient

from agent_orchestrator.critic import CriticDecision
from agent_orchestrator.planner import PlannerDecision
from agent_orchestrator.retriever_agent import ReformulationDecision
from api.deps import (
    DATABASE_URL,
    get_critic_client,
    get_embedding_client,
    get_planner_client,
    get_reformulation_client,
    get_rerank_client,
)
from api.main import app
from rag_core.retrieval.qdrant_index import SearchResult
from rag_core.retrieval.reranker import RerankResult
from shared.evidence import EvidenceItem


def _is_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_datastores() -> None:
    pg = urlparse(DATABASE_URL)
    if not _is_reachable(pg.hostname or "localhost", pg.port or 5432):
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}; run `docker compose up -d postgres`.")


class FakeEmbeddingClient:
    """Kept wired into the `client` fixture below even though no route
    currently calls the embedding client -- cheap insurance against a real
    Voyage call slipping in once the query endpoint starts using retrieval
    again (next milestone).
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeRerankClient:
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        q_tokens = set(query.lower().split())
        scored = sorted(
            range(len(documents)),
            key=lambda i: -len(q_tokens & set(documents[i].lower().split())),
        )[:top_k]
        return [RerankResult(index=i, score=1.0 - rank * 0.01) for rank, i in enumerate(scored)]


class FakePlannerClient:
    """Deterministic stand-in for AnthropicPlannerClient (duplicated per the
    convention in agent_orchestrator's own conftest.py -- not imported
    across the package boundary).
    """

    def __init__(
        self,
        decision: PlannerDecision | Callable[[str, list[EvidenceItem]], PlannerDecision] | None = None,
    ) -> None:
        self._decision = decision or PlannerDecision(
            reasoning="default fake decision",
            agents_needed=[],
            retriever_query=None,
            code_navigation_symbols=[],
            existing_context_sufficient=False,
        )
        self.calls: list[tuple[str, list[EvidenceItem]]] = []

    def plan(self, question: str, prior_evidence: list[EvidenceItem]) -> PlannerDecision:
        self.calls.append((question, prior_evidence))
        if callable(self._decision):
            return self._decision(question, prior_evidence)
        return self._decision


class FakeReformulationClient:
    """Deterministic stand-in for AnthropicReformulationClient. Not expected
    to be exercised by any endpoint test in this file today (the endpoint
    tests avoid the retriever path to sidestep the Qdrant-collection
    gotcha), kept wired for completeness and to match the existing
    get_embedding_client/get_rerank_client override pattern.
    """

    def __init__(
        self,
        decision: ReformulationDecision | Callable[[str, list[SearchResult]], ReformulationDecision] | None = None,
    ) -> None:
        self._decision = decision or ReformulationDecision(
            reasoning="default fake reformulation",
            reformulated_query="reformulated query",
        )
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def reformulate(self, first_pass_query: str, first_pass_results: list[SearchResult]) -> ReformulationDecision:
        self.calls.append((first_pass_query, first_pass_results))
        if callable(self._decision):
            return self._decision(first_pass_query, first_pass_results)
        return self._decision


class FakeCriticClient:
    """Deterministic stand-in for AnthropicCriticClient (duplicated per the
    convention in agent_orchestrator's own conftest.py -- not imported
    across the package boundary).
    """

    def __init__(
        self,
        decision: CriticDecision | Callable[[str, list[EvidenceItem]], CriticDecision] | None = None,
    ) -> None:
        self._decision = decision or CriticDecision(
            answer=None,
            refused=True,
            reason="default fake refusal",
            cited_evidence_indices=[],
        )
        self.calls: list[tuple[str, list[EvidenceItem]]] = []

    def synthesize(self, question: str, evidence: list[EvidenceItem]) -> CriticDecision:
        self.calls.append((question, evidence))
        if callable(self._decision):
            return self._decision(question, evidence)
        return self._decision


@pytest.fixture()
def fake_planner_client() -> FakePlannerClient:
    return FakePlannerClient()


@pytest.fixture()
def fake_reformulation_client() -> FakeReformulationClient:
    return FakeReformulationClient()


@pytest.fixture()
def fake_critic_client() -> FakeCriticClient:
    return FakeCriticClient()


@pytest.fixture()
def client(fake_planner_client, fake_reformulation_client, fake_critic_client):
    app.dependency_overrides[get_embedding_client] = lambda: FakeEmbeddingClient()
    app.dependency_overrides[get_rerank_client] = lambda: FakeRerankClient()
    app.dependency_overrides[get_planner_client] = lambda: fake_planner_client
    app.dependency_overrides[get_reformulation_client] = lambda: fake_reformulation_client
    app.dependency_overrides[get_critic_client] = lambda: fake_critic_client
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def pg_conn():
    conn = psycopg.connect(DATABASE_URL)
    yield conn
    conn.close()
