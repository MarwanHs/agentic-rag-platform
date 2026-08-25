from __future__ import annotations

import os
import socket
from collections.abc import Callable
from urllib.parse import urlparse

import psycopg
import pytest

from agent_orchestrator.planner import PlannerDecision
from shared.evidence import EvidenceItem

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")


def _is_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def pg_conn():
    parsed = urlparse(DATABASE_URL)
    if not _is_reachable(parsed.hostname or "localhost", parsed.port or 5432):
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}; run `docker compose up -d postgres`.")
    conn = psycopg.connect(DATABASE_URL)
    yield conn
    conn.close()


class FakePlannerClient:
    """Deterministic stand-in for AnthropicPlannerClient.

    Configure with either a fixed PlannerDecision (returned for every call)
    or a callable(question, prior_evidence) -> PlannerDecision for
    per-call control.
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


@pytest.fixture()
def fake_planner_client() -> FakePlannerClient:
    return FakePlannerClient()
