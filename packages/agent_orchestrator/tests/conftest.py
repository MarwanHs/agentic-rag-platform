from __future__ import annotations

from collections.abc import Callable

import pytest

from agent_orchestrator.planner import PlannerDecision
from shared.evidence import EvidenceItem


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
