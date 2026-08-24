from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_orchestrator import planner as planner_module
from agent_orchestrator.planner import (
    AnthropicPlannerClient,
    PlannerDecision,
    PlannerValidationError,
)
from shared.evidence import EvidenceItem


@dataclass
class FakeToolUseBlock:
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list[FakeToolUseBlock] = field(default_factory=list)


class FakeMessages:
    def __init__(self, response: FakeMessage) -> None:
        self._response = response
        self.create_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> FakeMessage:
        self.create_kwargs = kwargs
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: FakeMessage) -> None:
        self.messages = FakeMessages(response)


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any]) -> FakeAnthropicClient:
    response = FakeMessage(content=[FakeToolUseBlock(input=tool_input)])
    fake_client = FakeAnthropicClient(response)

    class FakeAnthropicModule:
        @staticmethod
        def Anthropic(api_key: str | None = None) -> FakeAnthropicClient:
            return fake_client

    monkeypatch.setattr(planner_module, "anthropic", FakeAnthropicModule)
    return fake_client


RETRIEVER_ONLY = {
    "reasoning": "Question needs a semantic search over the codebase.",
    "agents_needed": ["retriever"],
    "retriever_query": "how does the ingestion job queue claim work",
    "code_navigation_symbols": [],
    "existing_context_sufficient": False,
}

CODE_NAV_ONLY = {
    "reasoning": "Exact symbol lookup requested.",
    "agents_needed": ["code_navigation"],
    "retriever_query": None,
    "code_navigation_symbols": ["EvidenceItem"],
    "existing_context_sufficient": False,
}

BOTH_AGENTS = {
    "reasoning": "Needs both semantic search and exact symbol lookup.",
    "agents_needed": ["retriever", "code_navigation"],
    "retriever_query": "how is the planner routed",
    "code_navigation_symbols": ["PlannerDecision"],
    "existing_context_sufficient": False,
}

EMPTY_OFF_TOPIC = {
    "reasoning": "Question is unrelated to this codebase.",
    "agents_needed": [],
    "retriever_query": None,
    "code_navigation_symbols": [],
    "existing_context_sufficient": False,
}

EMPTY_EXISTING_CONTEXT_SUFFICIENT = {
    "reasoning": "Prior evidence already answers this follow-up.",
    "agents_needed": [],
    "retriever_query": None,
    "code_navigation_symbols": [],
    "existing_context_sufficient": True,
}


@pytest.mark.parametrize(
    "tool_input",
    [RETRIEVER_ONLY, CODE_NAV_ONLY, BOTH_AGENTS, EMPTY_OFF_TOPIC, EMPTY_EXISTING_CONTEXT_SUFFICIENT],
)
def test_plan_parses_consistent_decisions(monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any]) -> None:
    _install_fake_anthropic(monkeypatch, tool_input)
    client = AnthropicPlannerClient(api_key="unused")

    decision = client.plan("some question", [])

    assert decision == PlannerDecision(
        reasoning=tool_input["reasoning"],
        agents_needed=tool_input["agents_needed"],
        retriever_query=tool_input["retriever_query"],
        code_navigation_symbols=tool_input["code_navigation_symbols"],
        existing_context_sufficient=tool_input["existing_context_sufficient"],
    )


def test_plan_forces_route_query_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, RETRIEVER_ONLY)
    client = AnthropicPlannerClient(api_key="unused", model="claude-haiku-4-5-20251001")

    client.plan("some question", [])

    kwargs = fake_client.messages.create_kwargs
    assert kwargs is not None
    assert kwargs["tool_choice"] == {"type": "tool", "name": "route_query"}
    assert kwargs["tools"][0]["name"] == "route_query"
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


def test_plan_omits_prior_evidence_from_message_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, RETRIEVER_ONLY)
    client = AnthropicPlannerClient(api_key="unused")

    client.plan("some question", [])

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "Prior evidence" not in message
    assert "some question" in message


def test_plan_includes_prior_evidence_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, EMPTY_EXISTING_CONTEXT_SUFFICIENT)
    client = AnthropicPlannerClient(api_key="unused")
    prior_evidence = [
        EvidenceItem(
            source="retriever",
            file_path="rag_core/jobs/store.py",
            line_range=(10, 20),
            content="def claim_next_job(...): ...",
            score=0.91,
        )
    ]

    client.plan("what does it do next", prior_evidence)

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "Prior evidence" in message
    assert "rag_core/jobs/store.py:10-20" in message


@pytest.mark.parametrize(
    "tool_input",
    [
        {
            "reasoning": "retriever requested but no query given",
            "agents_needed": ["retriever"],
            "retriever_query": None,
            "code_navigation_symbols": [],
            "existing_context_sufficient": False,
        },
        {
            "reasoning": "query given but retriever not requested",
            "agents_needed": [],
            "retriever_query": "some query",
            "code_navigation_symbols": [],
            "existing_context_sufficient": False,
        },
        {
            "reasoning": "code_navigation requested but no symbols given",
            "agents_needed": ["code_navigation"],
            "retriever_query": None,
            "code_navigation_symbols": [],
            "existing_context_sufficient": False,
        },
        {
            "reasoning": "symbols given but code_navigation not requested",
            "agents_needed": [],
            "retriever_query": None,
            "code_navigation_symbols": ["Foo"],
            "existing_context_sufficient": False,
        },
    ],
)
def test_plan_raises_on_inconsistent_decision(monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any]) -> None:
    _install_fake_anthropic(monkeypatch, tool_input)
    client = AnthropicPlannerClient(api_key="unused")

    with pytest.raises(PlannerValidationError):
        client.plan("some question", [])


def test_fake_planner_client_returns_fixed_decision(fake_planner_client) -> None:
    decision = PlannerDecision(
        reasoning="fixed",
        agents_needed=["retriever"],
        retriever_query="q",
        code_navigation_symbols=[],
        existing_context_sufficient=False,
    )
    fake_planner_client._decision = decision

    result = fake_planner_client.plan("anything", [])

    assert result == decision
    assert fake_planner_client.calls == [("anything", [])]


def test_fake_planner_client_supports_callable_decision(fake_planner_client) -> None:
    def decide(question: str, prior_evidence: list[EvidenceItem]) -> PlannerDecision:
        if prior_evidence:
            return PlannerDecision(
                reasoning="reuse prior evidence",
                agents_needed=[],
                retriever_query=None,
                code_navigation_symbols=[],
                existing_context_sufficient=True,
            )
        return PlannerDecision(
            reasoning="fresh question",
            agents_needed=["retriever"],
            retriever_query=question,
            code_navigation_symbols=[],
            existing_context_sufficient=False,
        )

    fake_planner_client._decision = decide

    fresh = fake_planner_client.plan("what does X do", [])
    assert fresh.agents_needed == ["retriever"]

    prior = [
        EvidenceItem(source="retriever", file_path="a.py", line_range=(1, 2), content="...")
    ]
    followup = fake_planner_client.plan("and then?", prior)
    assert followup.existing_context_sufficient is True
