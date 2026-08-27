from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_orchestrator import critic as critic_module
from agent_orchestrator.critic import (
    AnthropicCriticClient,
    CriticDecision,
    CriticValidationError,
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

    monkeypatch.setattr(critic_module, "anthropic", FakeAnthropicModule)
    return fake_client


RETRIEVER_EVIDENCE = EvidenceItem(
    source="retriever",
    file_path="rag_core/jobs/store.py",
    line_range=(10, 20),
    content="def claim_next_job(...): ...",
    score=0.91,
)

CODE_NAV_EVIDENCE = EvidenceItem(
    source="code_navigation",
    file_path="agent_orchestrator/planner.py",
    line_range=(119, 125),
    content="class PlannerDecision: ...",
    symbol_name="PlannerDecision",
    symbol_kind="class",
    reference_kind="definition",
)

ANSWERED = {
    "answer": "The job queue claims work via SELECT ... FOR UPDATE SKIP LOCKED [0].",
    "refused": False,
    "reason": None,
    "cited_evidence_indices": [0],
}

ANSWERED_BOTH = {
    "answer": "PlannerDecision is a frozen dataclass [1] used by the routing planner [0].",
    "refused": False,
    "reason": None,
    "cited_evidence_indices": [0, 1],
}

REFUSED = {
    "answer": None,
    "refused": True,
    "reason": "The evidence doesn't cover how retries are handled.",
    "cited_evidence_indices": [],
}


@pytest.mark.parametrize(
    ("tool_input", "evidence"),
    [
        (ANSWERED, [RETRIEVER_EVIDENCE]),
        (ANSWERED_BOTH, [RETRIEVER_EVIDENCE, CODE_NAV_EVIDENCE]),
        (REFUSED, [RETRIEVER_EVIDENCE]),
    ],
)
def test_synthesize_parses_consistent_decisions(
    monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any], evidence: list[EvidenceItem]
) -> None:
    _install_fake_anthropic(monkeypatch, tool_input)
    client = AnthropicCriticClient(api_key="unused")

    decision = client.synthesize("some question", evidence)

    assert decision == CriticDecision(
        answer=tool_input["answer"],
        refused=tool_input["refused"],
        reason=tool_input["reason"],
        cited_evidence_indices=tool_input["cited_evidence_indices"],
    )


def test_synthesize_refuses_on_empty_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch, REFUSED)
    client = AnthropicCriticClient(api_key="unused")

    decision = client.synthesize("some question", [])

    assert decision.refused is True
    assert decision.answer is None
    assert decision.cited_evidence_indices == []


def test_synthesize_forces_synthesize_answer_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, ANSWERED)
    client = AnthropicCriticClient(api_key="unused", model="claude-sonnet-5")

    client.synthesize("some question", [RETRIEVER_EVIDENCE])

    kwargs = fake_client.messages.create_kwargs
    assert kwargs is not None
    assert kwargs["tool_choice"] == {"type": "tool", "name": "synthesize_answer"}
    assert kwargs["tools"][0]["name"] == "synthesize_answer"
    assert kwargs["model"] == "claude-sonnet-5"


def test_synthesize_defaults_to_sonnet_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, ANSWERED)
    client = AnthropicCriticClient(api_key="unused")

    client.synthesize("some question", [RETRIEVER_EVIDENCE])

    assert fake_client.messages.create_kwargs["model"] == "claude-sonnet-5"


def test_synthesize_message_says_no_evidence_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, REFUSED)
    client = AnthropicCriticClient(api_key="unused")

    client.synthesize("some question", [])

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "Evidence: none." in message
    assert "some question" in message


def test_synthesize_message_includes_indexed_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, ANSWERED_BOTH)
    client = AnthropicCriticClient(api_key="unused")

    client.synthesize("some question", [RETRIEVER_EVIDENCE, CODE_NAV_EVIDENCE])

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "[0] source=retriever" in message
    assert "rag_core/jobs/store.py:10-20" in message
    assert "score=0.91" in message
    assert "[1] source=code_navigation" in message
    assert "symbol=PlannerDecision (class, definition)" in message


@pytest.mark.parametrize(
    "tool_input",
    [
        {
            "answer": "an answer",
            "refused": True,
            "reason": "a reason",
            "cited_evidence_indices": [],
        },
        {
            "answer": None,
            "refused": False,
            "reason": None,
            "cited_evidence_indices": [0],
        },
        {
            "answer": "an answer",
            "refused": False,
            "reason": "unexpected reason",
            "cited_evidence_indices": [0],
        },
        {
            "answer": None,
            "refused": True,
            "reason": None,
            "cited_evidence_indices": [],
        },
        {
            "answer": None,
            "refused": True,
            "reason": "a reason",
            "cited_evidence_indices": [0],
        },
        {
            "answer": "an answer",
            "refused": False,
            "reason": None,
            "cited_evidence_indices": [],
        },
    ],
)
def test_synthesize_raises_on_inconsistent_decision(monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any]) -> None:
    _install_fake_anthropic(monkeypatch, tool_input)
    client = AnthropicCriticClient(api_key="unused")

    with pytest.raises(CriticValidationError):
        client.synthesize("some question", [RETRIEVER_EVIDENCE])


def test_synthesize_raises_on_out_of_range_index(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_input = {
        "answer": "an answer citing something that doesn't exist",
        "refused": False,
        "reason": None,
        "cited_evidence_indices": [5],
    }
    _install_fake_anthropic(monkeypatch, tool_input)
    client = AnthropicCriticClient(api_key="unused")

    with pytest.raises(CriticValidationError, match="5"):
        client.synthesize("some question", [RETRIEVER_EVIDENCE])


def test_fake_critic_client_returns_fixed_decision(fake_critic_client) -> None:
    decision = CriticDecision(
        answer="fixed answer",
        refused=False,
        reason=None,
        cited_evidence_indices=[0],
    )
    fake_critic_client._decision = decision

    result = fake_critic_client.synthesize("anything", [RETRIEVER_EVIDENCE])

    assert result == decision
    assert fake_critic_client.calls == [("anything", [RETRIEVER_EVIDENCE])]


def test_fake_critic_client_supports_callable_decision(fake_critic_client) -> None:
    def decide(question: str, evidence: list[EvidenceItem]) -> CriticDecision:
        if evidence:
            return CriticDecision(
                answer="answered from evidence",
                refused=False,
                reason=None,
                cited_evidence_indices=[0],
            )
        return CriticDecision(
            answer=None,
            refused=True,
            reason="no evidence",
            cited_evidence_indices=[],
        )

    fake_critic_client._decision = decide

    empty = fake_critic_client.synthesize("what does X do", [])
    assert empty.refused is True

    with_evidence = fake_critic_client.synthesize("what does X do", [RETRIEVER_EVIDENCE])
    assert with_evidence.refused is False
