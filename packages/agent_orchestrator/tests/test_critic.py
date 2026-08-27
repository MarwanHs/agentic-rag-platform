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
    id: str = "toolu_fake"


@dataclass
class FakeMessage:
    content: list[FakeToolUseBlock] = field(default_factory=list)
    stop_reason: str = "tool_use"


class FakeMessages:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._responses = responses
        self.create_kwargs_history: list[dict[str, Any]] = []

    @property
    def create_kwargs(self) -> dict[str, Any] | None:
        # Most tests only trigger one call; keep this the *last* call's
        # kwargs so single-call assertions read naturally even though a
        # validation failure can now trigger a second (retry) call.
        return self.create_kwargs_history[-1] if self.create_kwargs_history else None

    def create(self, **kwargs: Any) -> FakeMessage:
        self.create_kwargs_history.append(kwargs)
        index = min(len(self.create_kwargs_history) - 1, len(self._responses) - 1)
        return self._responses[index]


class FakeAnthropicClient:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self.messages = FakeMessages(responses)


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch, tool_input: dict[str, Any], stop_reason: str = "tool_use"
) -> FakeAnthropicClient:
    response = FakeMessage(content=[FakeToolUseBlock(input=tool_input)], stop_reason=stop_reason)
    return _install_fake_anthropic_sequence(monkeypatch, [response])


def _install_fake_anthropic_sequence(
    monkeypatch: pytest.MonkeyPatch, responses: list[FakeMessage]
) -> FakeAnthropicClient:
    fake_client = FakeAnthropicClient(responses)

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


@pytest.mark.parametrize("missing_field", ["answer", "refused", "reason", "cited_evidence_indices"])
def test_synthesize_raises_validation_error_on_missing_field(
    monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    # Regression test: a live query produced a tool_use response missing a
    # required field (observed for cited_evidence_indices and, on a separate
    # occasion, for the planner's equivalent field) -- reproduced even
    # against the unmodified original prompt, so this is a model-level
    # reliability quirk, not something fixable via prompt wording. Must
    # raise CriticValidationError, not a raw KeyError.
    incomplete_input = {k: v for k, v in REFUSED.items() if k != missing_field}
    _install_fake_anthropic(monkeypatch, incomplete_input)
    client = AnthropicCriticClient(api_key="unused")

    with pytest.raises(CriticValidationError, match=missing_field):
        client.synthesize("some question", [RETRIEVER_EVIDENCE])


def test_synthesize_raises_validation_error_on_max_tokens_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truncated_input = {
        "answer": "partial answer text cut off mid-generation",
        "refused": False,
        "reason": None,
        # cited_evidence_indices genuinely absent, mirroring observed truncation
    }
    _install_fake_anthropic(monkeypatch, truncated_input, stop_reason="max_tokens")
    client = AnthropicCriticClient(api_key="unused")

    with pytest.raises(CriticValidationError, match="max_tokens"):
        client.synthesize("some question", [RETRIEVER_EVIDENCE])


def test_synthesize_retries_once_after_validation_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = FakeMessage(
        content=[
            FakeToolUseBlock(
                input={"answer": "partial", "refused": False, "reason": None}
                # cited_evidence_indices missing
            )
        ]
    )
    fake_client = _install_fake_anthropic_sequence(
        monkeypatch, [malformed, FakeMessage(content=[FakeToolUseBlock(input=ANSWERED)])]
    )
    client = AnthropicCriticClient(api_key="unused")

    decision = client.synthesize("some question", [RETRIEVER_EVIDENCE])

    assert decision == CriticDecision(
        answer=ANSWERED["answer"],
        refused=ANSWERED["refused"],
        reason=ANSWERED["reason"],
        cited_evidence_indices=ANSWERED["cited_evidence_indices"],
    )
    assert len(fake_client.messages.create_kwargs_history) == 2

    retry_messages = fake_client.messages.create_kwargs_history[1]["messages"]
    assert retry_messages[0]["role"] == "user"
    assert retry_messages[1]["role"] == "assistant"
    assert retry_messages[1]["content"] == malformed.content

    # Retry feedback must be a proper tool_result block referencing the
    # original tool_use's id, immediately after the assistant's tool_use turn
    # -- a plain text user turn there is rejected outright by the Anthropic
    # API (400: "tool_use ids were found without tool_result blocks").
    retry_feedback_turn = retry_messages[2]
    assert retry_feedback_turn["role"] == "user"
    tool_result = retry_feedback_turn["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == malformed.content[0].id
    assert tool_result["is_error"] is True
    assert "invalid" in tool_result["content"]


def test_synthesize_raises_after_retry_also_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    first_malformed = {"answer": "a", "refused": False, "reason": None}  # missing cited_evidence_indices
    second_malformed = {"refused": False, "reason": None, "cited_evidence_indices": [0]}  # missing answer
    fake_client = _install_fake_anthropic_sequence(
        monkeypatch,
        [
            FakeMessage(content=[FakeToolUseBlock(input=first_malformed)]),
            FakeMessage(content=[FakeToolUseBlock(input=second_malformed)]),
        ],
    )
    client = AnthropicCriticClient(api_key="unused")

    # Matches the *missing field name* specifically -- the first attempt's
    # error text also mentions "answer" incidentally (it's a present field in
    # that tool_input), so this proves the retry's error propagated, not the
    # first attempt's.
    with pytest.raises(CriticValidationError, match=r"missing required field\(s\) \['answer'\]"):
        client.synthesize("some question", [RETRIEVER_EVIDENCE])

    assert len(fake_client.messages.create_kwargs_history) == 2


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
