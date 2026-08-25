from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_orchestrator import retriever_agent as retriever_agent_module
from agent_orchestrator.retriever_agent import (
    AnthropicReformulationClient,
    ReformulationDecision,
    _passes_confidence_gate,
    gather_retriever_evidence,
)
from rag_core.parsing.models import Chunk, ChunkKind
from rag_core.retrieval.qdrant_index import SearchResult, collection_name_for_codebase, ensure_collection, upsert_chunks


# --- _passes_confidence_gate: pure unit tests, no network/fixtures -------


def test_confidence_gate_passes_when_top_score_at_or_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_agent_module, "RERANK_CONFIDENCE_THRESHOLD", 0.5)

    assert _passes_confidence_gate([SearchResult(payload={}, score=0.5)]) is True
    assert _passes_confidence_gate([SearchResult(payload={}, score=0.9)]) is True


def test_confidence_gate_fails_when_top_score_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_agent_module, "RERANK_CONFIDENCE_THRESHOLD", 0.5)

    assert _passes_confidence_gate([SearchResult(payload={}, score=0.49)]) is False


def test_confidence_gate_only_looks_at_top_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_agent_module, "RERANK_CONFIDENCE_THRESHOLD", 0.5)

    results = [
        SearchResult(payload={}, score=0.9),
        SearchResult(payload={}, score=0.01),
        SearchResult(payload={}, score=0.01),
    ]

    assert _passes_confidence_gate(results) is True


def test_confidence_gate_fails_on_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever_agent_module, "RERANK_CONFIDENCE_THRESHOLD", 0.5)

    assert _passes_confidence_gate([]) is False


# --- gather_retriever_evidence: integration tests against real Qdrant ----


def _make_chunk(name: str, text: str, start_line: int = 1, end_line: int = 5) -> Chunk:
    return Chunk(
        file_path="sample.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        qualified_name=f"sample.py::{name}",
        start_line=start_line,
        end_line=end_line,
        text=text,
        docstring=None,
    )


def test_gather_retriever_evidence_returns_first_pass_when_gate_passes(
    qdrant_client, fake_embedding_client, fake_rerank_client, fake_reformulation_client
) -> None:
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    collection_name = collection_name_for_codebase(codebase_id)
    ensure_collection(qdrant_client, collection_name)

    chunks = [_make_chunk("parse_config", 'def parse_config(path):\n    """Parse a YAML config file into a dict."""\n    ...')]

    try:
        upsert_chunks(qdrant_client, codebase_id, chunks, fake_embedding_client)

        evidence = gather_retriever_evidence(
            qdrant_client,
            codebase_id,
            "parse config",
            fake_embedding_client,
            fake_rerank_client,
            fake_reformulation_client,
        )

        assert fake_reformulation_client.calls == []
        assert len(evidence) == 1
        assert evidence[0].source == "retriever"
        assert evidence[0].file_path == "sample.py"
        assert evidence[0].line_range == (1, 5)
        assert "parse_config" in evidence[0].content
        assert evidence[0].score == 1.0
    finally:
        qdrant_client.delete_collection(collection_name)


def test_gather_retriever_evidence_reformulates_when_gate_fails(
    qdrant_client, fake_embedding_client, fake_rerank_client, fake_reformulation_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FakeRerankClient always scores its top result 1.0, so bump the
    # threshold above that to force the gate to fail regardless of content.
    monkeypatch.setattr(retriever_agent_module, "RERANK_CONFIDENCE_THRESHOLD", 1.1)

    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    collection_name = collection_name_for_codebase(codebase_id)
    ensure_collection(qdrant_client, collection_name)

    chunks = [
        _make_chunk("parse_config", 'def parse_config(path):\n    """Parse a YAML config file."""\n    ...', 1, 5),
        _make_chunk("send_email", 'def send_email(to, subject, body):\n    """Send an email via SMTP."""\n    ...', 7, 11),
    ]

    fake_reformulation_client._decision = ReformulationDecision(
        reasoning="first pass was weak, try lexical terms closer to the target",
        reformulated_query="send email smtp",
    )

    try:
        upsert_chunks(qdrant_client, codebase_id, chunks, fake_embedding_client)

        evidence = gather_retriever_evidence(
            qdrant_client,
            codebase_id,
            "parse config",
            fake_embedding_client,
            fake_rerank_client,
            fake_reformulation_client,
        )

        assert len(fake_reformulation_client.calls) == 1
        first_pass_query, first_pass_results = fake_reformulation_client.calls[0]
        assert first_pass_query == "parse config"
        assert len(first_pass_results) >= 1

        # Final evidence reflects the *reformulated* query's search, not the
        # first pass -- and is returned even though its top score (1.0, still
        # below the 1.1 threshold) would fail the gate if re-checked.
        assert evidence[0].score == 1.0
        assert "send_email" in evidence[0].content
    finally:
        qdrant_client.delete_collection(collection_name)


def test_gather_retriever_evidence_treats_empty_first_pass_as_gate_failure(
    qdrant_client, fake_embedding_client, fake_rerank_client, fake_reformulation_client
) -> None:
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    collection_name = collection_name_for_codebase(codebase_id)
    ensure_collection(qdrant_client, collection_name)

    try:
        evidence = gather_retriever_evidence(
            qdrant_client,
            codebase_id,
            "anything at all",
            fake_embedding_client,
            fake_rerank_client,
            fake_reformulation_client,
        )

        assert len(fake_reformulation_client.calls) == 1
        assert fake_reformulation_client.calls[0][1] == []
        assert evidence == []
    finally:
        qdrant_client.delete_collection(collection_name)


# --- AnthropicReformulationClient: mocked anthropic.Anthropic boundary ---


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

    monkeypatch.setattr(retriever_agent_module, "anthropic", FakeAnthropicModule)
    return fake_client


REFORMULATION_TOOL_INPUT = {
    "reasoning": "First pass used vague terminology; narrowed to match likely docstring phrasing.",
    "reformulated_query": "send email via SMTP",
}


def test_reformulate_parses_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch, REFORMULATION_TOOL_INPUT)
    client = AnthropicReformulationClient(api_key="unused")

    decision = client.reformulate("send an email", [])

    assert decision == ReformulationDecision(
        reasoning=REFORMULATION_TOOL_INPUT["reasoning"],
        reformulated_query=REFORMULATION_TOOL_INPUT["reformulated_query"],
    )


def test_reformulate_forces_reformulate_query_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, REFORMULATION_TOOL_INPUT)
    client = AnthropicReformulationClient(api_key="unused", model="claude-haiku-4-5-20251001")

    client.reformulate("send an email", [])

    kwargs = fake_client.messages.create_kwargs
    assert kwargs is not None
    assert kwargs["tool_choice"] == {"type": "tool", "name": "reformulate_query"}
    assert kwargs["tools"][0]["name"] == "reformulate_query"
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


def test_reformulate_includes_query_and_results_in_message(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, REFORMULATION_TOOL_INPUT)
    client = AnthropicReformulationClient(api_key="unused")
    results = [
        SearchResult(
            payload={
                "file_path": "sample.py",
                "qualified_name": "sample.py::parse_config",
                "start_line": 1,
                "end_line": 5,
                "text": "def parse_config(path): ...",
            },
            score=0.31,
        )
    ]

    client.reformulate("parse config", results)

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "parse config" in message
    assert "sample.py:1-5" in message
    assert "sample.py::parse_config" in message


def test_reformulate_handles_empty_first_pass_results(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_anthropic(monkeypatch, REFORMULATION_TOOL_INPUT)
    client = AnthropicReformulationClient(api_key="unused")

    client.reformulate("obscure query", [])

    message = fake_client.messages.create_kwargs["messages"][0]["content"]
    assert "obscure query" in message
    assert "none" in message.lower()


# --- FakeReformulationClient ----------------------------------------------


def test_fake_reformulation_client_returns_fixed_decision(fake_reformulation_client) -> None:
    decision = ReformulationDecision(reasoning="fixed", reformulated_query="q2")
    fake_reformulation_client._decision = decision

    result = fake_reformulation_client.reformulate("q1", [])

    assert result == decision
    assert fake_reformulation_client.calls == [("q1", [])]


def test_fake_reformulation_client_supports_callable_decision(fake_reformulation_client) -> None:
    def decide(first_pass_query: str, first_pass_results: list[SearchResult]) -> ReformulationDecision:
        return ReformulationDecision(reasoning="dynamic", reformulated_query=f"{first_pass_query} refined")

    fake_reformulation_client._decision = decide

    result = fake_reformulation_client.reformulate("original", [])

    assert result.reformulated_query == "original refined"
