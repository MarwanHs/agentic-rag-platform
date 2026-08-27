from __future__ import annotations

from agent_orchestrator.critic import CriticDecision
from agent_orchestrator.planner import PlannerDecision
from agent_orchestrator.query import QueryCitation, answer_query
from shared.evidence import EvidenceItem

RETRIEVER_EVIDENCE = [
    EvidenceItem(
        source="retriever",
        file_path="rag_core/jobs/store.py",
        line_range=(10, 20),
        content="def claim_next_job(...): ...",
        score=0.91,
    )
]

CODE_NAV_EVIDENCE = [
    EvidenceItem(
        source="code_navigation",
        file_path="agent_orchestrator/planner.py",
        line_range=(119, 125),
        content="class PlannerDecision: ...",
        symbol_name="PlannerDecision",
        symbol_kind="class",
        reference_kind="definition",
    )
]


class _CountingStub:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self._evidence = evidence
        self.calls = 0

    def __call__(self, *args, **kwargs) -> list[EvidenceItem]:
        self.calls += 1
        return self._evidence


def _planner_decision(agents_needed: list[str], **overrides) -> PlannerDecision:
    defaults = dict(
        reasoning="test",
        agents_needed=agents_needed,
        retriever_query="some query" if "retriever" in agents_needed else None,
        code_navigation_symbols=["Foo"] if "code_navigation" in agents_needed else [],
        existing_context_sufficient=False,
    )
    defaults.update(overrides)
    return PlannerDecision(**defaults)


def _answered_critic_decision(cited_evidence_indices: list[int]) -> CriticDecision:
    return CriticDecision(
        answer="an answer",
        refused=False,
        reason=None,
        cited_evidence_indices=cited_evidence_indices,
    )


def test_only_retriever_requested_skips_code_navigation(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision(["retriever"])
    fake_critic_client._decision = _answered_critic_decision([0])
    retriever_stub = _CountingStub(RETRIEVER_EVIDENCE)
    code_nav_stub = _CountingStub([])

    result = answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=retriever_stub,
        gather_code_navigation_evidence_fn=code_nav_stub,
    )

    assert retriever_stub.calls == 1
    assert code_nav_stub.calls == 0
    assert result.sources_used == ["retriever"]


def test_only_code_navigation_requested_skips_retriever(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision(["code_navigation"])
    fake_critic_client._decision = _answered_critic_decision([0])
    retriever_stub = _CountingStub([])
    code_nav_stub = _CountingStub(CODE_NAV_EVIDENCE)

    result = answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=retriever_stub,
        gather_code_navigation_evidence_fn=code_nav_stub,
    )

    assert retriever_stub.calls == 0
    assert code_nav_stub.calls == 1
    assert result.sources_used == ["code_navigation"]


def test_both_agents_requested_evidence_concatenated_in_order(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision(["retriever", "code_navigation"])
    fake_critic_client._decision = _answered_critic_decision([0, 1])
    retriever_stub = _CountingStub(RETRIEVER_EVIDENCE)
    code_nav_stub = _CountingStub(CODE_NAV_EVIDENCE)

    answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=retriever_stub,
        gather_code_navigation_evidence_fn=code_nav_stub,
    )

    assert retriever_stub.calls == 1
    assert code_nav_stub.calls == 1
    critic_call_evidence = fake_critic_client.calls[0][1]
    assert critic_call_evidence == RETRIEVER_EVIDENCE + CODE_NAV_EVIDENCE


def test_empty_agents_needed_calls_neither_stub_but_critic_still_runs(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision([])
    fake_critic_client._decision = CriticDecision(
        answer=None, refused=True, reason="no evidence", cited_evidence_indices=[]
    )
    retriever_stub = _CountingStub([])
    code_nav_stub = _CountingStub([])

    result = answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=retriever_stub,
        gather_code_navigation_evidence_fn=code_nav_stub,
    )

    assert retriever_stub.calls == 0
    assert code_nav_stub.calls == 0
    assert len(fake_critic_client.calls) == 1
    assert fake_critic_client.calls[0][1] == []
    assert result.refused is True
    assert result.sources_used == []


def test_citation_indices_map_correctly_to_query_citations(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision(["retriever", "code_navigation"])
    fake_critic_client._decision = _answered_critic_decision([1])

    result = answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=_CountingStub(RETRIEVER_EVIDENCE),
        gather_code_navigation_evidence_fn=_CountingStub(CODE_NAV_EVIDENCE),
    )

    assert result.citations == [
        QueryCitation(
            source="code_navigation",
            file_path="agent_orchestrator/planner.py",
            line_range=(119, 125),
        )
    ]


def test_sources_used_equals_planner_agents_needed(fake_planner_client, fake_critic_client) -> None:
    fake_planner_client._decision = _planner_decision(["retriever", "code_navigation"])
    fake_critic_client._decision = _answered_critic_decision([0, 1])

    result = answer_query(
        "question",
        "codebase-1",
        conn=None,
        qdrant_client=None,
        embedding_client=None,
        rerank_client=None,
        planner_client=fake_planner_client,
        reformulation_client=None,
        critic_client=fake_critic_client,
        gather_retriever_evidence_fn=_CountingStub(RETRIEVER_EVIDENCE),
        gather_code_navigation_evidence_fn=_CountingStub(CODE_NAV_EVIDENCE),
    )

    assert result.sources_used == ["retriever", "code_navigation"]
