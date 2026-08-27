from __future__ import annotations

from agent_orchestrator.critic import CriticDecision
from agent_orchestrator.planner import PlannerDecision
from rag_core.code_navigation.indexer import index_file
from rag_core.code_navigation.schema import ensure_schema
from rag_core.jobs.store import create_job, update_status
from rag_core.parsing.python_parser import parse_source

SOURCE = '''"""Demo module."""


def add(a, b):
    """Add two numbers."""
    return a + b
'''


def _make_ready_job(pg_conn) -> str:
    job = create_job(pg_conn, "https://github.com/example/repo")
    update_status(pg_conn, job.id, "ready")
    return job.id


def _cleanup(pg_conn, job_id: str) -> None:
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
    pg_conn.commit()


def test_list_codebases_only_shows_ready(client, pg_conn) -> None:
    codebase_id = _make_ready_job(pg_conn)
    try:
        resp = client.get("/codebases")
        assert resp.status_code == 200
        ids = {c["id"] for c in resp.json()}
        assert codebase_id in ids
    finally:
        _cleanup(pg_conn, codebase_id)


def test_query_returns_404_for_unknown_codebase(client) -> None:
    resp = client.post("/codebases/does-not-exist/query", json={"question": "what does this do"})
    assert resp.status_code == 404


def test_query_returns_409_when_not_ready(client, pg_conn) -> None:
    job = create_job(pg_conn, "https://github.com/example/repo")
    try:
        resp = client.post(f"/codebases/{job.id}/query", json={"question": "what does this do"})
        assert resp.status_code == 409
    finally:
        _cleanup(pg_conn, job.id)


def test_query_returns_answered_response_via_code_navigation(
    client, pg_conn, fake_planner_client, fake_critic_client
) -> None:
    codebase_id = _make_ready_job(pg_conn)
    ensure_schema(pg_conn)
    parsed = parse_source(SOURCE, "demo.py")
    try:
        index_file(pg_conn, codebase_id, parsed)

        fake_planner_client._decision = PlannerDecision(
            reasoning="exact symbol lookup",
            agents_needed=["code_navigation"],
            retriever_query=None,
            code_navigation_symbols=["add"],
            existing_context_sufficient=False,
        )
        fake_critic_client._decision = CriticDecision(
            answer="add(a, b) returns a + b [0].",
            refused=False,
            reason=None,
            cited_evidence_indices=[0],
        )

        resp = client.post(f"/codebases/{codebase_id}/query", json={"question": "what does add do"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["refused"] is False
        assert body["answer"] == "add(a, b) returns a + b [0]."
        assert body["reason"] is None
        assert body["sources_used"] == ["code_navigation"]
        assert body["citations"] == [
            {
                "source": "code_navigation",
                "file_path": "demo.py",
                "line_range": {"start": 4, "end": 6},
            }
        ]
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM symbol_references WHERE codebase_id = %s", (codebase_id,))
            cur.execute("DELETE FROM symbols WHERE codebase_id = %s", (codebase_id,))
        pg_conn.commit()
        _cleanup(pg_conn, codebase_id)


def test_query_returns_refused_response_when_no_agents_needed(
    client, pg_conn, fake_planner_client, fake_critic_client
) -> None:
    codebase_id = _make_ready_job(pg_conn)
    try:
        fake_planner_client._decision = PlannerDecision(
            reasoning="off-topic question",
            agents_needed=[],
            retriever_query=None,
            code_navigation_symbols=[],
            existing_context_sufficient=False,
        )
        fake_critic_client._decision = CriticDecision(
            answer=None,
            refused=True,
            reason="No evidence was gathered for this question.",
            cited_evidence_indices=[],
        )

        resp = client.post(f"/codebases/{codebase_id}/query", json={"question": "what's the weather today"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["refused"] is True
        assert body["answer"] is None
        assert body["reason"] == "No evidence was gathered for this question."
        assert body["citations"] == []
        assert body["sources_used"] == []
    finally:
        _cleanup(pg_conn, codebase_id)
