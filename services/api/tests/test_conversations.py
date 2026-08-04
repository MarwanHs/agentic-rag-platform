from __future__ import annotations

from rag_core.conversations.store import create_conversation
from rag_core.jobs.store import create_job, update_status


def _make_ready_job(pg_conn) -> str:
    job = create_job(pg_conn, "https://github.com/example/repo")
    update_status(pg_conn, job.id, "ready")
    return job.id


def _cleanup_job(pg_conn, job_id: str) -> None:
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM conversations WHERE codebase_id = %s", (job_id,))
        cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
    pg_conn.commit()


def test_create_conversation_returns_404_for_unknown_codebase(client) -> None:
    resp = client.post("/codebases/does-not-exist/conversations")
    assert resp.status_code == 404


def test_create_conversation_returns_409_when_not_ready(client, pg_conn) -> None:
    job = create_job(pg_conn, "https://github.com/example/repo")
    try:
        resp = client.post(f"/codebases/{job.id}/conversations")
        assert resp.status_code == 409
    finally:
        _cleanup_job(pg_conn, job.id)


def test_create_conversation_succeeds_when_ready(client, pg_conn) -> None:
    codebase_id = _make_ready_job(pg_conn)
    try:
        resp = client.post(f"/codebases/{codebase_id}/conversations")
        assert resp.status_code == 201
        body = resp.json()
        assert body["conversation_id"]
    finally:
        _cleanup_job(pg_conn, codebase_id)


def test_send_message_returns_404_for_unknown_conversation(client) -> None:
    resp = client.post("/conversations/does-not-exist/messages", json={"message": "what else calls it?"})
    assert resp.status_code == 404


def test_send_message_returns_501_for_real_conversation(client, pg_conn) -> None:
    codebase_id = _make_ready_job(pg_conn)
    conversation = create_conversation(pg_conn, codebase_id)
    try:
        resp = client.post(
            f"/conversations/{conversation.id}/messages", json={"message": "what else calls it?"}
        )
        assert resp.status_code == 501
    finally:
        _cleanup_job(pg_conn, codebase_id)
