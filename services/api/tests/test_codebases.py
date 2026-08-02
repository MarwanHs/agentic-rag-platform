from __future__ import annotations

from rag_core.jobs.store import create_job, update_status


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


def test_query_returns_501_when_ready(client, pg_conn) -> None:
    codebase_id = _make_ready_job(pg_conn)
    try:
        resp = client.post(f"/codebases/{codebase_id}/query", json={"question": "what does this do"})
        assert resp.status_code == 501
    finally:
        _cleanup(pg_conn, codebase_id)
