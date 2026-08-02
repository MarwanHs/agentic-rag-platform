from __future__ import annotations

from rag_core.jobs.schema import ensure_schema
from rag_core.jobs.store import create_job, get_job, list_ready_codebases, update_status


def test_job_lifecycle(pg_conn) -> None:
    ensure_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/repo")

    try:
        assert job.status == "queued"

        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "queued"
        assert fetched.current_batch is None
        assert fetched.failure_reason is None

        update_status(pg_conn, job.id, "embedding", current_batch="3/10")
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "embedding"
        assert fetched.current_batch == "3/10"

        assert job.id not in {c.id for c in list_ready_codebases(pg_conn)}

        update_status(pg_conn, job.id, "ready")
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "ready"
        assert job.id in {c.id for c in list_ready_codebases(pg_conn)}
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s", (job.id,))
        pg_conn.commit()


def test_job_failure(pg_conn) -> None:
    ensure_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/repo")

    try:
        update_status(pg_conn, job.id, "failed", failure_reason="clone failed: repository not found")
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.failure_reason == "clone failed: repository not found"
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s", (job.id,))
        pg_conn.commit()


def test_get_unknown_job_returns_none(pg_conn) -> None:
    assert get_job(pg_conn, "does-not-exist") is None
