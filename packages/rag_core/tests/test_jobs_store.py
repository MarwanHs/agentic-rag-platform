from __future__ import annotations

from rag_core.jobs.schema import ensure_schema
from rag_core.jobs.store import (
    create_job,
    dequeue_next_job,
    get_job,
    list_ready_codebases,
    update_pipeline_state,
    update_status,
)


def test_job_lifecycle(pg_conn) -> None:
    ensure_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/repo")

    try:
        assert job.status == "queued"

        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "queued"
        assert fetched.pipeline_state == {}
        assert fetched.failure_reason is None

        update_status(pg_conn, job.id, "embedding")
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "embedding"

        update_pipeline_state(pg_conn, job.id, {"embed": {"batches_done": 3, "batches_total": 10}})
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.pipeline_state == {"embed": {"batches_done": 3, "batches_total": 10}}
        assert fetched.status == "embedding"  # update_pipeline_state must not touch status

        assert job.id not in {c.id for c in list_ready_codebases(pg_conn)}

        update_status(pg_conn, job.id, "ready")
        fetched = get_job(pg_conn, job.id)
        assert fetched is not None
        assert fetched.status == "ready"
        assert fetched.pipeline_state == {"embed": {"batches_done": 3, "batches_total": 10}}  # untouched
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


def test_dequeue_next_job_claims_oldest_queued_job(pg_conn) -> None:
    ensure_schema(pg_conn)
    older = create_job(pg_conn, "https://github.com/example/older")
    newer = create_job(pg_conn, "https://github.com/example/newer")
    # force a distinguishable created_at ordering -- both jobs are inserted
    # back-to-back and could otherwise land in the same timestamp bucket
    with pg_conn.cursor() as cur:
        cur.execute("UPDATE jobs SET created_at = now() - interval '1 minute' WHERE id = %s", (older.id,))
    pg_conn.commit()

    try:
        claimed = dequeue_next_job(pg_conn)
        assert claimed is not None
        assert claimed.id == older.id
        assert claimed.status == "cloning"

        fetched = get_job(pg_conn, older.id)
        assert fetched is not None
        assert fetched.status == "cloning"

        # the newer job is still queued and untouched
        fetched_newer = get_job(pg_conn, newer.id)
        assert fetched_newer is not None
        assert fetched_newer.status == "queued"
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id IN (%s, %s)", (older.id, newer.id))
        pg_conn.commit()


def test_dequeue_next_job_skips_non_queued_jobs(pg_conn) -> None:
    ensure_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/repo")
    update_status(pg_conn, job.id, "ready")

    try:
        claimed = dequeue_next_job(pg_conn)
        assert claimed is None or claimed.id != job.id
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s", (job.id,))
        pg_conn.commit()


def test_dequeue_next_job_returns_none_when_empty(pg_conn) -> None:
    ensure_schema(pg_conn)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT id FROM jobs WHERE status = 'queued'")
        leftover_ids = [row[0] for row in cur.fetchall()]
    if leftover_ids:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE status = 'queued'")
        pg_conn.commit()

    assert dequeue_next_job(pg_conn) is None
