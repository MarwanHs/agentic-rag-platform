"""Job row lifecycle: create, read, list, dequeue, and advance status.

`update_status` and `update_pipeline_state` are deliberately separate: status
transitions (decision #5) and per-worker pipeline progress (decision #39) are
different concerns written at different times during a job's execution, and
neither should clobber the other's most recent value.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    url: str
    status: str
    pipeline_state: dict
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class CodebaseSummary:
    id: str
    url: str


def create_job(conn: psycopg.Connection, url: str) -> Job:
    job_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("INSERT INTO jobs (id, url, status) VALUES (%s, %s, 'queued')", (job_id, url))
    conn.commit()
    return Job(id=job_id, url=url, status="queued", pipeline_state={}, failure_reason=None)


def get_job(conn: psycopg.Connection, job_id: str) -> Job | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, url, status, pipeline_state, failure_reason FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Job(id=row[0], url=row[1], status=row[2], pipeline_state=row[3], failure_reason=row[4])


def update_status(
    conn: psycopg.Connection,
    job_id: str,
    status: str,
    failure_reason: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status = %s, failure_reason = %s, updated_at = now() WHERE id = %s",
            (status, failure_reason, job_id),
        )
    conn.commit()


def update_pipeline_state(conn: psycopg.Connection, job_id: str, pipeline_state: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET pipeline_state = %s, updated_at = now() WHERE id = %s",
            (Jsonb(pipeline_state), job_id),
        )
    conn.commit()


def dequeue_next_job(conn: psycopg.Connection) -> Job | None:
    """Claim the oldest queued job (decision #37).

    `FOR UPDATE SKIP LOCKED` lets multiple workers poll concurrently without
    blocking on or double-claiming the same row -- the locking clause is
    included from the start even though today's default is a single worker.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, url, status, pipeline_state, failure_reason
            FROM jobs
            WHERE status = 'queued'
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None

        cur.execute(
            "UPDATE jobs SET status = 'cloning', updated_at = now() WHERE id = %s",
            (row[0],),
        )
    conn.commit()
    return Job(id=row[0], url=row[1], status="cloning", pipeline_state=row[3], failure_reason=row[4])


def list_ready_codebases(conn: psycopg.Connection) -> list[CodebaseSummary]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, url FROM jobs WHERE status = 'ready' ORDER BY created_at DESC")
        return [CodebaseSummary(id=row[0], url=row[1]) for row in cur.fetchall()]
