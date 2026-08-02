"""Job row lifecycle: create, read, list, and advance status.

Ingestion execution (clone/parse/embed/index) is not wired up yet -- that's
the job-queue milestone. `update_status` exists so status transitions are
correct and testable now, even though nothing calls it automatically yet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    url: str
    status: str
    current_batch: str | None
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
    return Job(id=job_id, url=url, status="queued", current_batch=None, failure_reason=None)


def get_job(conn: psycopg.Connection, job_id: str) -> Job | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, url, status, current_batch, failure_reason FROM jobs WHERE id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Job(id=row[0], url=row[1], status=row[2], current_batch=row[3], failure_reason=row[4])


def update_status(
    conn: psycopg.Connection,
    job_id: str,
    status: str,
    current_batch: str | None = None,
    failure_reason: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = %s, current_batch = %s, failure_reason = %s, updated_at = now()
            WHERE id = %s
            """,
            (status, current_batch, failure_reason, job_id),
        )
    conn.commit()


def list_ready_codebases(conn: psycopg.Connection) -> list[CodebaseSummary]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, url FROM jobs WHERE status = 'ready' ORDER BY created_at DESC")
        return [CodebaseSummary(id=row[0], url=row[1]) for row in cur.fetchall()]
