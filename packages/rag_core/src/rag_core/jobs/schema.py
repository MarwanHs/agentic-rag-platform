"""Job status table (decisions #3, #5, #21, #23).

A job's own id doubles as the codebase id once it reaches 'ready' -- there is
no separate codebases table, since a codebase is exactly "a job that
succeeded" (decision #6: readiness is all-or-nothing).
"""

from __future__ import annotations

import psycopg

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'cloning', 'parsing', 'embedding', 'indexing', 'ready', 'failed')
    ),
    pipeline_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);

-- Migration for tables created before decision #39: current_batch TEXT was
-- an unused stub, replaced by pipeline_state JSONB (per-worker execution
-- state -- batch progress, attempt counts -- not a cross-job query target).
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS pipeline_state JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE jobs DROP COLUMN IF EXISTS current_batch;
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
