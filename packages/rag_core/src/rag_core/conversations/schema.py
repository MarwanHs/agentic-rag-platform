"""Conversation table for multi-turn follow-up (decisions #29, #30).

Turn-by-turn state (history, checkpoints) lives in LangGraph's Postgres
checkpointer once that's implemented -- this table only tracks which
conversations exist and which codebase each is scoped to. Must be created
after the jobs schema, since codebase_id references jobs(id).
"""

from __future__ import annotations

import psycopg

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    codebase_id TEXT NOT NULL REFERENCES jobs (id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_codebase_id_idx ON conversations (codebase_id);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
