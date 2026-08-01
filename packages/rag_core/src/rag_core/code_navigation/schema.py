"""Relational symbols/references schema (decisions #21, #23).

A single shared Postgres database, tenant-isolated by `codebase_id` column
(unlike the vector store, which isolates by collection per decision #14).
"""

from __future__ import annotations

import psycopg

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS symbols (
    id BIGSERIAL PRIMARY KEY,
    codebase_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('function', 'method', 'class', 'constant', 'import')),
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    docstring TEXT,
    source_module TEXT,
    UNIQUE (codebase_id, file_path, qualified_name)
);

CREATE INDEX IF NOT EXISTS symbols_codebase_name_idx ON symbols (codebase_id, name);
CREATE INDEX IF NOT EXISTS symbols_codebase_qualified_name_idx ON symbols (codebase_id, qualified_name);

CREATE TABLE IF NOT EXISTS symbol_references (
    id BIGSERIAL PRIMARY KEY,
    codebase_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('call', 'subclass')),
    name TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    enclosing_qualified_name TEXT,
    symbol_id BIGINT REFERENCES symbols (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS symbol_references_codebase_name_idx ON symbol_references (codebase_id, name);
CREATE INDEX IF NOT EXISTS symbol_references_codebase_symbol_idx ON symbol_references (codebase_id, symbol_id);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
