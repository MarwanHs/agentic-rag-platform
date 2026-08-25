"""Populates the symbols/references schema during ingestion (decision #21).

Reference resolution is a first pass: a reference is linked to a symbol_id
only when its bare name matches exactly one symbol in the codebase. Ambiguous
(multiple candidates) or unresolved (zero candidates, e.g. a call into a
third-party import) references are kept with symbol_id NULL and remain
findable by name. There is no cross-file import-graph resolution yet.

Callers indexing a whole codebase should insert every file's symbols first,
then every file's references, so cross-file calls can resolve against the
complete symbol set.
"""

from __future__ import annotations

import psycopg

from rag_core.parsing.models import ParsedFile, Reference, Symbol


def delete_file(conn: psycopg.Connection, codebase_id: str, file_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM symbol_references WHERE codebase_id = %s AND file_path = %s",
            (codebase_id, file_path),
        )
        cur.execute(
            "DELETE FROM symbols WHERE codebase_id = %s AND file_path = %s",
            (codebase_id, file_path),
        )


def upsert_symbols(conn: psycopg.Connection, codebase_id: str, symbols: list[Symbol]) -> dict[str, int]:
    ids: dict[str, int] = {}
    with conn.cursor() as cur:
        for symbol in symbols:
            cur.execute(
                """
                INSERT INTO symbols (
                    codebase_id, file_path, kind, name, qualified_name,
                    start_line, end_line, start_byte, end_byte, source_text, docstring, source_module
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (codebase_id, file_path, qualified_name) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    start_line = EXCLUDED.start_line,
                    end_line = EXCLUDED.end_line,
                    start_byte = EXCLUDED.start_byte,
                    end_byte = EXCLUDED.end_byte,
                    source_text = EXCLUDED.source_text,
                    docstring = EXCLUDED.docstring,
                    source_module = EXCLUDED.source_module
                RETURNING id
                """,
                (
                    codebase_id,
                    symbol.file_path,
                    symbol.kind.value,
                    symbol.name,
                    symbol.qualified_name,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.start_byte,
                    symbol.end_byte,
                    symbol.source_text,
                    symbol.docstring,
                    symbol.source_module,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            ids[symbol.qualified_name] = row[0]
    return ids


def upsert_references(conn: psycopg.Connection, codebase_id: str, references: list[Reference]) -> None:
    with conn.cursor() as cur:
        for reference in references:
            symbol_id = _resolve_symbol_id(cur, codebase_id, reference.name)
            cur.execute(
                """
                INSERT INTO symbol_references (
                    codebase_id, file_path, kind, name, start_line, end_line,
                    start_byte, end_byte, source_text, enclosing_qualified_name, symbol_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    codebase_id,
                    reference.file_path,
                    reference.kind.value,
                    reference.name,
                    reference.start_line,
                    reference.end_line,
                    reference.start_byte,
                    reference.end_byte,
                    reference.source_text,
                    reference.enclosing_qualified_name,
                    symbol_id,
                ),
            )


def _resolve_symbol_id(cur: psycopg.Cursor, codebase_id: str, name: str) -> int | None:
    cur.execute("SELECT id FROM symbols WHERE codebase_id = %s AND name = %s", (codebase_id, name))
    rows = cur.fetchall()
    return rows[0][0] if len(rows) == 1 else None


def index_file(conn: psycopg.Connection, codebase_id: str, parsed: ParsedFile) -> None:
    delete_file(conn, codebase_id, parsed.file_path)
    upsert_symbols(conn, codebase_id, parsed.symbols)
    upsert_references(conn, codebase_id, parsed.references)
    conn.commit()
