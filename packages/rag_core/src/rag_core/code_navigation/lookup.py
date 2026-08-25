"""First-pass code-navigation lookups: find definition / find callers (decision #21)."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True, slots=True)
class SymbolLocation:
    kind: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: str | None
    source_text: str | None


@dataclass(frozen=True, slots=True)
class ReferenceLocation:
    kind: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    enclosing_qualified_name: str | None
    source_text: str | None


def find_definition(conn: psycopg.Connection, codebase_id: str, name: str) -> list[SymbolLocation]:
    with conn.cursor() as cur:
        if "." in name:
            cur.execute(
                """
                SELECT kind, name, qualified_name, file_path, start_line, end_line, docstring, source_text
                FROM symbols
                WHERE codebase_id = %s AND qualified_name LIKE %s
                ORDER BY qualified_name
                """,
                (codebase_id, f"%::{name}"),
            )
        else:
            cur.execute(
                """
                SELECT kind, name, qualified_name, file_path, start_line, end_line, docstring, source_text
                FROM symbols
                WHERE codebase_id = %s AND name = %s
                ORDER BY qualified_name
                """,
                (codebase_id, name),
            )
        return [SymbolLocation(*row) for row in cur.fetchall()]


def find_references(conn: psycopg.Connection, codebase_id: str, name: str) -> list[ReferenceLocation]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind, name, file_path, start_line, end_line, enclosing_qualified_name, source_text
            FROM symbol_references
            WHERE codebase_id = %s AND name = %s
            ORDER BY file_path, start_line
            """,
            (codebase_id, name),
        )
        return [ReferenceLocation(*row) for row in cur.fetchall()]
