from __future__ import annotations

import logging
import uuid

from rag_core.code_navigation.indexer import index_file
from rag_core.code_navigation.schema import ensure_schema
from rag_core.parsing.python_parser import parse_source

from agent_orchestrator import code_navigation_agent
from agent_orchestrator.code_navigation_agent import gather_code_navigation_evidence

SOURCE = '''"""Demo module."""


def add(a, b):
    """Add two numbers."""
    return a + b


def total(values):
    """Sum a list using add."""
    result = 0
    for v in values:
        result = add(result, v)
    return result
'''

AMBIGUOUS_SOURCE = '''class Foo:
    def validate(self):
        return True


class Bar:
    def validate(self):
        return False
'''


def _cleanup(pg_conn, codebase_id: str) -> None:
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM symbol_references WHERE codebase_id = %s", (codebase_id,))
        cur.execute("DELETE FROM symbols WHERE codebase_id = %s", (codebase_id,))
    pg_conn.commit()


def test_definition_and_reference_produce_two_evidence_items(pg_conn) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    parsed = parse_source(SOURCE, "demo.py")

    try:
        index_file(pg_conn, codebase_id, parsed)

        evidence = gather_code_navigation_evidence(pg_conn, codebase_id, ["add"])

        assert len(evidence) == 2
        definition = next(e for e in evidence if e.reference_kind == "definition")
        reference = next(e for e in evidence if e.reference_kind == "call")

        assert definition.source == "code_navigation"
        assert definition.file_path == "demo.py"
        assert definition.line_range == (4, 6)
        assert definition.content == 'def add(a, b):\n    """Add two numbers."""\n    return a + b'
        assert definition.symbol_name == "add"
        assert definition.symbol_kind == "function"

        assert reference.source == "code_navigation"
        assert reference.file_path == "demo.py"
        assert reference.content == "add(result, v)"
        assert reference.symbol_name == "add"
        assert reference.symbol_kind is None
    finally:
        _cleanup(pg_conn, codebase_id)


def test_no_matches_returns_empty_and_logs(pg_conn, caplog) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"

    try:
        with caplog.at_level(logging.INFO, logger="agent_orchestrator.code_navigation_agent"):
            evidence = gather_code_navigation_evidence(pg_conn, codebase_id, ["totally_unknown_symbol"])

        assert evidence == []
        assert any(
            "no matches" in record.message and "totally_unknown_symbol" in record.message
            for record in caplog.records
        )
    finally:
        _cleanup(pg_conn, codebase_id)


def test_ambiguous_matches_are_not_collapsed(pg_conn) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    parsed = parse_source(AMBIGUOUS_SOURCE, "ambiguous.py")

    try:
        index_file(pg_conn, codebase_id, parsed)

        evidence = gather_code_navigation_evidence(pg_conn, codebase_id, ["validate"])

        definitions = [e for e in evidence if e.reference_kind == "definition"]
        assert len(definitions) == 2
        assert {e.line_range for e in definitions} == {(2, 3), (7, 8)}
    finally:
        _cleanup(pg_conn, codebase_id)


def test_duplicate_symbol_names_deduplicated(pg_conn, monkeypatch) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    parsed = parse_source(SOURCE, "demo.py")

    try:
        index_file(pg_conn, codebase_id, parsed)

        call_counts = {"find_definition": 0, "find_references": 0}
        original_find_definition = code_navigation_agent.find_definition
        original_find_references = code_navigation_agent.find_references

        def counting_find_definition(*args, **kwargs):
            call_counts["find_definition"] += 1
            return original_find_definition(*args, **kwargs)

        def counting_find_references(*args, **kwargs):
            call_counts["find_references"] += 1
            return original_find_references(*args, **kwargs)

        monkeypatch.setattr(code_navigation_agent, "find_definition", counting_find_definition)
        monkeypatch.setattr(code_navigation_agent, "find_references", counting_find_references)

        evidence = gather_code_navigation_evidence(pg_conn, codebase_id, ["add", "add", "add"])

        assert call_counts["find_definition"] == 1
        assert call_counts["find_references"] == 1
        assert len(evidence) == 2  # one definition + one reference, not tripled
    finally:
        _cleanup(pg_conn, codebase_id)


def test_null_source_text_falls_back_to_docstring(pg_conn, caplog) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"

    try:
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO symbols (
                    codebase_id, file_path, kind, name, qualified_name,
                    start_line, end_line, start_byte, end_byte, source_text, docstring
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
                """,
                (
                    codebase_id,
                    "legacy.py",
                    "function",
                    "legacy_fn",
                    "legacy.py::legacy_fn",
                    1,
                    3,
                    0,
                    40,
                    "Legacy docstring from a pre-decision-#43 ingest.",
                ),
            )
        pg_conn.commit()

        with caplog.at_level(logging.WARNING, logger="agent_orchestrator.code_navigation_agent"):
            evidence = gather_code_navigation_evidence(pg_conn, codebase_id, ["legacy_fn"])

        assert len(evidence) == 1
        assert evidence[0].content == "Legacy docstring from a pre-decision-#43 ingest."
        assert any(
            "source_text missing" in record.message and "legacy_fn" in record.message
            for record in caplog.records
        )
    finally:
        _cleanup(pg_conn, codebase_id)
