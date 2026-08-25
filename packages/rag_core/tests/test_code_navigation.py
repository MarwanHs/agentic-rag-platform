from __future__ import annotations

import uuid

from rag_core.code_navigation.indexer import index_file
from rag_core.code_navigation.lookup import find_definition, find_references
from rag_core.code_navigation.schema import ensure_schema
from rag_core.parsing.python_parser import parse_source

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


def test_find_definition_and_references(pg_conn) -> None:
    ensure_schema(pg_conn)
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    parsed = parse_source(SOURCE, "demo.py")

    try:
        index_file(pg_conn, codebase_id, parsed)

        definitions = find_definition(pg_conn, codebase_id, "add")
        assert len(definitions) == 1
        assert definitions[0].kind == "function"
        assert definitions[0].qualified_name == "demo.py::add"
        assert definitions[0].source_text == (
            'def add(a, b):\n    """Add two numbers."""\n    return a + b'
        )

        references = find_references(pg_conn, codebase_id, "add")
        assert len(references) == 1
        assert references[0].enclosing_qualified_name == "demo.py::total"
        assert references[0].source_text == "add(result, v)"

        qualified_definitions = find_definition(pg_conn, codebase_id, "total.add")
        assert qualified_definitions == []  # "add" isn't nested under "total"; sanity-check dotted lookup path

        dotted_lookup = find_definition(pg_conn, codebase_id, "demo.add")
        assert dotted_lookup == []
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM symbol_references WHERE codebase_id = %s", (codebase_id,))
            cur.execute("DELETE FROM symbols WHERE codebase_id = %s", (codebase_id,))
        pg_conn.commit()
