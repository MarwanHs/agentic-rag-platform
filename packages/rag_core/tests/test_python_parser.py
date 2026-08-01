from __future__ import annotations

from pathlib import Path

from rag_core.parsing.models import ChunkKind, ReferenceKind, SymbolKind
from rag_core.parsing.python_parser import parse_source

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def test_sample_module_chunks() -> None:
    parsed = parse_source(_load("sample_module.py"), "sample_module.py")

    function_chunk_qnames = {c.qualified_name for c in parsed.chunks if c.kind == ChunkKind.FUNCTION}
    assert function_chunk_qnames == {
        "sample_module.py::double",
        "sample_module.py::Greeter.__init__",
        "sample_module.py::Greeter.greet",
        "sample_module.py::Greeter._format",
        "sample_module.py::LoudGreeter.greet",
    }

    module_chunk = next(c for c in parsed.chunks if c.kind == ChunkKind.MODULE)
    assert module_chunk.docstring == "Sample module for parser tests."
    assert "class Greeter" in module_chunk.text
    assert "class LoudGreeter" in module_chunk.text
    assert "def greet" not in module_chunk.text  # method bodies live in their own chunk, not here

    double_chunk = next(c for c in parsed.chunks if c.name == "double")
    assert double_chunk.docstring == "Return twice the given value."
    assert "# Computes the double of a number." in double_chunk.text  # leading comment attached
    assert "def helper(x):" in double_chunk.text  # nested function folded into the parent chunk


def test_sample_module_symbols() -> None:
    parsed = parse_source(_load("sample_module.py"), "sample_module.py")
    by_qname = {s.qualified_name: s for s in parsed.symbols}

    assert by_qname["sample_module.py::double"].kind == SymbolKind.FUNCTION

    greeter = by_qname["sample_module.py::Greeter"]
    assert greeter.kind == SymbolKind.CLASS
    assert greeter.docstring == "Greets people by name."

    assert by_qname["sample_module.py::Greeter.__init__"].kind == SymbolKind.METHOD
    assert by_qname["sample_module.py::Greeter.__init__"].docstring == "Store the name to greet."
    assert by_qname["sample_module.py::LoudGreeter.greet"].kind == SymbolKind.METHOD
    assert by_qname["sample_module.py::MAX_RETRIES"].kind == SymbolKind.CONSTANT

    imports = {s.name: s for s in parsed.symbols if s.kind == SymbolKind.IMPORT}
    assert imports["os"].source_module == "os"
    assert imports["j"].source_module == "json"
    assert imports["OrderedDict"].source_module == "collections"
    assert imports["Opt"].source_module == "typing"

    # nested function is folded into its parent, not indexed as its own symbol
    assert "helper" not in {s.name for s in parsed.symbols}


def test_sample_module_references() -> None:
    parsed = parse_source(_load("sample_module.py"), "sample_module.py")
    calls = [r for r in parsed.references if r.kind == ReferenceKind.CALL]
    subclasses = [r for r in parsed.references if r.kind == ReferenceKind.SUBCLASS]

    assert any(
        r.name == "helper" and r.enclosing_qualified_name == "sample_module.py::double" for r in calls
    )
    assert any(
        r.name == "double" and r.enclosing_qualified_name == "sample_module.py::Greeter.greet" for r in calls
    )
    assert any(
        r.name == "_format" and r.enclosing_qualified_name == "sample_module.py::Greeter.greet" for r in calls
    )
    assert any(
        r.name == "double" and r.enclosing_qualified_name == "sample_module.py::LoudGreeter.greet"
        for r in calls
    )

    assert len(subclasses) == 1
    assert subclasses[0].name == "Greeter"
    assert subclasses[0].enclosing_qualified_name == "sample_module.py::LoudGreeter"


def test_real_file_from_repo() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "services" / "api" / "src" / "api" / "main.py"
    parsed = parse_source(target.read_text(), "services/api/src/api/main.py")

    function_names = {c.name for c in parsed.chunks if c.kind == ChunkKind.FUNCTION}
    assert function_names == {"health"}

    import_names = {s.name for s in parsed.symbols if s.kind == SymbolKind.IMPORT}
    assert "FastAPI" in import_names
