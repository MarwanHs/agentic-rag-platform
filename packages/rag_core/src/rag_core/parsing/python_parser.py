"""Tree-sitter-based structural parser for Python source.

Produces retrieval chunks (function-level, module-level) and code-navigation
entities (symbols, references) from a single AST pass, per the "Shared
parsing layer" section of docs/architecture.md. Docstrings and leading
comments are attached structurally (first-statement-in-body, immediately
preceding named sibling) rather than via line-proximity heuristics.
"""

from __future__ import annotations

import re
import textwrap

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from rag_core.parsing.models import (
    Chunk,
    ChunkKind,
    ParsedFile,
    Reference,
    ReferenceKind,
    Symbol,
    SymbolKind,
)

_LANGUAGE = Language(tspython.language())

_STRING_PREFIX_RE = re.compile(r"^[a-zA-Z]*")
_QUOTES = ('"""', "'''", '"', "'")


def parse_source(source: str, file_path: str) -> ParsedFile:
    parser = Parser(_LANGUAGE)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    extractor = _Extractor(source_bytes, file_path)
    extractor.run(tree.root_node)
    return ParsedFile(
        file_path=file_path,
        chunks=extractor.chunks,
        symbols=extractor.symbols,
        references=extractor.references,
    )


class _Extractor:
    def __init__(self, source_bytes: bytes, file_path: str) -> None:
        self.source = source_bytes
        self.file_path = file_path
        self.chunks: list[Chunk] = []
        self.symbols: list[Symbol] = []
        self.references: list[Reference] = []
        self._function_spans: list[tuple[int, int]] = []

    def run(self, root: Node) -> None:
        for child in root.named_children:
            self._visit(child, scope=[])
        self._build_module_chunk(root)

    # -- dispatch -----------------------------------------------------

    def _visit(self, node: Node, scope: list[str]) -> None:
        if node.type == "decorated_definition":
            inner = node.named_children[-1]
            if inner.type == "function_definition":
                self._handle_function(node, inner, scope)
            elif inner.type == "class_definition":
                self._handle_class(node, inner, scope)
        elif node.type == "function_definition":
            self._handle_function(node, node, scope)
        elif node.type == "class_definition":
            self._handle_class(node, node, scope)
        elif node.type in ("import_statement", "import_from_statement"):
            self._handle_import(node, scope)
        elif node.type == "expression_statement" and not scope:
            self._handle_possible_constant(node)

    # -- functions / methods --------------------------------------------

    def _handle_function(self, outer_node: Node, def_node: Node, scope: list[str]) -> None:
        name = self._text(def_node.child_by_field_name("name"))
        qualified_name = self._qualify(scope, name)
        kind = SymbolKind.METHOD if scope else SymbolKind.FUNCTION

        span_start = self._leading_comment_start(outer_node)
        span_end = outer_node.end_byte
        self._function_spans.append((span_start, span_end))

        docstring = self._docstring_from_container(def_node.child_by_field_name("body"))

        self.chunks.append(
            Chunk(
                file_path=self.file_path,
                kind=ChunkKind.FUNCTION,
                name=name,
                qualified_name=qualified_name,
                start_line=self._line_for_byte(span_start),
                end_line=def_node.end_point[0] + 1,
                text=self.source[span_start:span_end].decode("utf-8"),
                docstring=docstring,
            )
        )
        self.symbols.append(
            Symbol(
                file_path=self.file_path,
                kind=kind,
                name=name,
                qualified_name=qualified_name,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte,
                docstring=docstring,
            )
        )

        body = def_node.child_by_field_name("body")
        if body is not None:
            self._collect_references(body, enclosing=qualified_name)

    # -- classes ----------------------------------------------------------

    def _handle_class(self, outer_node: Node, def_node: Node, scope: list[str]) -> None:
        name = self._text(def_node.child_by_field_name("name"))
        qualified_name = self._qualify(scope, name)

        self.symbols.append(
            Symbol(
                file_path=self.file_path,
                kind=SymbolKind.CLASS,
                name=name,
                qualified_name=qualified_name,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                start_byte=def_node.start_byte,
                end_byte=def_node.end_byte,
                docstring=self._docstring_from_container(def_node.child_by_field_name("body")),
            )
        )

        superclasses = def_node.child_by_field_name("superclasses")
        if superclasses is not None:
            for arg in superclasses.named_children:
                base_name = self._rightmost_name(arg)
                if base_name is None:
                    continue
                self.references.append(
                    Reference(
                        file_path=self.file_path,
                        kind=ReferenceKind.SUBCLASS,
                        name=base_name,
                        start_line=arg.start_point[0] + 1,
                        end_line=arg.end_point[0] + 1,
                        start_byte=arg.start_byte,
                        end_byte=arg.end_byte,
                        enclosing_qualified_name=qualified_name,
                    )
                )

        body = def_node.child_by_field_name("body")
        new_scope = [*scope, name]
        if body is not None:
            for child in body.named_children:
                self._visit(child, new_scope)

    # -- imports ------------------------------------------------------

    def _handle_import(self, node: Node, scope: list[str]) -> None:
        if scope:
            return
        if node.type == "import_statement":
            for child in node.named_children:
                self._add_import_symbol(child, source_module=None)
        else:
            module_node = node.child_by_field_name("module_name")
            source_module = self._text(module_node) if module_node is not None else None
            for child in node.named_children:
                if child is module_node or child.type == "wildcard_import":
                    continue
                self._add_import_symbol(child, source_module=source_module)

    def _add_import_symbol(self, node: Node, source_module: str | None) -> None:
        if node.type == "dotted_name":
            full = self._text(node)
            name = full if source_module is not None else full.split(".")[0]
            self._append_import(node, name, source_module or full)
        elif node.type == "aliased_import":
            original = self._text(node.child_by_field_name("name"))
            alias = self._text(node.child_by_field_name("alias"))
            self._append_import(node, alias, source_module or original)
        elif node.type == "identifier":
            self._append_import(node, self._text(node), source_module)

    def _append_import(self, node: Node, name: str, source_module: str | None) -> None:
        self.symbols.append(
            Symbol(
                file_path=self.file_path,
                kind=SymbolKind.IMPORT,
                name=name,
                qualified_name=self._qualify([], name),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                source_module=source_module,
            )
        )

    # -- module-level constants --------------------------------------

    def _handle_possible_constant(self, node: Node) -> None:
        if not node.named_children:
            return
        child = node.named_children[0]
        if child.type != "assignment":
            return
        left = child.child_by_field_name("left")
        if left is None or left.type != "identifier":
            return
        self.symbols.append(
            Symbol(
                file_path=self.file_path,
                kind=SymbolKind.CONSTANT,
                name=self._text(left),
                qualified_name=self._qualify([], self._text(left)),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            )
        )

    # -- references (calls) --------------------------------------------

    def _collect_references(self, node: Node, enclosing: str) -> None:
        for child in node.named_children:
            if child.type == "call":
                name = self._rightmost_name(child.child_by_field_name("function"))
                if name is not None:
                    self.references.append(
                        Reference(
                            file_path=self.file_path,
                            kind=ReferenceKind.CALL,
                            name=name,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_byte=child.start_byte,
                            end_byte=child.end_byte,
                            enclosing_qualified_name=enclosing,
                        )
                    )
            self._collect_references(child, enclosing)

    # -- module chunk -----------------------------------------------------

    def _build_module_chunk(self, root: Node) -> None:
        pieces: list[bytes] = []
        cursor = 0
        for start, end in sorted(self._function_spans):
            if start > cursor:
                pieces.append(self.source[cursor:start])
            cursor = max(cursor, end)
        if cursor < len(self.source):
            pieces.append(self.source[cursor:])
        text = b"".join(pieces).decode("utf-8")
        docstring = self._docstring_from_container(root)

        if not text.strip() and not docstring:
            return

        self.chunks.insert(
            0,
            Chunk(
                file_path=self.file_path,
                kind=ChunkKind.MODULE,
                name=None,
                qualified_name=self.file_path,
                start_line=1,
                end_line=root.end_point[0] + 1,
                text=text,
                docstring=docstring,
            ),
        )

    # -- helpers ------------------------------------------------------

    def _text(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.source[node.start_byte : node.end_byte].decode("utf-8")

    def _qualify(self, scope: list[str], name: str) -> str:
        dotted = ".".join([*scope, name])
        return f"{self.file_path}::{dotted}"

    def _line_for_byte(self, offset: int) -> int:
        return self.source.count(b"\n", 0, offset) + 1

    def _leading_comment_start(self, node: Node) -> int:
        start = node.start_byte
        sibling = node.prev_named_sibling
        while sibling is not None and sibling.type == "comment":
            start = sibling.start_byte
            sibling = sibling.prev_named_sibling
        return start

    def _docstring_from_container(self, container: Node | None) -> str | None:
        if container is None or not container.named_children:
            return None
        first = container.named_children[0]
        if first.type != "expression_statement" or not first.named_children:
            return None
        inner = first.named_children[0]
        if inner.type != "string":
            return None
        return _clean_string_literal(self._text(inner))

    def _rightmost_name(self, node: Node | None) -> str | None:
        if node is None:
            return None
        if node.type == "identifier":
            return self._text(node)
        if node.type == "attribute":
            attr = node.child_by_field_name("attribute")
            return self._text(attr) if attr is not None else None
        return None


def _clean_string_literal(raw: str) -> str:
    text = raw
    match = _STRING_PREFIX_RE.match(text)
    if match:
        text = text[match.end() :]
    for quote in _QUOTES:
        if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
            text = text[len(quote) : -len(quote)]
            break
    return textwrap.dedent(text).strip()
