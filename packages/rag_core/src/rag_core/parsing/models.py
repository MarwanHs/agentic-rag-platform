from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChunkKind(str, Enum):
    FUNCTION = "function"
    MODULE = "module"


class SymbolKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CONSTANT = "constant"
    IMPORT = "import"


class ReferenceKind(str, Enum):
    CALL = "call"
    SUBCLASS = "subclass"


@dataclass(frozen=True, slots=True)
class Chunk:
    file_path: str
    kind: ChunkKind
    name: str | None
    qualified_name: str
    start_line: int
    end_line: int
    text: str
    docstring: str | None = None


@dataclass(frozen=True, slots=True)
class Symbol:
    file_path: str
    kind: SymbolKind
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source_text: str
    docstring: str | None = None
    source_module: str | None = None


@dataclass(frozen=True, slots=True)
class Reference:
    file_path: str
    kind: ReferenceKind
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    source_text: str
    enclosing_qualified_name: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedFile:
    file_path: str
    chunks: list[Chunk]
    symbols: list[Symbol]
    references: list[Reference]
