"""Unified evidence schema shared between retriever and code-navigation results (decision #35)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    source: Literal["retriever", "code_navigation"]
    file_path: str
    line_range: tuple[int, int]
    content: str
    score: float | None = None
    symbol_name: str | None = None
    symbol_kind: str | None = None
    reference_kind: str | None = None
