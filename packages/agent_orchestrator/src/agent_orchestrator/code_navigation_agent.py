"""Code-navigation-as-agent: deterministic symbol/reference evidence gathering (decision #44).

Given the planner's selected symbol names, calls both find_definition and
find_references for every name, unconditionally -- the planner doesn't (and
per decision #34, shouldn't) say whether "where's this defined" or "who
calls this" was intended, so both run and the critic sorts out relevance.

No LLM/external API dependency (decision #34): unlike the planner, this is
a plain function over a real Postgres connection, with no Protocol/
fake-client pattern needed.
"""

from __future__ import annotations

import logging

import psycopg

from rag_core.code_navigation.lookup import (
    ReferenceLocation,
    SymbolLocation,
    find_definition,
    find_references,
)
from shared.evidence import EvidenceItem

logger = logging.getLogger(__name__)


def gather_code_navigation_evidence(
    conn: psycopg.Connection, codebase_id: str, symbol_names: list[str]
) -> list[EvidenceItem]:
    deduped_names: list[str] = []
    seen: set[str] = set()
    for name in symbol_names:
        if name not in seen:
            seen.add(name)
            deduped_names.append(name)

    evidence: list[EvidenceItem] = []
    for name in deduped_names:
        definitions = find_definition(conn, codebase_id, name)
        references = find_references(conn, codebase_id, name)

        if not definitions and not references:
            logger.info("code-navigation: no matches for symbol %r in codebase %s", name, codebase_id)
            continue

        evidence.extend(_definition_evidence(codebase_id, loc) for loc in definitions)
        evidence.extend(_reference_evidence(codebase_id, loc) for loc in references)

    return evidence


def _definition_evidence(codebase_id: str, loc: SymbolLocation) -> EvidenceItem:
    return EvidenceItem(
        source="code_navigation",
        file_path=loc.file_path,
        line_range=(loc.start_line, loc.end_line),
        content=_definition_content(codebase_id, loc),
        symbol_name=loc.name,
        symbol_kind=loc.kind,
        reference_kind="definition",
    )


def _definition_content(codebase_id: str, loc: SymbolLocation) -> str:
    if loc.source_text is not None:
        return loc.source_text
    logger.warning(
        "code-navigation: source_text missing for symbol %r in codebase %s -- re-ingest to include it",
        loc.qualified_name,
        codebase_id,
    )
    if loc.docstring:
        return loc.docstring
    return f"{loc.kind} {loc.qualified_name} (source text unavailable -- re-ingest this codebase to include it)"


def _reference_evidence(codebase_id: str, loc: ReferenceLocation) -> EvidenceItem:
    return EvidenceItem(
        source="code_navigation",
        file_path=loc.file_path,
        line_range=(loc.start_line, loc.end_line),
        content=_reference_content(codebase_id, loc),
        symbol_name=loc.name,
        symbol_kind=None,
        reference_kind=loc.kind,
    )


def _reference_content(codebase_id: str, loc: ReferenceLocation) -> str:
    if loc.source_text is not None:
        return loc.source_text
    logger.warning(
        "code-navigation: source_text missing for reference to %r in codebase %s -- re-ingest to include it",
        loc.name,
        codebase_id,
    )
    return ""
