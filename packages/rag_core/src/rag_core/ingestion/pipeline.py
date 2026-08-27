"""Ingestion orchestrator: clone -> parse -> embed/index -> code-navigation
index for one queued job (decisions #3, #6, #7, #15, #19-21, #23, #37-39).

Batches are the unit of retry/progress for the embed stage (decision #7);
a job is only marked `ready` once every stage has fully succeeded, never
partially (decision #6). Symbols are inserted for every parsed file before
any references are inserted, per code_navigation.indexer's own contract --
interleaving per file would let a reference into a not-yet-indexed file fail
to resolve depending on file processing order.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import psycopg
from qdrant_client import QdrantClient

from rag_core.code_navigation.indexer import delete_file, upsert_references, upsert_symbols
from rag_core.ingestion.clone import clone_repo
from rag_core.jobs.store import Job, update_pipeline_state, update_status
from rag_core.parsing.models import ParsedFile
from rag_core.parsing.python_parser import parse_source
from rag_core.retrieval.embeddings import EmbeddingClient
from rag_core.retrieval.qdrant_index import collection_name_for_codebase, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)

# Decision #46: env-configurable, not a hardcoded constant -- an operational
# cost/throughput dial that plausibly differs per deployer depending on
# their Voyage rate-limit tier, same reasoning as MAX_CONCURRENT_JOBS
# (decision #38) and RERANK_CONFIDENCE_THRESHOLD (decision #45).
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
EMBED_MAX_ATTEMPTS = 3
EMBED_RETRY_BACKOFF_SECONDS = 2.0

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    ".eggs",
}


def discover_python_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def parse_files(root: Path, files: list[Path]) -> list[ParsedFile]:
    parsed = []
    for path in files:
        rel_path = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8", errors="replace")
        parsed.append(parse_source(source, rel_path))
    return parsed


def run_pipeline(
    conn: psycopg.Connection,
    qdrant_client: QdrantClient,
    embedding_client: EmbeddingClient,
    job: Job,
    clone_dir: Path | None = None,
) -> None:
    """Execute clone -> parse -> embed/index -> code-nav index for `job`.

    The job's own id doubles as the codebase id throughout (see
    jobs/schema.py). `clone_dir`, when passed, is an injection point for
    tests to skip the real `git clone` -- in that case the caller owns the
    directory and this function will not delete it.
    """
    codebase_id = job.id
    owns_clone_dir = clone_dir is None

    try:
        if clone_dir is None:
            update_status(conn, job.id, "cloning")
            clone_dir = clone_repo(job.url)

        update_status(conn, job.id, "parsing")
        files = discover_python_files(clone_dir)
        parsed_files = parse_files(clone_dir, files)
        logger.info("job %s: parsed %d Python file(s)", job.id, len(parsed_files))

        update_status(conn, job.id, "embedding")
        _run_embedding_stage(conn, qdrant_client, embedding_client, codebase_id, parsed_files)

        update_status(conn, job.id, "indexing")
        _run_indexing_stage(conn, codebase_id, parsed_files)

        update_status(conn, job.id, "ready")
        logger.info("job %s: ready", job.id)
    except Exception as exc:  # noqa: BLE001 -- any stage failure fails the job, never partially (decision #6)
        logger.exception("job %s: failed", job.id)
        conn.rollback()  # clear any aborted-transaction state left by a failed stage before writing status
        update_status(conn, job.id, "failed", failure_reason=str(exc))
    finally:
        if owns_clone_dir and clone_dir is not None:
            shutil.rmtree(clone_dir, ignore_errors=True)


def _run_embedding_stage(
    conn: psycopg.Connection,
    qdrant_client: QdrantClient,
    embedding_client: EmbeddingClient,
    codebase_id: str,
    parsed_files: list[ParsedFile],
) -> None:
    ensure_collection(qdrant_client, collection_name_for_codebase(codebase_id))

    all_chunks = [chunk for parsed in parsed_files for chunk in parsed.chunks]
    batches = [all_chunks[i : i + EMBED_BATCH_SIZE] for i in range(0, len(all_chunks), EMBED_BATCH_SIZE)]
    total = len(batches)

    for i, batch in enumerate(batches, start=1):
        _upsert_batch_with_retry(qdrant_client, codebase_id, batch, embedding_client)
        update_pipeline_state(conn, codebase_id, {"embed": {"batches_done": i, "batches_total": total}})
        logger.info("codebase %s: embedded batch %d/%d", codebase_id, i, total)


def _upsert_batch_with_retry(
    qdrant_client: QdrantClient,
    codebase_id: str,
    batch: list,
    embedding_client: EmbeddingClient,
) -> None:
    """Basic retry on transient embed/upsert errors. Not a rate-limit-aware
    backoff strategy -- flagged as a follow-up if real throughput needs it.
    """
    last_exc: Exception | None = None
    for attempt in range(1, EMBED_MAX_ATTEMPTS + 1):
        try:
            upsert_chunks(qdrant_client, codebase_id, batch, embedding_client)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < EMBED_MAX_ATTEMPTS:
                logger.warning(
                    "codebase %s: embed batch failed (attempt %d/%d), retrying: %s",
                    codebase_id,
                    attempt,
                    EMBED_MAX_ATTEMPTS,
                    exc,
                )
                time.sleep(EMBED_RETRY_BACKOFF_SECONDS)
    assert last_exc is not None
    raise last_exc


def _run_indexing_stage(conn: psycopg.Connection, codebase_id: str, parsed_files: list[ParsedFile]) -> None:
    # Defensive: clears any rows from a prior partial/retried run of this
    # codebase_id before re-inserting, so this stage stays safe to re-run.
    for parsed in parsed_files:
        delete_file(conn, codebase_id, parsed.file_path)

    for parsed in parsed_files:
        upsert_symbols(conn, codebase_id, parsed.symbols)
    for parsed in parsed_files:
        upsert_references(conn, codebase_id, parsed.references)

    conn.commit()
