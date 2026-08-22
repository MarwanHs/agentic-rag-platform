#!/usr/bin/env python3
"""End-to-end ingestion smoke test (decisions #3, #6, #15, #37).

Starts Postgres + Qdrant via docker compose, enqueues a real ingestion job
against a real repo (this project's own repo by default), runs it through
the same pipeline the worker uses, prints stage-by-stage progress as it
goes, then probes Qdrant and Postgres directly to confirm real data landed
before exiting 0 (pass) or 1 (fail).

This calls create_job + dequeue_next_job + run_pipeline directly rather than
going through the HTTP API and a separate worker process -- same code path,
fewer moving parts to manage for a one-shot test run.

Usage:
    uv run python scripts/smoke_test_ingestion.py
    uv run python scripts/smoke_test_ingestion.py --url https://github.com/some/repo
    uv run python scripts/smoke_test_ingestion.py --cleanup

Requires VOYAGE_API_KEY in the environment (real embedding calls happen here).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://github.com/MarwanHs/agentic-rag-platform"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_test")


def fail(message: str) -> None:
    logger.error("FAIL: %s", message)
    sys.exit(1)


def start_docker_services() -> None:
    if shutil.which("docker") is None:
        fail("docker is not on PATH -- install Docker Desktop (or equivalent) and try again")
    logger.info("starting postgres + qdrant via docker compose")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=REPO_ROOT, check=True)


def wait_for(name: str, check: Callable[[], None], timeout: float = 60.0) -> None:
    logger.info("waiting for %s to become reachable...", name)
    deadline = time.time() + timeout
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            check()
            logger.info("%s is up", name)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1)
    fail(f"{name} did not become reachable within {timeout:.0f}s: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="repo URL to ingest (default: this project's own repo)")
    parser.add_argument("--cleanup", action="store_true", help="delete the ingested data after a successful run")
    args = parser.parse_args()

    # fail fast on the cheap check before importing anything or touching docker
    if not os.environ.get("VOYAGE_API_KEY"):
        fail("VOYAGE_API_KEY is not set -- export it before running this script (real embedding calls happen here)")

    start_docker_services()

    import psycopg
    from qdrant_client import QdrantClient

    wait_for("postgres", lambda: psycopg.connect(DATABASE_URL).close())
    wait_for("qdrant", lambda: QdrantClient(url=QDRANT_URL).get_collections())

    from rag_core.code_navigation.schema import ensure_schema as ensure_code_navigation_schema
    from rag_core.ingestion.pipeline import run_pipeline
    from rag_core.jobs.schema import ensure_schema as ensure_jobs_schema
    from rag_core.jobs.store import create_job, dequeue_next_job, get_job
    from rag_core.retrieval.embeddings import VoyageEmbeddingClient
    from rag_core.retrieval.qdrant_index import collection_name_for_codebase

    conn = psycopg.connect(DATABASE_URL)
    ensure_jobs_schema(conn)
    ensure_code_navigation_schema(conn)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    embedding_client = VoyageEmbeddingClient()

    logger.info("creating job for %s", args.url)
    created = create_job(conn, args.url)

    job = dequeue_next_job(conn)
    if job is None or job.id != created.id:
        fail(
            "could not claim the job we just created -- is a real worker "
            "(services/worker) already running against this database and racing us for it?"
        )

    logger.info("running pipeline for job %s -- clone, parse, embed, index (this can take a while)", job.id)
    run_pipeline(conn, qdrant_client, embedding_client, job)

    fetched = get_job(conn, job.id)
    if fetched is None:
        fail("job row disappeared after running -- unexpected")
    logger.info("final job status: %s", fetched.status)

    if fetched.status == "failed":
        fail(f"pipeline reported failure: {fetched.failure_reason}")
    if fetched.status != "ready":
        fail(f"pipeline ended in unexpected status {fetched.status!r} (pipeline_state={fetched.pipeline_state})")

    logger.info("probing results...")
    collection_name = collection_name_for_codebase(job.id)
    if not qdrant_client.collection_exists(collection_name):
        fail(f"qdrant collection {collection_name} does not exist")
    point_count = qdrant_client.count(collection_name=collection_name).count
    if point_count == 0:
        fail("qdrant collection exists but has zero points")
    logger.info("qdrant: %d chunks embedded into %s", point_count, collection_name)

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM symbols WHERE codebase_id = %s", (job.id,))
        symbol_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM symbol_references WHERE codebase_id = %s", (job.id,))
        reference_count = cur.fetchone()[0]
    if symbol_count == 0:
        fail("postgres has zero symbols for this codebase -- code-navigation indexing did not run correctly")
    logger.info("postgres: %d symbols, %d references", symbol_count, reference_count)

    logger.info("PASS -- codebase_id=%s is ready (url=%s)", job.id, args.url)

    if args.cleanup:
        logger.info("--cleanup passed: deleting ingested data")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM symbol_references WHERE codebase_id = %s", (job.id,))
            cur.execute("DELETE FROM symbols WHERE codebase_id = %s", (job.id,))
            cur.execute("DELETE FROM jobs WHERE id = %s", (job.id,))
        conn.commit()
        qdrant_client.delete_collection(collection_name)
        logger.info("cleaned up")
    else:
        logger.info(
            "codebase_id=%s left in place for later use (e.g. querying once the planner/critic exist) "
            "-- pass --cleanup to remove it instead",
            job.id,
        )

    conn.close()


if __name__ == "__main__":
    main()
