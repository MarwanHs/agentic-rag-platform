"""Standalone ingestion worker loop (decisions #37, #38).

Single-process, one-job-at-a-time: claims the oldest queued job via
`dequeue_next_job`'s `FOR UPDATE SKIP LOCKED`, and runs the pipeline to
completion (success or failure) before dequeuing again. `MAX_CONCURRENT_JOBS`
is read but not used to run multiple jobs concurrently yet -- see
docs/architecture.md decision #38; scheduling for that is future work, not
built here.

If this process dies mid-job, that job is left stuck in a non-'queued'
status and `dequeue_next_job` won't pick it up again. That's a known,
deliberately unaddressed gap -- crash recovery is out of scope for now.
"""

from __future__ import annotations

import logging
import time

import psycopg
from qdrant_client import QdrantClient

from rag_core.code_navigation.schema import ensure_schema as ensure_code_navigation_schema
from rag_core.ingestion.pipeline import run_pipeline
from rag_core.jobs.schema import ensure_schema as ensure_jobs_schema
from rag_core.jobs.store import dequeue_next_job
from rag_core.retrieval.embeddings import VoyageEmbeddingClient
from worker.config import DATABASE_URL, MAX_CONCURRENT_JOBS, POLL_INTERVAL_SECONDS, QDRANT_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")


def run_forever() -> None:
    logger.info(
        "worker starting (max_concurrent_jobs=%s, poll_interval=%ss)",
        MAX_CONCURRENT_JOBS,
        POLL_INTERVAL_SECONDS,
    )
    conn = psycopg.connect(DATABASE_URL)
    ensure_jobs_schema(conn)
    ensure_code_navigation_schema(conn)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    embedding_client = VoyageEmbeddingClient()

    try:
        while True:
            job = dequeue_next_job(conn)
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            logger.info("claimed job %s (%s)", job.id, job.url)
            run_pipeline(conn, qdrant_client, embedding_client, job)
            logger.info("finished job %s", job.id)
    finally:
        conn.close()


if __name__ == "__main__":
    run_forever()
