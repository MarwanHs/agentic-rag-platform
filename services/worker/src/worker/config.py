from __future__ import annotations

import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Decision #38: default is one active ingestion job at a time -- the embed
# stage is bounded by the embedding API's rate limit, so running several
# jobs concurrently doesn't add real throughput without a rate-limit
# budgeting strategy this worker doesn't implement yet. Read from env so
# raising it later is a config change, not a code change; this first pass
# still only runs the single-job poll loop below regardless of the value.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))

POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", "2"))
