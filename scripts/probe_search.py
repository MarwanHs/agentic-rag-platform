#!/usr/bin/env python3
"""Probe real hybrid search against an already-ingested codebase.

Not the agentic query pipeline -- there's no planner or critic-synthesizer
yet. This calls rag_core.retrieval.qdrant_index.hybrid_search directly, the
same function the retriever agent will eventually call, against real
ingested data: a real embedding call, real Qdrant dense+BM25 fusion (RRF),
a real Voyage rerank pass, and real chunks back. Proof retrieval itself
works, independent of whether the orchestration layer exists yet.

Usage:
    uv run python scripts/probe_search.py "how do we claim the next queued job"
    uv run python scripts/probe_search.py "hybrid search" --codebase-id <job_id> --limit 3

Requires VOYAGE_API_KEY. Defaults to the most recently ingested `ready` codebase.
"""

from __future__ import annotations

import argparse
import os
import sys

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

DEFAULT_QUERY = "how do we claim the next queued job"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="question to search for")
    parser.add_argument("--codebase-id", default=None, help="defaults to the most recently ingested ready codebase")
    parser.add_argument("--limit", type=int, default=5, help="number of results to show (default: 5)")
    args = parser.parse_args()

    if not os.environ.get("VOYAGE_API_KEY"):
        fail("VOYAGE_API_KEY is not set -- export it before running this script")

    import psycopg
    from qdrant_client import QdrantClient

    from rag_core.jobs.store import list_ready_codebases
    from rag_core.retrieval.embeddings import VoyageEmbeddingClient
    from rag_core.retrieval.qdrant_index import hybrid_search
    from rag_core.retrieval.reranker import VoyageRerankClient

    conn = psycopg.connect(DATABASE_URL)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    codebase_id = args.codebase_id
    if codebase_id is None:
        ready = list_ready_codebases(conn)
        if not ready:
            fail("no ready codebases found -- run scripts/smoke_test_ingestion.py first")
        codebase_id = ready[0].id
        print(f"no --codebase-id given, using most recently ingested: {codebase_id} ({ready[0].url})")

    print(f'searching codebase {codebase_id} for: "{args.query}"\n')

    embedding_client = VoyageEmbeddingClient()
    rerank_client = VoyageRerankClient()
    results = hybrid_search(qdrant_client, codebase_id, args.query, embedding_client, rerank_client, limit=args.limit)

    if not results:
        fail("hybrid_search returned zero results -- collection may be empty, or wrong codebase_id")

    for rank, result in enumerate(results, start=1):
        payload = result.payload
        location = f"{payload['file_path']}:{payload['start_line']}-{payload['end_line']}"
        print(f"#{rank}  score={result.score:.3f}  {payload['qualified_name']}  ({location})")
        lines = payload["text"].strip().splitlines()
        for line in lines[:4]:
            print(f"    {line}")
        if len(lines) > 4:
            print("    ...")
        print()

    conn.close()


if __name__ == "__main__":
    main()
