"""Shared FastAPI dependencies: datastore connections and Voyage clients.

Reused as-is from the retrieval/code-navigation milestone rather than
building a second connection path -- same env vars, same client classes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
from qdrant_client import QdrantClient

from rag_core.retrieval.embeddings import EmbeddingClient, VoyageEmbeddingClient
from rag_core.retrieval.reranker import RerankClient, VoyageRerankClient

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

_qdrant_client: QdrantClient | None = None
_embedding_client: EmbeddingClient | None = None
_rerank_client: RerankClient | None = None


def get_pg_conn() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = VoyageEmbeddingClient()
    return _embedding_client


def get_rerank_client() -> RerankClient:
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = VoyageRerankClient()
    return _rerank_client
