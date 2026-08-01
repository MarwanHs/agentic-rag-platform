from __future__ import annotations

import hashlib
import os
import socket
from urllib.parse import urlparse

import psycopg
import pytest
from qdrant_client import QdrantClient

from rag_core.retrieval.reranker import RerankResult

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("TEST_QDRANT_URL", "http://localhost:6333")


def _is_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def pg_conn():
    parsed = urlparse(DATABASE_URL)
    if not _is_reachable(parsed.hostname or "localhost", parsed.port or 5432):
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}; run `docker compose up -d postgres`.")
    conn = psycopg.connect(DATABASE_URL)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def qdrant_client():
    parsed = urlparse(QDRANT_URL)
    if not _is_reachable(parsed.hostname or "localhost", parsed.port or 6333):
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}; run `docker compose up -d qdrant`.")
    return QdrantClient(url=QDRANT_URL)


class FakeEmbeddingClient:
    """Deterministic stand-in for VoyageEmbeddingClient. Not semantically
    meaningful (hash-based) -- tests that need real relevance ranking rely
    on the real BM25 sparse vectors and the fake reranker's token overlap,
    not on this dense signal.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeRerankClient:
    """Deterministic stand-in for VoyageRerankClient, ranking by token overlap."""

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        q_tokens = set(query.lower().split())
        scored = sorted(
            range(len(documents)),
            key=lambda i: -len(q_tokens & set(documents[i].lower().split())),
        )[:top_k]
        return [RerankResult(index=i, score=1.0 - rank * 0.01) for rank, i in enumerate(scored)]


@pytest.fixture()
def fake_embedding_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture()
def fake_rerank_client() -> FakeRerankClient:
    return FakeRerankClient()
