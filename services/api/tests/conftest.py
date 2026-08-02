from __future__ import annotations

import socket
from urllib.parse import urlparse

import psycopg
import pytest
from fastapi.testclient import TestClient

from api.deps import DATABASE_URL, get_embedding_client, get_rerank_client
from api.main import app
from rag_core.retrieval.reranker import RerankResult


def _is_reachable(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_datastores() -> None:
    pg = urlparse(DATABASE_URL)
    if not _is_reachable(pg.hostname or "localhost", pg.port or 5432):
        pytest.skip(f"Postgres not reachable at {DATABASE_URL}; run `docker compose up -d postgres`.")


class FakeEmbeddingClient:
    """Kept wired into the `client` fixture below even though no route
    currently calls the embedding client -- cheap insurance against a real
    Voyage call slipping in once the query endpoint starts using retrieval
    again (next milestone).
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeRerankClient:
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        q_tokens = set(query.lower().split())
        scored = sorted(
            range(len(documents)),
            key=lambda i: -len(q_tokens & set(documents[i].lower().split())),
        )[:top_k]
        return [RerankResult(index=i, score=1.0 - rank * 0.01) for rank, i in enumerate(scored)]


@pytest.fixture()
def client():
    app.dependency_overrides[get_embedding_client] = lambda: FakeEmbeddingClient()
    app.dependency_overrides[get_rerank_client] = lambda: FakeRerankClient()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def pg_conn():
    conn = psycopg.connect(DATABASE_URL)
    yield conn
    conn.close()
