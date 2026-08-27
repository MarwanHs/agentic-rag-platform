"""Shared FastAPI dependencies: datastore connections and Voyage clients.

Reused as-is from the retrieval/code-navigation milestone rather than
building a second connection path -- same env vars, same client classes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
from qdrant_client import QdrantClient

from agent_orchestrator.critic import AnthropicCriticClient, CriticClient
from agent_orchestrator.planner import AnthropicPlannerClient, PlannerClient
from agent_orchestrator.retriever_agent import AnthropicReformulationClient, ReformulationClient
from rag_core.retrieval.embeddings import EmbeddingClient, VoyageEmbeddingClient
from rag_core.retrieval.reranker import RerankClient, VoyageRerankClient

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

_qdrant_client: QdrantClient | None = None
_embedding_client: EmbeddingClient | None = None
_rerank_client: RerankClient | None = None
_planner_client: PlannerClient | None = None
_reformulation_client: ReformulationClient | None = None
_critic_client: CriticClient | None = None


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


def get_planner_client() -> PlannerClient:
    global _planner_client
    if _planner_client is None:
        _planner_client = AnthropicPlannerClient()
    return _planner_client


def get_reformulation_client() -> ReformulationClient:
    global _reformulation_client
    if _reformulation_client is None:
        _reformulation_client = AnthropicReformulationClient()
    return _reformulation_client


def get_critic_client() -> CriticClient:
    global _critic_client
    if _critic_client is None:
        _critic_client = AnthropicCriticClient()
    return _critic_client
