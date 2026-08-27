from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import QdrantClient

from agent_orchestrator.critic import CriticClient
from agent_orchestrator.planner import PlannerClient
from agent_orchestrator.query import answer_query
from agent_orchestrator.retriever_agent import ReformulationClient
from api.deps import (
    get_critic_client,
    get_embedding_client,
    get_pg_conn,
    get_planner_client,
    get_qdrant_client,
    get_reformulation_client,
    get_rerank_client,
)
from api.models import Citation, CodebaseSummary, LineRange, QueryRequest, QueryResponse
from rag_core.jobs.store import get_job, list_ready_codebases
from rag_core.retrieval.embeddings import EmbeddingClient
from rag_core.retrieval.reranker import RerankClient

router = APIRouter(tags=["codebases"])


@router.get("/codebases", response_model=list[CodebaseSummary])
def list_codebases(conn: psycopg.Connection = Depends(get_pg_conn)) -> list[CodebaseSummary]:
    return [CodebaseSummary(id=c.id, url=c.url) for c in list_ready_codebases(conn)]


@router.post("/codebases/{codebase_id}/query", response_model=QueryResponse)
def query_codebase(
    codebase_id: str,
    body: QueryRequest,
    conn: psycopg.Connection = Depends(get_pg_conn),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
    embedding_client: EmbeddingClient = Depends(get_embedding_client),
    rerank_client: RerankClient = Depends(get_rerank_client),
    planner_client: PlannerClient = Depends(get_planner_client),
    reformulation_client: ReformulationClient = Depends(get_reformulation_client),
    critic_client: CriticClient = Depends(get_critic_client),
) -> QueryResponse:
    job = get_job(conn, codebase_id)
    if job is None:
        raise HTTPException(status_code=404, detail="codebase not found")
    if job.status != "ready":
        raise HTTPException(status_code=409, detail=f"codebase is not ready (status: {job.status})")

    result = answer_query(
        body.question,
        codebase_id,
        conn,
        qdrant_client,
        embedding_client,
        rerank_client,
        planner_client,
        reformulation_client,
        critic_client,
    )

    return QueryResponse(
        answer=result.answer,
        refused=result.refused,
        reason=result.reason,
        citations=[
            Citation(
                source=c.source,
                file_path=c.file_path,
                line_range=LineRange(start=c.line_range[0], end=c.line_range[1]),
            )
            for c in result.citations
        ],
        sources_used=result.sources_used,
    )
