from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pg_conn
from api.models import CodebaseSummary, QueryRequest, QueryResponse
from rag_core.jobs.store import get_job, list_ready_codebases

router = APIRouter(tags=["codebases"])


@router.get("/codebases", response_model=list[CodebaseSummary])
def list_codebases(conn: psycopg.Connection = Depends(get_pg_conn)) -> list[CodebaseSummary]:
    return [CodebaseSummary(id=c.id, url=c.url) for c in list_ready_codebases(conn)]


@router.post("/codebases/{codebase_id}/query", response_model=QueryResponse)
def query_codebase(
    codebase_id: str,
    body: QueryRequest,
    conn: psycopg.Connection = Depends(get_pg_conn),
) -> QueryResponse:
    job = get_job(conn, codebase_id)
    if job is None:
        raise HTTPException(status_code=404, detail="codebase not found")
    if job.status != "ready":
        raise HTTPException(status_code=409, detail=f"codebase is not ready (status: {job.status})")

    # No planner (decision #9) and no critic-synthesizer (decision #11) exist
    # yet. `refused=true` in QueryResponse is specifically the critic's
    # sufficiency verdict -- returning it here without a critic would be
    # indistinguishable, from the response shape alone, from a real refusal.
    # So this fails loudly instead of faking that verdict; wire up the real
    # response once the planner/critic exist (next milestone).
    raise HTTPException(
        status_code=501,
        detail="Query answering is not implemented yet: no planner or critic-synthesizer exists "
        "(see docs/architecture.md decisions #9, #11).",
    )
