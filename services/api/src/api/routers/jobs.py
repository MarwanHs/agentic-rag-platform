from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pg_conn
from api.models import CreateJobRequest, CreateJobResponse, JobStatusResponse
from rag_core.jobs.store import create_job, get_job

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=CreateJobResponse, status_code=202)
def create_job_endpoint(
    body: CreateJobRequest, conn: psycopg.Connection = Depends(get_pg_conn)
) -> CreateJobResponse:
    job = create_job(conn, body.url)
    return CreateJobResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, conn: psycopg.Connection = Depends(get_pg_conn)) -> JobStatusResponse:
    job = get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        pipeline_state=job.pipeline_state,
        failure_reason=job.failure_reason,
    )
