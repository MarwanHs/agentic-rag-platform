"""Request/response schemas for the API contract (decision #26)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "cloning", "parsing", "embedding", "indexing", "ready", "failed"]
AgentSource = Literal["retriever", "code_navigation"]


class CreateJobRequest(BaseModel):
    url: str


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_batch: str | None = None
    failure_reason: str | None = None


class CodebaseSummary(BaseModel):
    id: str
    url: str


class QueryRequest(BaseModel):
    question: str


class LineRange(BaseModel):
    start: int
    end: int


class Citation(BaseModel):
    source: AgentSource
    file_path: str
    line_range: LineRange


class QueryResponse(BaseModel):
    answer: str | None
    refused: bool
    reason: str | None
    citations: list[Citation]
    sources_used: list[AgentSource]
