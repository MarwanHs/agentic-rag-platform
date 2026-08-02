from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI

from api.deps import DATABASE_URL
from api.routers.codebases import router as codebases_router
from api.routers.jobs import router as jobs_router
from rag_core.code_navigation.schema import ensure_schema as ensure_code_navigation_schema
from rag_core.jobs.schema import ensure_schema as ensure_jobs_schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_jobs_schema(conn)
        ensure_code_navigation_schema(conn)
    yield


app = FastAPI(title="agentic-rag-platform", lifespan=lifespan)
app.include_router(jobs_router)
app.include_router(codebases_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
