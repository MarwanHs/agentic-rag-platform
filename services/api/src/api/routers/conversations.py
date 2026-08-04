from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pg_conn
from api.models import CreateConversationResponse, MessageRequest, MessageResponse
from rag_core.conversations.store import create_conversation, get_conversation
from rag_core.jobs.store import get_job

router = APIRouter(tags=["conversations"])


@router.post(
    "/codebases/{codebase_id}/conversations",
    response_model=CreateConversationResponse,
    status_code=201,
)
def create_conversation_endpoint(
    codebase_id: str, conn: psycopg.Connection = Depends(get_pg_conn)
) -> CreateConversationResponse:
    job = get_job(conn, codebase_id)
    if job is None:
        raise HTTPException(status_code=404, detail="codebase not found")
    if job.status != "ready":
        raise HTTPException(status_code=409, detail=f"codebase is not ready (status: {job.status})")

    conversation = create_conversation(conn, codebase_id)
    return CreateConversationResponse(conversation_id=conversation.id)


@router.post("/conversations/{conversation_id}/messages", response_model=MessageResponse)
def send_message(
    conversation_id: str,
    body: MessageRequest,
    conn: psycopg.Connection = Depends(get_pg_conn),
) -> MessageResponse:
    conversation = get_conversation(conn, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    # No planner, critic-synthesizer, or LangGraph orchestration exists yet
    # (decisions #9, #11, #29). Faking `refused` or `used_existing_context`
    # here would be indistinguishable, downstream, from a real planner/critic
    # verdict -- same principle as POST /codebases/{id}/query's 501 stub.
    raise HTTPException(
        status_code=501,
        detail="Conversational query answering is not implemented yet: no planner, critic-synthesizer, "
        "or LangGraph orchestration exists (see docs/architecture.md decisions #9, #11, #29).",
    )
