"""Conversation row lifecycle: create and look up by id.

Orchestration/message-turn handling doesn't exist yet (decision #29's
LangGraph checkpointer is a future milestone) -- this only persists which
conversations exist, mirroring rag_core.jobs.store's shape.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    codebase_id: str


def create_conversation(conn: psycopg.Connection, codebase_id: str) -> Conversation:
    conversation_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (id, codebase_id) VALUES (%s, %s)",
            (conversation_id, codebase_id),
        )
    conn.commit()
    return Conversation(id=conversation_id, codebase_id=codebase_id)


def get_conversation(conn: psycopg.Connection, conversation_id: str) -> Conversation | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id, codebase_id FROM conversations WHERE id = %s", (conversation_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return Conversation(id=row[0], codebase_id=row[1])
