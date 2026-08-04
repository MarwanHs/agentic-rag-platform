from __future__ import annotations

from rag_core.conversations.schema import ensure_schema as ensure_conversations_schema
from rag_core.conversations.store import create_conversation, get_conversation
from rag_core.jobs.schema import ensure_schema as ensure_jobs_schema
from rag_core.jobs.store import create_job


def test_conversation_lifecycle(pg_conn) -> None:
    ensure_jobs_schema(pg_conn)
    ensure_conversations_schema(pg_conn)
    job = create_job(pg_conn, "https://github.com/example/repo")

    try:
        conversation = create_conversation(pg_conn, job.id)
        assert conversation.codebase_id == job.id

        fetched = get_conversation(pg_conn, conversation.id)
        assert fetched is not None
        assert fetched.id == conversation.id
        assert fetched.codebase_id == job.id
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE codebase_id = %s", (job.id,))
            cur.execute("DELETE FROM jobs WHERE id = %s", (job.id,))
        pg_conn.commit()


def test_get_unknown_conversation_returns_none(pg_conn) -> None:
    ensure_conversations_schema(pg_conn)
    assert get_conversation(pg_conn, "does-not-exist") is None
