from __future__ import annotations

import uuid

from rag_core.parsing.models import Chunk, ChunkKind
from rag_core.retrieval.qdrant_index import (
    collection_name_for_codebase,
    ensure_collection,
    hybrid_search,
    upsert_chunks,
)


def test_hybrid_search_returns_relevant_chunk(qdrant_client, fake_embedding_client, fake_rerank_client) -> None:
    codebase_id = f"test-{uuid.uuid4().hex[:8]}"
    collection_name = collection_name_for_codebase(codebase_id)
    ensure_collection(qdrant_client, collection_name)

    chunks = [
        Chunk(
            file_path="sample.py",
            kind=ChunkKind.FUNCTION,
            name="parse_config",
            qualified_name="sample.py::parse_config",
            start_line=1,
            end_line=5,
            text='def parse_config(path):\n    """Parse a YAML config file into a dict."""\n    ...',
            docstring="Parse a YAML config file into a dict.",
        ),
        Chunk(
            file_path="sample.py",
            kind=ChunkKind.FUNCTION,
            name="send_email",
            qualified_name="sample.py::send_email",
            start_line=7,
            end_line=11,
            text='def send_email(to, subject, body):\n    """Send an email via SMTP."""\n    ...',
            docstring="Send an email via SMTP.",
        ),
    ]

    try:
        upsert_chunks(qdrant_client, codebase_id, chunks, fake_embedding_client)

        results = hybrid_search(
            qdrant_client,
            codebase_id,
            "how do we parse the config file",
            fake_embedding_client,
            fake_rerank_client,
            limit=1,
        )

        assert len(results) == 1
        assert results[0].payload["name"] == "parse_config"
    finally:
        qdrant_client.delete_collection(collection_name)
