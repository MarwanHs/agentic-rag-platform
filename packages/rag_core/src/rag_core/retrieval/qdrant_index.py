"""Per-codebase Qdrant collection: dense (voyage-code-3) + native BM25 sparse
vectors, fused with RRF (decisions #14, #19, #20, #23), reranked with Voyage's
reranker before reaching the critic-synthesizer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from rag_core.parsing.models import Chunk
from rag_core.retrieval.embeddings import DEFAULT_OUTPUT_DIMENSION, EmbeddingClient
from rag_core.retrieval.reranker import RerankClient

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL_NAME = "Qdrant/bm25"
DEFAULT_PREFETCH_LIMIT = 25

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    return _bm25_model


def collection_name_for_codebase(codebase_id: str) -> str:
    return f"codebase_{codebase_id}"


def ensure_collection(
    client: QdrantClient, collection_name: str, dense_dim: int = DEFAULT_OUTPUT_DIMENSION
) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def _point_id(codebase_id: str, chunk: Chunk) -> str:
    raw = f"{codebase_id}:{chunk.qualified_name}:{chunk.start_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def upsert_chunks(
    client: QdrantClient,
    codebase_id: str,
    chunks: list[Chunk],
    embedding_client: EmbeddingClient,
) -> None:
    if not chunks:
        return
    collection_name = collection_name_for_codebase(codebase_id)
    texts = [chunk.text for chunk in chunks]
    dense_vectors = embedding_client.embed_documents(texts)
    sparse_vectors = list(_get_bm25_model().embed(texts))

    points = [
        models.PointStruct(
            id=_point_id(codebase_id, chunk),
            vector={
                DENSE_VECTOR_NAME: dense_vectors[i],
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_vectors[i].indices.tolist(),
                    values=sparse_vectors[i].values.tolist(),
                ),
            },
            payload={
                "file_path": chunk.file_path,
                "kind": chunk.kind.value,
                "name": chunk.name,
                "qualified_name": chunk.qualified_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "docstring": chunk.docstring,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)


@dataclass(frozen=True, slots=True)
class SearchResult:
    payload: dict
    score: float


def hybrid_search(
    client: QdrantClient,
    codebase_id: str,
    query_text: str,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
    limit: int = 10,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
) -> list[SearchResult]:
    collection_name = collection_name_for_codebase(codebase_id)
    dense_query = embedding_client.embed_query(query_text)
    sparse_query = next(iter(_get_bm25_model().embed([query_text])))

    fused = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=dense_query, using=DENSE_VECTOR_NAME, limit=prefetch_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                ),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,
        with_payload=True,
    ).points

    if not fused:
        return []

    documents = [point.payload["text"] for point in fused]
    reranked = rerank_client.rerank(query_text, documents, top_k=min(limit, len(documents)))

    return [SearchResult(payload=fused[result.index].payload, score=result.score) for result in reranked]
