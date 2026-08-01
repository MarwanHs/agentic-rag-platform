"""Voyage reranker, applied to the fused hybrid-search candidate set (decision #20)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import voyageai

DEFAULT_RERANK_MODEL = "rerank-2"


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class RerankClient(Protocol):
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]: ...


class VoyageRerankClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_RERANK_MODEL) -> None:
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        result = self._client.rerank(query=query, documents=documents, model=self._model, top_k=top_k)
        return [RerankResult(index=item.index, score=item.relevance_score) for item in result.results]
