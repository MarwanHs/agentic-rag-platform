"""voyage-code-3 embedding client (decision #19)."""

from __future__ import annotations

from typing import Protocol

import voyageai

DEFAULT_EMBEDDING_MODEL = "voyage-code-3"
DEFAULT_OUTPUT_DIMENSION = 1024


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        output_dimension: int = DEFAULT_OUTPUT_DIMENSION,
    ) -> None:
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self._output_dimension = output_dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(
            texts,
            model=self._model,
            input_type="document",
            output_dimension=self._output_dimension,
        )
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed(
            [text],
            model=self._model,
            input_type="query",
            output_dimension=self._output_dimension,
        )
        return result.embeddings[0]
