"""Embedding backend interface and Furiosa implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FuriosaEmbedding:
    def __init__(self, endpoint: ModelEndpoint, client: FuriosaClient) -> None:
        self.endpoint = endpoint
        self.client = client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items or any(not text.strip() for text in items):
            raise ValueError("texts must contain at least one non-empty string")

        payload = self.client.post_json(
            self.endpoint.base_url,
            "embeddings",
            {"model": self.endpoint.model, "input": items},
        )
        try:
            rows = sorted(payload["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise FuriosaApiError("Embedding response has an invalid data field") from exc
        if len(vectors) != len(items) or any(not isinstance(vector, list) for vector in vectors):
            raise FuriosaApiError("Embedding response count or vector type is invalid")
        return vectors

