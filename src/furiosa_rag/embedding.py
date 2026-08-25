"""Embedding backend interface and Furiosa implementation."""

from __future__ import annotations

import math
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
            rows = payload["data"]
            if not isinstance(rows, list):
                raise TypeError
            indices = [row["index"] for row in rows]
            if any(type(index) is not int for index in indices):
                raise TypeError
            if sorted(indices) != list(range(len(items))):
                raise ValueError
            ordered_rows = sorted(rows, key=lambda row: row["index"])
            vectors = [row["embedding"] for row in ordered_rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise FuriosaApiError("Embedding response has an invalid data field") from exc
        if len(vectors) != len(items) or any(not isinstance(vector, list) for vector in vectors):
            raise FuriosaApiError("Embedding response count or vector type is invalid")
        dimensions = {len(vector) for vector in vectors}
        if dimensions == {0} or len(dimensions) != 1:
            raise FuriosaApiError("Embedding vectors must have one consistent non-zero dimension")
        if any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for vector in vectors
            for value in vector
        ):
            raise FuriosaApiError("Embedding vectors must contain only finite numeric values")
        return vectors
