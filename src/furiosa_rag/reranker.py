"""Reranking backend interface and Furiosa implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint


@dataclass(frozen=True, slots=True)
class RankedDocument:
    index: int
    score: float
    text: str


class RerankerBackend(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int = 3
    ) -> list[RankedDocument]: ...


class FuriosaReranker:
    def __init__(self, endpoint: ModelEndpoint, client: FuriosaClient) -> None:
        self.endpoint = endpoint
        self.client = client

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int = 3
    ) -> list[RankedDocument]:
        items = list(documents)
        if not query.strip():
            raise ValueError("query must not be empty")
        if not items or any(not document.strip() for document in items):
            raise ValueError("documents must contain at least one non-empty string")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        payload = self.client.post_json(
            self.endpoint.base_url,
            "rerank",
            {
                "model": self.endpoint.model,
                "query": query,
                "documents": items,
                "top_n": min(top_n, len(items)),
            },
        )
        try:
            rows = payload["results"]
            if not isinstance(rows, list):
                raise TypeError
            indices = [row["index"] for row in rows]
            if any(type(index) is not int for index in indices):
                raise TypeError
            if len(indices) != len(set(indices)):
                raise ValueError
            if any(index < 0 or index >= len(items) for index in indices):
                raise IndexError
            return [
                RankedDocument(
                    index=index,
                    score=float(row["relevance_score"]),
                    text=items[index],
                )
                for row, index in zip(rows, indices, strict=True)
            ]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise FuriosaApiError("Reranker response has an invalid results field") from exc
