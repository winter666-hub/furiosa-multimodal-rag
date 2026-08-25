"""Provider-neutral data models for the Text RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str
    sources: tuple[RetrievedChunk, ...]
    latency_ms: dict[str, float | bool]
    cache_path: str | None = None
