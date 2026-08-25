from pathlib import Path

import pytest

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.chunking import PageChunker
from furiosa_rag.models import Chunk, PageText
from furiosa_rag.pipeline import RagConfig, TextRagPipeline
from furiosa_rag.reranker import RankedDocument
from furiosa_rag.retrieval import CosineRetriever


def test_chunker_preserves_page_number_and_overlap() -> None:
    pages = [PageText(2, "one two three four five")]
    chunks = PageChunker(chunk_size=3, chunk_overlap=1).split(pages)
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert chunks[0].chunk_id == "page-2-chunk-1"


def test_cosine_retrieval_returns_most_similar_first() -> None:
    chunks = [Chunk("a", 1, "alpha"), Chunk("b", 2, "beta")]
    results = CosineRetriever().search([1.0, 0.0], chunks, [[0.0, 1.0], [1.0, 0.0]])
    assert results[0].chunk.chunk_id == "b"
    assert results[0].retrieval_score == 1.0


def test_rag_config_rejects_top_n_greater_than_top_k() -> None:
    with pytest.raises(ValueError, match="top_n"):
        RagConfig(top_k=2, top_n=3)


class FakeExtractor:
    def extract(self, pdf_path: str | Path) -> list[PageText]:
        return [PageText(1, "alpha beta gamma")]


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeReranker:
    def rerank(
        self, query: str, documents: list[str], *, top_n: int = 3
    ) -> list[RankedDocument]:
        return [RankedDocument(index=0, score=0.9, text=documents[0])]


class FakeLlm:
    def generate(self, prompt: str, *, max_tokens: int = 64) -> str:
        return "answer"


def test_pipeline_reuses_cached_document_embeddings(tmp_path: Path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"test-pdf-identity")
    embedding = FakeEmbedding()
    pipeline = TextRagPipeline(
        embedding,
        FakeReranker(),
        FakeLlm(),
        config=RagConfig(chunk_size=3, chunk_overlap=0, top_k=1, top_n=1),
        extractor=FakeExtractor(),
        cache=DocumentEmbeddingCache(tmp_path / "cache"),
    )

    first = pipeline.answer(pdf_path, "first question")
    second = pipeline.answer(pdf_path, "second question")

    assert first.latency_ms["cache_hit"] is False
    assert second.latency_ms["cache_hit"] is True
    assert second.latency_ms["document_embedding"] == 0.0
    assert len(embedding.calls) == 3
    assert embedding.calls[0] == ["alpha beta gamma"]
