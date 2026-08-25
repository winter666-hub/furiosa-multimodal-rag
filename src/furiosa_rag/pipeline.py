"""End-to-end Text RAG orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.chunking import PageChunker
from furiosa_rag.document import PdfTextExtractor
from furiosa_rag.embedding import EmbeddingBackend
from furiosa_rag.llm import LlmBackend
from furiosa_rag.models import RagAnswer, RetrievedChunk
from furiosa_rag.reranker import RerankerBackend
from furiosa_rag.retrieval import CosineRetriever


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class RagConfig:
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 10
    top_n: int = 3
    answer_max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if self.top_n > self.top_k:
            raise ValueError("top_n must be less than or equal to top_k")


class TextRagPipeline:
    def __init__(
        self,
        embedding: EmbeddingBackend,
        reranker: RerankerBackend,
        llm: LlmBackend,
        *,
        config: RagConfig | None = None,
        extractor: PdfTextExtractor | None = None,
        retriever: CosineRetriever | None = None,
        cache: DocumentEmbeddingCache | None = None,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker
        self.llm = llm
        self.config = config or RagConfig()
        self.extractor = extractor or PdfTextExtractor()
        self.retriever = retriever or CosineRetriever()
        self.cache = cache or DocumentEmbeddingCache()

    @staticmethod
    def _measure(operation: Callable[[], ResultT]) -> tuple[ResultT, float]:
        started = time.perf_counter()
        result = operation()
        return result, (time.perf_counter() - started) * 1000

    def _embedding_model_id(self) -> str:
        endpoint = getattr(self.embedding, "endpoint", None)
        return str(getattr(endpoint, "model", type(self.embedding).__qualname__))

    def answer(
        self, pdf_path: str | Path, question: str, *, rebuild_cache: bool = False
    ) -> RagAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")
        total_started = time.perf_counter()
        latency: dict[str, float | bool] = {}
        cache_key, latency["cache_key"] = self._measure(
            lambda: self.cache.cache_key(
                pdf_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                embedding_model=self._embedding_model_id(),
            )
        )
        cached = None
        if not rebuild_cache:
            cached, latency["cache_load"] = self._measure(lambda: self.cache.load(cache_key))
        else:
            latency["cache_load"] = 0.0
        latency["cache_hit"] = cached is not None

        if cached is not None:
            chunks = cached.chunks
            chunk_vectors = cached.embeddings
            latency["text_extraction"] = 0.0
            latency["chunking"] = 0.0
            latency["document_embedding"] = 0.0
            latency["cache_save"] = 0.0
        else:
            pages, latency["text_extraction"] = self._measure(
                lambda: self.extractor.extract(pdf_path)
            )
            chunker = PageChunker(self.config.chunk_size, self.config.chunk_overlap)
            chunks, latency["chunking"] = self._measure(lambda: chunker.split(pages))
            if not chunks:
                raise ValueError("No chunks were generated from the PDF")
            chunk_vectors, latency["document_embedding"] = self._measure(
                lambda: self.embedding.embed([chunk.text for chunk in chunks])
            )
            _, latency["cache_save"] = self._measure(
                lambda: self.cache.save(
                    cache_key,
                    chunks,
                    chunk_vectors,
                    metadata={
                        "pdf_path": str(Path(pdf_path)),
                        "chunk_size": self.config.chunk_size,
                        "chunk_overlap": self.config.chunk_overlap,
                        "embedding_model": self._embedding_model_id(),
                    },
                )
            )
        query_vectors, latency["query_embedding"] = self._measure(
            lambda: self.embedding.embed([question])
        )
        retrieved, latency["retrieval"] = self._measure(
            lambda: self.retriever.search(
                query_vectors[0], chunks, chunk_vectors, top_k=self.config.top_k
            )
        )
        ranked, latency["reranking"] = self._measure(
            lambda: self.reranker.rerank(
                question,
                [item.chunk.text for item in retrieved],
                top_n=self.config.top_n,
            )
        )
        sources = tuple(
            RetrievedChunk(
                chunk=retrieved[item.index].chunk,
                retrieval_score=retrieved[item.index].retrieval_score,
                rerank_score=item.score,
            )
            for item in ranked
        )
        context = "\n\n".join(
            f"[Source: page {item.chunk.page_number}, chunk {item.chunk.chunk_id}]\n"
            f"{item.chunk.text}"
            for item in sources
        )
        prompt = (
            "Answer the question using only the provided document context. "
            "If the context is insufficient, say so. Cite sources as [page N, chunk ID].\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )
        answer, latency["answer_generation"] = self._measure(
            lambda: self.llm.generate(prompt, max_tokens=self.config.answer_max_tokens)
        )
        latency["total"] = (time.perf_counter() - total_started) * 1000
        return RagAnswer(
            answer=answer,
            sources=sources,
            latency_ms=latency,
            cache_path=str(self.cache.path_for(cache_key)),
        )
