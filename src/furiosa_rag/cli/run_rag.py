"""Run the Text RAG pipeline for one PDF and question."""

from __future__ import annotations

import argparse

from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.pipeline import RagConfig, TextRagPipeline
from furiosa_rag.reranker import FuriosaReranker


def _endpoint(settings: Settings, name: str):
    return next(endpoint for endpoint in settings.endpoints if endpoint.name == name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Furiosa Text RAG on a PDF")
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Ignore and rebuild document embeddings"
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    pipeline = TextRagPipeline(
        FuriosaEmbedding(_endpoint(settings, "embedding"), client),
        FuriosaReranker(_endpoint(settings, "reranker"), client),
        FuriosaLlm(_endpoint(settings, "llm"), client),
        config=RagConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            top_n=args.top_n,
        ),
    )
    result = pipeline.answer(args.pdf, args.question, rebuild_cache=args.rebuild_cache)
    print("\nANSWER\n" + result.answer)
    print("\nSOURCES")
    for source in result.sources:
        print(
            f"page={source.chunk.page_number} chunk={source.chunk.chunk_id} "
            f"retrieval={source.retrieval_score:.4f} rerank={source.rerank_score:.4f}"
        )
    print("\nLATENCY_MS")
    for stage, latency_ms in result.latency_ms.items():
        if isinstance(latency_ms, bool):
            print(f"{stage}={str(latency_ms).lower()}")
        else:
            print(f"{stage}={latency_ms:.1f}")
    print(f"cache_path={result.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
