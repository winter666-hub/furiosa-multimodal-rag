"""Benchmark Text RAG latency across multiple top-k values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.pipeline import RagConfig, TextRagPipeline
from furiosa_rag.reranker import FuriosaReranker


def _endpoint(settings: Settings, name: str) -> ModelEndpoint:
    return next(endpoint for endpoint in settings.endpoints if endpoint.name == name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Furiosa Text RAG top-k settings")
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--output", default="data/benchmarks/top_k_benchmark.csv")
    args = parser.parse_args()

    if any(args.top_n > top_k for top_k in args.top_k):
        parser.error("top_n must be less than or equal to every top_k value")

    settings = Settings.from_env()
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    embedding = FuriosaEmbedding(_endpoint(settings, "embedding"), client)
    reranker = FuriosaReranker(_endpoint(settings, "reranker"), client)
    llm = FuriosaLlm(_endpoint(settings, "llm"), client)
    cache = DocumentEmbeddingCache()
    rows: list[dict[str, object]] = []

    for top_k in args.top_k:
        pipeline = TextRagPipeline(
            embedding,
            reranker,
            llm,
            config=RagConfig(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=top_k,
                top_n=args.top_n,
            ),
            cache=cache,
        )
        result = pipeline.answer(args.pdf, args.question)
        top_source = result.sources[0]
        row = {
            "top_k": top_k,
            "top_n": args.top_n,
            "cache_hit": result.latency_ms["cache_hit"],
            "retrieval_latency": round(float(result.latency_ms["retrieval"]), 1),
            "reranking_latency": round(float(result.latency_ms["reranking"]), 1),
            "answer_generation_latency": round(
                float(result.latency_ms["answer_generation"]), 1
            ),
            "total_latency": round(float(result.latency_ms["total"]), 1),
            "top_source_page": top_source.chunk.page_number,
            "top_source_chunk": top_source.chunk.chunk_id,
            "top_rerank_score": round(float(top_source.rerank_score or 0.0), 4),
        }
        rows.append(row)
        print(" ".join(f"{key}={value}" for key, value in row.items()))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

