"""Benchmark routing strategies through the real text and multimodal pipelines."""

from __future__ import annotations

import argparse
import json

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.e2e_benchmark import (
    STRATEGIES,
    export_e2e_csv,
    load_e2e_jsonl,
    run_e2e,
    summarize_e2e,
)
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.pipeline import MultimodalRagPipeline, RagConfig, TextRagPipeline
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.router import AdaptiveQueryRouter, LLMQueryRouter, QueryRouter
from furiosa_rag.vision import FuriosaVision


def _endpoint(settings: Settings, name: str) -> ModelEndpoint:
    return next(endpoint for endpoint in settings.endpoints if endpoint.name == name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark E2E RAG routing strategies")
    parser.add_argument("pdf")
    parser.add_argument("dataset")
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--vision-dpi", type=float, default=144.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = Settings.from_env()
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    config = RagConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        top_n=args.top_n,
        vision_max_tokens=settings.vision_max_tokens,
        vision_dpi=args.vision_dpi,
    )
    embedding = FuriosaEmbedding(_endpoint(settings, "embedding"), client)
    reranker = FuriosaReranker(_endpoint(settings, "reranker"), client)
    llm = FuriosaLlm(_endpoint(settings, "llm"), client)
    cache = DocumentEmbeddingCache()
    text_pipeline = TextRagPipeline(
        embedding, reranker, llm, config=config, cache=cache
    )
    multimodal_pipeline = MultimodalRagPipeline(
        embedding,
        reranker,
        llm,
        vision=FuriosaVision(_endpoint(settings, "vision"), client),
        config=config,
        cache=cache,
    )

    router: QueryRouter | None = None
    if args.strategy != "always_vision":
        llm_router = LLMQueryRouter(_endpoint(settings, "llm"), client)
        router = llm_router if args.strategy == "llm" else AdaptiveQueryRouter(llm_router)

    results = run_e2e(
        load_e2e_jsonl(args.dataset),
        args.pdf,
        strategy=args.strategy,
        text_pipeline=text_pipeline,
        multimodal_pipeline=multimodal_pipeline,
        router=router,
    )
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    summary = summarize_e2e(results)
    print("\nSummary")
    for key, value in summary.items():
        if key.endswith("rate"):
            print(f"{key}: {float(value):.2%}")
        else:
            print(f"{key}: {value}")
    export_e2e_csv(results, args.output)
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
