"""Run the multimodal RAG pipeline for one PDF and question."""

from __future__ import annotations

import argparse

from furiosa_rag.cli.run_rag import _endpoint
from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.pipeline import MultimodalRagPipeline, RagConfig
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.vision import FuriosaVision


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Furiosa Multimodal RAG on a PDF")
    parser.add_argument("pdf")
    parser.add_argument("question")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--vision-dpi", type=float, default=144.0)
    parser.add_argument(
        "--vision-max-tokens",
        type=int,
        default=None,
        help="Override FURIOSA_VISION_MAX_TOKENS (default: 256)",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    settings = Settings.from_env()
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    pipeline = MultimodalRagPipeline(
        FuriosaEmbedding(_endpoint(settings, "embedding"), client),
        FuriosaReranker(_endpoint(settings, "reranker"), client),
        FuriosaLlm(_endpoint(settings, "llm"), client),
        vision=FuriosaVision(_endpoint(settings, "vision"), client),
        config=RagConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            top_n=args.top_n,
            vision_dpi=args.vision_dpi,
            vision_max_tokens=(
                args.vision_max_tokens
                if args.vision_max_tokens is not None
                else settings.vision_max_tokens
            ),
        ),
    )
    result = pipeline.answer_multimodal(args.pdf, args.question, rebuild_cache=args.rebuild_cache)
    print("\nANSWER\n" + result.answer)
    print("\nSOURCES")
    for source in result.sources:
        print(
            f"page={source.chunk.page_number} chunk={source.chunk.chunk_id} "
            f"retrieval={source.retrieval_score:.4f} rerank={source.rerank_score:.4f}"
        )
    print("\nVISION")
    print(f"selected_page={result.vision.selected_page}")
    print(f"used={str(result.vision.used).lower()}")
    print(f"model={result.vision.model}")
    if result.vision.error:
        print(f"error={result.vision.error}")
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
