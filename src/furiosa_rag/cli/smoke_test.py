"""Send one minimal inference request to each Text RAG model API."""

from __future__ import annotations

import time
from collections.abc import Callable

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.reranker import FuriosaReranker


def _endpoint(settings: Settings, name: str) -> ModelEndpoint:
    return next(endpoint for endpoint in settings.endpoints if endpoint.name == name)


def _run(name: str, operation: Callable[[], object]) -> bool:
    started = time.perf_counter()
    try:
        response_info = operation()
    except (FuriosaApiError, ValueError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"[FAIL] {name:<9} {elapsed_ms:8.1f} ms error={exc}")
        return False
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"[OK]   {name:<9} {elapsed_ms:8.1f} ms response={response_info}")
    return True


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    required = ("llm", "embedding", "reranker")
    endpoints = {name: _endpoint(settings, name) for name in required}
    disabled = [name for name, endpoint in endpoints.items() if not endpoint.enabled]
    if disabled:
        print(f"Missing endpoint URLs: {', '.join(disabled)}")
        return 2

    client = FuriosaClient(settings.api_key, settings.request_timeout)

    def check_embedding() -> str:
        vectors = FuriosaEmbedding(endpoints["embedding"], client).embed(["hello"])
        return f"vectors={len(vectors)}, dimensions={len(vectors[0])}"

    def check_reranker() -> str:
        ranked = FuriosaReranker(endpoints["reranker"], client).rerank(
            "capital of Korea", ["Seoul is the capital of Korea.", "Paris is in France."], top_n=1
        )
        return f"results={len(ranked)}, top_index={ranked[0].index}, score={ranked[0].score:.4f}"

    def check_llm() -> str:
        answer = FuriosaLlm(endpoints["llm"], client).generate(
            "Reply only with the word pong.", max_tokens=32
        )
        return f"content={answer[:80]!r}"

    checks = (
        ("embedding", check_embedding),
        ("reranker", check_reranker),
        ("llm", check_llm),
    )
    results = [_run(name, operation) for name, operation in checks]
    print(f"\nInference smoke test: {sum(results)}/{len(results)} APIs passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
