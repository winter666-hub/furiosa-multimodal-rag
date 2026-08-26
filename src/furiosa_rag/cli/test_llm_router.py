"""Smoke-test LLMQueryRouter against the configured hosted LLM endpoint."""

from __future__ import annotations

import argparse
import time

from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import Settings
from furiosa_rag.router import LLMQueryRouter


def main() -> int:
    parser = argparse.ArgumentParser(description="Route one question with LLMQueryRouter")
    parser.add_argument("question", help="user question to classify")
    args = parser.parse_args()

    settings = Settings.from_env()
    endpoint = next(item for item in settings.endpoints if item.name == "llm")
    router = LLMQueryRouter(
        endpoint,
        FuriosaClient(api_key=settings.api_key, timeout=settings.request_timeout),
    )

    started = time.perf_counter_ns()
    decision = router.route(args.question)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    print(f"question={args.question}")
    print(f"route={decision.route.value}")
    print(f"reason={decision.reason}")
    print(f"routing_latency_ms={latency_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
