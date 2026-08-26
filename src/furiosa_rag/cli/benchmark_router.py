"""Benchmark the deterministic rule-based query router without external APIs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import Settings
from furiosa_rag.router import (
    AdaptiveQueryRouter,
    EnhancedRuleBasedQueryRouter,
    LLMQueryRouter,
    QueryRoute,
    QueryRouter,
    RuleBasedQueryRouter,
)

REQUIRED_FIELDS = {"id", "question", "expected_route", "category"}
CSV_FIELDS = [
    "id",
    "question",
    "category",
    "expected_route",
    "actual_route",
    "correct",
    "reason",
    "routing_latency_ms",
    "used_llm_router",
]


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"line {line_number} is missing fields: {', '.join(sorted(missing))}"
                )
            if row["expected_route"] not in {route.value for route in QueryRoute}:
                raise ValueError(f"invalid expected_route on line {line_number}")
            rows.append({field: str(row[field]) for field in REQUIRED_FIELDS})
    if not rows:
        raise ValueError("benchmark dataset is empty")
    return rows


def evaluate(
    rows: list[dict[str, str]], router: QueryRouter | None = None
) -> list[dict[str, Any]]:
    active_router = router or RuleBasedQueryRouter()
    results: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter_ns()
        decision = active_router.route(row["question"])
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        results.append(
            {
                **row,
                "actual_route": decision.route.value,
                "correct": decision.route.value == row["expected_route"],
                "reason": decision.reason,
                "routing_latency_ms": latency_ms,
                "used_llm_router": (
                    decision.used_llm_router or isinstance(active_router, LLMQueryRouter)
                ),
            }
        )
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, int | float]:
    total = len(results)
    correct = sum(bool(row["correct"]) for row in results)
    category_totals = Counter(str(row["category"]) for row in results)
    category_correct = Counter(
        str(row["category"]) for row in results if bool(row["correct"])
    )

    def category_accuracy(category: str) -> float:
        count = category_totals[category]
        return category_correct[category] / count if count else 0.0

    false_positives = sum(
        row["expected_route"] == QueryRoute.TEXT_ONLY.value
        and row["actual_route"] == QueryRoute.VISUAL_REQUIRED.value
        for row in results
    )
    false_negatives = sum(
        row["expected_route"] == QueryRoute.VISUAL_REQUIRED.value
        and row["actual_route"] == QueryRoute.TEXT_ONLY.value
        for row in results
    )
    vision_calls = sum(
        row["actual_route"] == QueryRoute.VISUAL_REQUIRED.value for row in results
    )
    llm_router_calls = sum(bool(row.get("used_llm_router", False)) for row in results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "text_only_accuracy": category_accuracy("text"),
        "explicit_visual_accuracy": category_accuracy("explicit_visual"),
        "implicit_visual_accuracy": category_accuracy("implicit_visual"),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_vision_call_rate": vision_calls / total if total else 0.0,
        "average_routing_latency_ms": (
            sum(float(row["routing_latency_ms"]) for row in results) / total
            if total
            else 0.0
        ),
        "llm_router_calls": llm_router_calls,
        "llm_router_call_rate": llm_router_calls / total if total else 0.0,
    }


def export_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            exported = dict(row)
            exported["routing_latency_ms"] = f"{float(row['routing_latency_ms']):.6f}"
            writer.writerow(exported)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a deterministic query router")
    parser.add_argument("dataset", help="JSONL router evaluation dataset")
    parser.add_argument(
        "--router", choices=("rule", "enhanced", "llm", "adaptive"), default="rule",
        help="router implementation to benchmark (default: rule)",
    )
    parser.add_argument("--output", help="optional CSV result path")
    args = parser.parse_args()

    if args.router == "rule":
        router: QueryRouter = RuleBasedQueryRouter()
    elif args.router == "enhanced":
        router = EnhancedRuleBasedQueryRouter()
    else:
        settings = Settings.from_env()
        endpoint = next(item for item in settings.endpoints if item.name == "llm")
        llm_router = LLMQueryRouter(
            endpoint,
            FuriosaClient(api_key=settings.api_key, timeout=settings.request_timeout),
        )
        router = llm_router if args.router == "llm" else AdaptiveQueryRouter(llm_router)
    results = evaluate(load_jsonl(args.dataset), router)
    for row in results:
        print(json.dumps(row, ensure_ascii=False))

    summary = summarize(results)
    print("\nSummary")
    for key, value in summary.items():
        if key.endswith("accuracy") or key.endswith("rate"):
            print(f"{key}: {float(value):.2%}")
        else:
            print(f"{key}: {value}")

    if args.output:
        export_csv(results, args.output)
        print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
