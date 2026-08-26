"""Strategy-level end-to-end RAG benchmark orchestration."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from furiosa_rag.models import MultimodalRagAnswer, RagAnswer
from furiosa_rag.router import QueryRoute, QueryRouter

STRATEGIES = ("always_vision", "llm", "adaptive")
REQUIRED_FIELDS = {"id", "question", "category"}
CSV_FIELDS = [
    "id", "question", "category", "strategy", "route", "routing_reason",
    "routing_latency_ms", "vision_used", "selected_page",
    "query_embedding_latency_ms", "reranking_latency_ms",
    "page_rendering_latency_ms", "vision_analysis_latency_ms",
    "answer_generation_latency_ms", "total_latency_ms", "cache_state",
    "answer", "sources", "error",
]


class TextPipeline(Protocol):
    def answer(self, pdf_path: str | Path, question: str) -> RagAnswer: ...


class MultimodalPipeline(Protocol):
    def answer_multimodal(
        self, pdf_path: str | Path, question: str
    ) -> MultimodalRagAnswer: ...


def load_e2e_jsonl(path: str | Path) -> list[dict[str, str]]:
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
            if row["category"] not in {"text", "explicit_visual", "implicit_visual"}:
                raise ValueError(f"invalid category on line {line_number}")
            rows.append({field: str(row[field]) for field in REQUIRED_FIELDS})
    if not rows:
        raise ValueError("E2E benchmark dataset is empty")
    return rows


def _sources_json(result: RagAnswer | MultimodalRagAnswer) -> str:
    return json.dumps(
        [
            {
                "page": source.chunk.page_number,
                "chunk": source.chunk.chunk_id,
                "retrieval_score": source.retrieval_score,
                "rerank_score": source.rerank_score,
            }
            for source in result.sources
        ],
        ensure_ascii=False,
    )


def _latency(result: RagAnswer | MultimodalRagAnswer, key: str) -> float:
    value = result.latency_ms.get(key, 0.0)
    return 0.0 if isinstance(value, bool) else float(value)


def run_e2e(
    rows: list[dict[str, str]],
    pdf_path: str | Path,
    *,
    strategy: str,
    text_pipeline: TextPipeline,
    multimodal_pipeline: MultimodalPipeline,
    router: QueryRouter | None = None,
) -> list[dict[str, Any]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    if strategy != "always_vision" and router is None:
        raise ValueError(f"strategy {strategy} requires a router")

    results: list[dict[str, Any]] = []
    for item in rows:
        total_started = time.perf_counter_ns()
        row: dict[str, Any] = {
            **item,
            "strategy": strategy,
            "route": "",
            "routing_reason": "",
            "routing_latency_ms": 0.0,
            "vision_used": False,
            "selected_page": None,
            "query_embedding_latency_ms": 0.0,
            "reranking_latency_ms": 0.0,
            "page_rendering_latency_ms": 0.0,
            "vision_analysis_latency_ms": 0.0,
            "answer_generation_latency_ms": 0.0,
            "cache_state": "unknown",
            "answer": "",
            "sources": "[]",
            "error": "",
        }
        try:
            if strategy == "always_vision":
                route = QueryRoute.VISUAL_REQUIRED
                row["routing_reason"] = "always_vision strategy"
            else:
                routing_started = time.perf_counter_ns()
                decision = router.route(item["question"])  # type: ignore[union-attr]
                row["routing_latency_ms"] = (
                    time.perf_counter_ns() - routing_started
                ) / 1_000_000
                route = decision.route
                row["routing_reason"] = decision.reason
            row["route"] = route.value

            if route is QueryRoute.VISUAL_REQUIRED:
                result = multimodal_pipeline.answer_multimodal(pdf_path, item["question"])
                row["vision_used"] = result.vision.used
                row["selected_page"] = result.vision.selected_page
                if result.vision.error:
                    row["error"] = result.vision.error
            else:
                result = text_pipeline.answer(pdf_path, item["question"])

            row["query_embedding_latency_ms"] = _latency(result, "query_embedding")
            row["reranking_latency_ms"] = _latency(result, "reranking")
            row["page_rendering_latency_ms"] = _latency(result, "page_rendering")
            row["vision_analysis_latency_ms"] = _latency(result, "vision_analysis")
            row["answer_generation_latency_ms"] = _latency(result, "answer_generation")
            cache_hit = result.latency_ms.get("cache_hit")
            row["cache_state"] = (
                "warm" if cache_hit is True else "cold" if cache_hit is False else "unknown"
            )
            row["answer"] = result.answer
            row["sources"] = _sources_json(result)
        except Exception as exc:  # keep the benchmark running after a per-question failure
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_latency_ms"] = (time.perf_counter_ns() - total_started) / 1_000_000
        results.append(row)
    return results


def summarize_e2e(results: list[dict[str, Any]]) -> dict[str, int | float]:
    total = len(results)
    failures = sum(bool(row["error"]) for row in results)
    vision_calls = sum(bool(row["vision_used"]) for row in results)
    category_totals: Counter[str] = Counter(str(row["category"]) for row in results)

    def average(field: str, subset: list[dict[str, Any]] = results) -> float:
        return sum(float(row[field]) for row in subset) / len(subset) if subset else 0.0

    summary: dict[str, int | float] = {
        "total_questions": total,
        "success_count": total - failures,
        "failure_count": failures,
        "vision_calls": vision_calls,
        "vision_call_rate": vision_calls / total if total else 0.0,
        "average_routing_latency_ms": average("routing_latency_ms"),
        "average_vision_latency_ms": average(
            "vision_analysis_latency_ms",
            [row for row in results if bool(row["vision_used"])],
        ),
        "average_total_latency_ms": average("total_latency_ms"),
    }
    for category in ("text", "explicit_visual", "implicit_visual"):
        subset = [row for row in results if row["category"] == category]
        summary[f"average_total_latency_{category}_ms"] = (
            average("total_latency_ms", subset) if category_totals[category] else 0.0
        )
    return summary


def export_e2e_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
