import csv
import json
from pathlib import Path

from furiosa_rag.cli.benchmark_router import evaluate, export_csv, load_jsonl, summarize
from furiosa_rag.router import (
    EnhancedRuleBasedQueryRouter,
    QueryRoute,
    RuleBasedQueryRouter,
)


def _result(expected: str, actual: str, category: str) -> dict[str, object]:
    return {
        "id": "test",
        "question": "question",
        "category": category,
        "expected_route": expected,
        "actual_route": actual,
        "correct": expected == actual,
        "reason": "test",
        "routing_latency_ms": 0.1,
    }


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "T01",
                "question": "why?",
                "expected_route": "TEXT_ONLY",
                "category": "text",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_jsonl(path)[0]["id"] == "T01"


def test_summary_calculates_errors_and_category_accuracy() -> None:
    results = [
        _result("TEXT_ONLY", "TEXT_ONLY", "text"),
        _result("TEXT_ONLY", "VISUAL_REQUIRED", "text"),
        _result("VISUAL_REQUIRED", "VISUAL_REQUIRED", "explicit_visual"),
        _result("VISUAL_REQUIRED", "TEXT_ONLY", "implicit_visual"),
    ]
    summary = summarize(results)
    assert summary["total"] == 4
    assert summary["correct"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["text_only_accuracy"] == 0.5
    assert summary["explicit_visual_accuracy"] == 1.0
    assert summary["implicit_visual_accuracy"] == 0.0
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["predicted_vision_call_rate"] == 0.5


def test_csv_export(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "results.csv"
    export_csv([_result("TEXT_ONLY", "TEXT_ONLY", "text")], output)
    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["actual_route"] == "TEXT_ONLY"
    assert rows[0]["correct"] == "True"


def test_enhanced_router_preserves_explicit_behavior() -> None:
    decision = EnhancedRuleBasedQueryRouter().route("Compare Figure 1 and Table 1")
    assert decision.route is QueryRoute.VISUAL_REQUIRED
    assert decision.reason.startswith("matched explicit visual keyword:")


def test_enhanced_router_records_semantic_pattern() -> None:
    decision = EnhancedRuleBasedQueryRouter().route(
        "Encoder와 Decoder의 블록 구성 차이를 비교해줘"
    )
    assert decision.route is QueryRoute.VISUAL_REQUIRED
    assert decision.reason == "matched semantic pattern: structure_comparison"


def test_enhanced_router_requires_semantic_combination() -> None:
    router = EnhancedRuleBasedQueryRouter()
    assert router.route("구조를 설명해줘").route is QueryRoute.TEXT_ONLY
    assert router.route("위치는 어디인가?").route is QueryRoute.TEXT_ONLY
    assert router.route("positional encoding이 필요한 이유는?").route is QueryRoute.TEXT_ONLY


def test_benchmark_dataset_reproduces_rule_baseline_and_improves_implicit() -> None:
    dataset = Path(__file__).parents[1] / "benchmarks" / "router_eval.jsonl"
    rows = load_jsonl(dataset)
    rule_summary = summarize(evaluate(rows, RuleBasedQueryRouter()))
    enhanced_summary = summarize(evaluate(rows, EnhancedRuleBasedQueryRouter()))

    assert rule_summary["accuracy"] == 20 / 30
    assert rule_summary["false_positives"] == 0
    assert rule_summary["false_negatives"] == 10
    assert enhanced_summary["text_only_accuracy"] == 1.0
    assert enhanced_summary["explicit_visual_accuracy"] == 1.0
    assert enhanced_summary["implicit_visual_accuracy"] == 1.0
    assert enhanced_summary["false_positives"] == 0


def test_holdout_dataset_shape_categories_ids_and_routes() -> None:
    dataset = Path(__file__).parents[1] / "benchmarks" / "router_holdout_eval.jsonl"
    rows = load_jsonl(dataset)

    assert len(rows) == 60
    assert len({row["id"] for row in rows}) == 60
    assert {
        category: sum(row["category"] == category for row in rows)
        for category in ("text", "explicit_visual", "implicit_visual")
    } == {"text": 20, "explicit_visual": 20, "implicit_visual": 20}
    assert {row["expected_route"] for row in rows} <= {
        route.value for route in QueryRoute
    }
    assert all(
        row["expected_route"]
        == ("TEXT_ONLY" if row["category"] == "text" else "VISUAL_REQUIRED")
        for row in rows
    )


def test_rule_router_existing_explicit_keyword_behavior() -> None:
    router = RuleBasedQueryRouter()
    assert router.route("Figure 1의 구조를 설명해줘").route is QueryRoute.VISUAL_REQUIRED
    assert router.route("Table 1의 complexity를 비교해줘").route is QueryRoute.VISUAL_REQUIRED
    assert router.route("그림에서 연결 관계를 설명해줘").route is QueryRoute.VISUAL_REQUIRED
    assert router.route("왜 multi-head attention을 사용하는가?").route is QueryRoute.TEXT_ONLY
