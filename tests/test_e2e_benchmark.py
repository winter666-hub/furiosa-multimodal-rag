from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from furiosa_rag.cli.benchmark_e2e import _clients
from furiosa_rag.config import Settings
from furiosa_rag.e2e_benchmark import (
    export_e2e_csv,
    load_e2e_jsonl,
    run_e2e,
    summarize_e2e,
)
from furiosa_rag.models import (
    Chunk,
    MultimodalRagAnswer,
    RagAnswer,
    RetrievedChunk,
    VisionUsage,
)
from furiosa_rag.router import QueryRoute, RoutingDecision


def _source() -> RetrievedChunk:
    return RetrievedChunk(Chunk("page-3-chunk-1", 3, "evidence"), 0.8, 0.9)


def _text_result() -> RagAnswer:
    return RagAnswer(
        "text answer",
        (_source(),),
        {
            "cache_hit": True,
            "query_embedding": 2.0,
            "reranking": 3.0,
            "answer_generation": 5.0,
            "total": 10.0,
        },
    )


def _visual_result() -> MultimodalRagAnswer:
    return MultimodalRagAnswer(
        "visual answer",
        (_source(),),
        VisionUsage(3, True, "fake-vision"),
        {
            "cache_hit": False,
            "query_embedding": 2.0,
            "reranking": 3.0,
            "page_rendering": 4.0,
            "vision_analysis": 8.0,
            "answer_generation": 5.0,
            "total": 22.0,
        },
    )


def _rows() -> list[dict[str, str]]:
    return [
        {"id": "Q1", "question": "text q", "category": "text"},
        {"id": "Q2", "question": "visual q", "category": "implicit_visual"},
    ]


def test_always_vision_uses_multimodal_for_every_question() -> None:
    text_pipeline = Mock()
    multimodal_pipeline = Mock()
    multimodal_pipeline.answer_multimodal.return_value = _visual_result()

    results = run_e2e(
        _rows(), "paper.pdf", strategy="always_vision",
        text_pipeline=text_pipeline, multimodal_pipeline=multimodal_pipeline,
    )

    assert multimodal_pipeline.answer_multimodal.call_count == 2
    text_pipeline.answer.assert_not_called()
    assert all(row["vision_used"] for row in results)


@pytest.mark.parametrize("strategy", ("llm", "adaptive"))
def test_routed_strategy_uses_text_pipeline_for_text_route(strategy: str) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "router text")
    text_pipeline = Mock()
    text_pipeline.answer.return_value = _text_result()
    multimodal_pipeline = Mock()

    result = run_e2e(
        _rows()[:1], "paper.pdf", strategy=strategy, router=router,
        text_pipeline=text_pipeline, multimodal_pipeline=multimodal_pipeline,
    )[0]

    text_pipeline.answer.assert_called_once()
    multimodal_pipeline.answer_multimodal.assert_not_called()
    assert result["route"] == "TEXT_ONLY"
    assert result["vision_used"] is False


@pytest.mark.parametrize("strategy", ("llm", "adaptive"))
def test_routed_strategy_uses_multimodal_for_visual_route(strategy: str) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(
        QueryRoute.VISUAL_REQUIRED, "router visual"
    )
    text_pipeline = Mock()
    multimodal_pipeline = Mock()
    multimodal_pipeline.answer_multimodal.return_value = _visual_result()

    result = run_e2e(
        _rows()[1:], "paper.pdf", strategy=strategy, router=router,
        text_pipeline=text_pipeline, multimodal_pipeline=multimodal_pipeline,
    )[0]

    multimodal_pipeline.answer_multimodal.assert_called_once()
    text_pipeline.answer.assert_not_called()
    assert result["route"] == "VISUAL_REQUIRED"
    assert result["selected_page"] == 3


def test_summary_calculates_vision_calls_and_latencies() -> None:
    text_pipeline = Mock()
    text_pipeline.answer.return_value = _text_result()
    multimodal_pipeline = Mock()
    multimodal_pipeline.answer_multimodal.return_value = _visual_result()
    router = Mock()
    router.route.side_effect = [
        RoutingDecision(QueryRoute.TEXT_ONLY, "text"),
        RoutingDecision(QueryRoute.VISUAL_REQUIRED, "visual"),
    ]
    results = run_e2e(
        _rows(), "paper.pdf", strategy="llm", router=router,
        text_pipeline=text_pipeline, multimodal_pipeline=multimodal_pipeline,
    )

    summary = summarize_e2e(results)

    assert summary["total_questions"] == 2
    assert summary["vision_calls"] == 1
    assert summary["vision_call_rate"] == 0.5
    assert summary["average_vision_latency_ms"] == 8.0
    assert float(summary["average_total_latency_ms"]) >= 0


def test_csv_export_contains_required_fields(tmp_path: Path) -> None:
    pipeline = Mock()
    pipeline.answer_multimodal.return_value = _visual_result()
    results = run_e2e(
        _rows()[:1], "paper.pdf", strategy="always_vision",
        text_pipeline=Mock(), multimodal_pipeline=pipeline,
    )
    output = tmp_path / "nested" / "e2e.csv"

    export_e2e_csv(results, output)

    with output.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["strategy"] == "always_vision"
    assert row["vision_used"] == "True"
    assert row["answer"] == "visual answer"
    assert row["error"] == ""
    assert row["paper"] == ""
    assert row["expected_route"] == ""
    assert row["expected_page"] == ""
    assert row["expected_visual_evidence"] == ""


def test_csv_export_contains_optional_metadata(tmp_path: Path) -> None:
    pipeline = Mock()
    pipeline.answer_multimodal.return_value = _visual_result()
    item = {
        "id": "Q1",
        "question": "visual q",
        "category": "implicit_visual",
        "paper": "bert",
        "expected_route": "VISUAL_REQUIRED",
        "expected_page": 5,
        "expected_visual_evidence": "BERT Figure 2",
    }
    results = run_e2e(
        [item], "paper.pdf", strategy="always_vision",
        text_pipeline=Mock(), multimodal_pipeline=pipeline,
    )
    output = tmp_path / "e2e.csv"

    export_e2e_csv(results, output)

    with output.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    assert row["paper"] == "bert"
    assert row["expected_route"] == "VISUAL_REQUIRED"
    assert row["expected_page"] == "5"
    assert row["expected_visual_evidence"] == "BERT Figure 2"
    assert row["selected_page"] == "3"


def test_pipeline_error_is_recorded_and_next_question_continues() -> None:
    text_pipeline = Mock()
    text_pipeline.answer.side_effect = [RuntimeError("pipeline failed"), _text_result()]
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text")

    results = run_e2e(
        _rows(), "paper.pdf", strategy="llm", router=router,
        text_pipeline=text_pipeline, multimodal_pipeline=Mock(),
    )

    assert results[0]["error"] == "RuntimeError: pipeline failed"
    assert results[1]["answer"] == "text answer"
    assert summarize_e2e(results)["failure_count"] == 1


def test_small_e2e_dataset_has_five_questions_per_category() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "e2e_eval_small.jsonl"
    rows = load_e2e_jsonl(path)
    assert len(rows) == 15
    assert len({row["id"] for row in rows}) == 15
    assert {
        category: sum(row["category"] == category for row in rows)
        for category in ("text", "explicit_visual", "implicit_visual")
    } == {"text": 5, "explicit_visual": 5, "implicit_visual": 5}


def test_load_e2e_jsonl_keeps_legacy_row_unchanged(tmp_path: Path) -> None:
    item = {"id": "Q1", "question": "text q", "category": "text"}
    dataset = tmp_path / "legacy.jsonl"
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")

    assert load_e2e_jsonl(dataset) == [item]


def test_load_e2e_jsonl_preserves_optional_metadata(tmp_path: Path) -> None:
    item = {
        "id": "Q1",
        "question": "visual q",
        "category": "implicit_visual",
        "paper": "bert",
        "expected_route": "VISUAL_REQUIRED",
        "expected_page": 5,
        "expected_visual_evidence": "BERT Figure 2",
    }
    dataset = tmp_path / "ack.jsonl"
    dataset.write_text(json.dumps(item) + "\n", encoding="utf-8")

    assert load_e2e_jsonl(dataset) == [item]


def test_e2e_clients_use_separate_general_and_vision_timeouts() -> None:
    settings = Settings(
        api_key="test-key",
        request_timeout=10,
        vision_request_timeout=60,
        vision_max_tokens=256,
        endpoints=(),
    )

    client, vision_client = _clients(settings)

    assert client.timeout == 10
    assert vision_client.timeout == 60
