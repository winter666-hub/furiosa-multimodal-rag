from __future__ import annotations

from unittest.mock import Mock

import pytest

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.cli.benchmark_router import evaluate, summarize
from furiosa_rag.config import ModelEndpoint
from furiosa_rag.router import LLMQueryRouter, LLMRouterError, QueryRoute


def _router_with_output(output: str) -> tuple[LLMQueryRouter, Mock]:
    client = Mock()
    client.post_json.return_value = {
        "choices": [{"message": {"content": output}}]
    }
    endpoint = ModelEndpoint("llm", "http://llm.example/v1", "router-model")
    return LLMQueryRouter(endpoint, client), client


def test_llm_router_parses_text_only_and_sends_only_question() -> None:
    router, client = _router_with_output("TEXT_ONLY")

    decision = router.route("Why is layer normalization useful?")

    assert decision.route is QueryRoute.TEXT_ONLY
    assert decision.reason == "LLM classified question as TEXT_ONLY"
    assert client.post_json.call_args.args[:2] == (
        "http://llm.example/v1",
        "chat/completions",
    )
    payload = client.post_json.call_args.args[2]
    assert payload["model"] == "router-model"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 4
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][1] == {
        "role": "user",
        "content": "Why is layer normalization useful?",
    }


def test_llm_router_parses_visual_required() -> None:
    router, _ = _router_with_output("\nVISUAL_REQUIRED\n")
    assert router.route("Which curve is highest?").route is QueryRoute.VISUAL_REQUIRED


@pytest.mark.parametrize(
    "output",
    ("visual", "VISUAL_REQUIRED because it mentions a curve", "", "TEXT_ONLY."),
)
def test_llm_router_rejects_malformed_output(output: str) -> None:
    router, _ = _router_with_output(output)
    with pytest.raises(LLMRouterError, match="invalid LLM router output"):
        router.route("question")


def test_llm_router_propagates_timeout_or_api_error() -> None:
    router, client = _router_with_output("TEXT_ONLY")
    error = FuriosaApiError("POST endpoint failed: timed out")
    client.post_json.side_effect = error

    with pytest.raises(FuriosaApiError) as raised:
        router.route("question")
    assert raised.value is error


def test_llm_router_benchmark_integration() -> None:
    router, client = _router_with_output("TEXT_ONLY")
    client.post_json.side_effect = [
        {"choices": [{"message": {"content": "TEXT_ONLY"}}]},
        {"choices": [{"message": {"content": "VISUAL_REQUIRED"}}]},
    ]
    rows = [
        {
            "id": "M01",
            "question": "Why use softmax?",
            "expected_route": "TEXT_ONLY",
            "category": "text",
        },
        {
            "id": "M02",
            "question": "Which plotted line is higher?",
            "expected_route": "VISUAL_REQUIRED",
            "category": "implicit_visual",
        },
    ]

    results = evaluate(rows, router)
    summary = summarize(results)

    assert [row["actual_route"] for row in results] == [
        "TEXT_ONLY",
        "VISUAL_REQUIRED",
    ]
    assert all(row["correct"] for row in results)
    assert all(float(row["routing_latency_ms"]) >= 0 for row in results)
    assert summary["accuracy"] == 1.0
    assert summary["false_positives"] == 0
    assert summary["false_negatives"] == 0
    assert float(summary["average_routing_latency_ms"]) >= 0
