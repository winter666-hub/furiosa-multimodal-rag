from __future__ import annotations

from unittest.mock import Mock

import pytest

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.cli.benchmark_router import evaluate, summarize
from furiosa_rag.router import AdaptiveQueryRouter, QueryRoute, RoutingDecision


@pytest.mark.parametrize(
    "question",
    ("Figure 3에서 경로를 찾아줘", "Table 2의 값을 비교해줘"),
)
def test_adaptive_explicit_shortcut_does_not_call_llm(question: str) -> None:
    llm_router = Mock()
    router = AdaptiveQueryRouter(llm_router)

    decision = router.route(question)

    assert decision.route is QueryRoute.VISUAL_REQUIRED
    assert decision.reason.startswith("adaptive explicit visual shortcut: ")
    assert decision.used_llm_router is False
    llm_router.route.assert_not_called()


def test_adaptive_text_question_uses_llm_fallback() -> None:
    llm_router = Mock()
    llm_router.route.return_value = RoutingDecision(
        QueryRoute.TEXT_ONLY, "LLM classified question as TEXT_ONLY"
    )
    router = AdaptiveQueryRouter(llm_router)

    decision = router.route("왜 attention score에 softmax를 사용하는가?")

    assert decision.route is QueryRoute.TEXT_ONLY
    assert decision.used_llm_router is True
    assert decision.reason.startswith("adaptive LLM fallback: ")
    llm_router.route.assert_called_once()


def test_adaptive_implicit_visual_question_uses_llm_fallback() -> None:
    llm_router = Mock()
    llm_router.route.return_value = RoutingDecision(
        QueryRoute.VISUAL_REQUIRED, "LLM classified question as VISUAL_REQUIRED"
    )
    router = AdaptiveQueryRouter(llm_router)

    decision = router.route("점선 화살표의 출발 블록과 도착 블록은 무엇인가?")

    assert decision.route is QueryRoute.VISUAL_REQUIRED
    assert decision.used_llm_router is True
    llm_router.route.assert_called_once()


def test_adaptive_propagates_llm_error() -> None:
    llm_router = Mock()
    error = FuriosaApiError("timed out")
    llm_router.route.side_effect = error

    with pytest.raises(FuriosaApiError) as raised:
        AdaptiveQueryRouter(llm_router).route("attention의 역할은?")
    assert raised.value is error


def test_adaptive_benchmark_records_llm_usage_and_summary() -> None:
    llm_router = Mock()
    llm_router.route.return_value = RoutingDecision(
        QueryRoute.TEXT_ONLY, "LLM classified question as TEXT_ONLY"
    )
    router = AdaptiveQueryRouter(llm_router)
    rows = [
        {
            "id": "A1",
            "question": "Figure 1의 색을 알려줘",
            "expected_route": "VISUAL_REQUIRED",
            "category": "explicit_visual",
        },
        {
            "id": "A2",
            "question": "왜 softmax를 사용하는가?",
            "expected_route": "TEXT_ONLY",
            "category": "text",
        },
    ]

    results = evaluate(rows, router)
    summary = summarize(results)

    assert [row["used_llm_router"] for row in results] == [False, True]
    assert summary["llm_router_calls"] == 1
    assert summary["llm_router_call_rate"] == 0.5
