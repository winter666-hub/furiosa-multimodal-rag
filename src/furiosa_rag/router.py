"""Deterministic query routing for text and visual document questions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint


class QueryRoute(str, Enum):
    TEXT_ONLY = "TEXT_ONLY"
    VISUAL_REQUIRED = "VISUAL_REQUIRED"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route: QueryRoute
    reason: str
    used_llm_router: bool = False


class QueryRouter(Protocol):
    """Common interface implemented by deterministic query routers."""

    def route(self, question: str) -> RoutingDecision:
        ...


class LLMRouterError(RuntimeError):
    """Raised when an LLM router response is not a valid routing label."""


class LLMQueryRouter:
    """Classify question intent with a text-only OpenAI-compatible LLM endpoint."""

    SYSTEM_PROMPT = """You are a routing classifier for a PDF question-answering system.

Classify whether answering the user's question requires inspecting visual
evidence from the PDF page.

VISUAL_REQUIRED:
The answer depends on visual/spatial information such as:
- figure or diagram content
- arrows or connections
- relative positions
- colors, lines, shapes
- chart/graph interpretation
- table row/column/cell lookup
- visual ordering/layout

TEXT_ONLY:
The question can be answered reliably from extracted document text without
inspecting page layout or visual elements.

Important:
Words such as "architecture", "structure", "comparison", or "flow" alone do
NOT imply VISUAL_REQUIRED.
Judge the actual information required by the question.

Return exactly one label:
TEXT_ONLY
VISUAL_REQUIRED"""

    def __init__(self, endpoint: ModelEndpoint, client: FuriosaClient) -> None:
        self.endpoint = endpoint
        self.client = client

    def route(self, question: str) -> RoutingDecision:
        if not question.strip():
            raise ValueError("question must not be empty")

        payload = self.client.post_json(
            self.endpoint.base_url,
            "chat/completions",
            {
                "model": self.endpoint.model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "max_tokens": 4,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FuriosaApiError(
                "LLM router response is missing choices[0].message.content"
            ) from exc
        if not isinstance(content, str):
            raise FuriosaApiError("LLM router response content is not a string")

        label = content.strip()
        try:
            route = QueryRoute(label)
        except ValueError as exc:
            raise LLMRouterError(f"invalid LLM router output: {content!r}") from exc
        return RoutingDecision(route, f"LLM classified question as {route.value}")


class RuleBasedQueryRouter:
    """Route only questions containing an explicit reference to visual content."""

    VISUAL_KEYWORDS = (
        "figure",
        "fig.",
        "table",
        "diagram",
        "chart",
        "image",
        "그림",
        "표에서",
        "도표",
        "다이어그램",
        "이미지",
    )

    def route(self, question: str) -> RoutingDecision:
        normalized = question.casefold()
        matched = next(
            (keyword for keyword in self.VISUAL_KEYWORDS if keyword in normalized), None
        )
        if matched is not None:
            return RoutingDecision(
                QueryRoute.VISUAL_REQUIRED,
                f"matched explicit visual keyword: {matched}",
            )
        return RoutingDecision(
            QueryRoute.TEXT_ONLY,
            "no explicit visual keyword matched",
        )


class EnhancedRuleBasedQueryRouter:
    """Add conservative semantic combinations while preserving baseline rules."""

    STRUCTURAL_TERMS = (
        "encoder", "decoder", "attention block", "feed-forward", "add & norm",
        "positional encoding", "transformer", "architecture", "블록", "모델",
    )
    COMPOSITION_TERMS = ("구성", "배치")
    COMPARISON_TERMS = ("비교", "차이")
    CONNECTION_TERMS = ("연결", "관계", "전달", "경로")
    FLOW_TERMS = ("입력", "출력", "데이터", "흐름", "시작", "끝", "거쳐", "과정")
    LAYOUT_TERMS = ("순서", "배치", "위치", "어디")
    ARCHITECTURE_CONTEXT_TERMS = ("전체", "흐름", "구성")

    def __init__(self) -> None:
        self._baseline = RuleBasedQueryRouter()

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def route(self, question: str) -> RoutingDecision:
        baseline_decision = self._baseline.route(question)
        if baseline_decision.route is QueryRoute.VISUAL_REQUIRED:
            return baseline_decision

        normalized = question.casefold()
        has_structure = self._contains_any(normalized, self.STRUCTURAL_TERMS) or (
            "입력" in normalized and "출력" in normalized
        )
        has_encoder_decoder = "encoder" in normalized and "decoder" in normalized
        patterns = (
            (
                "structure_comparison",
                has_structure
                and self._contains_any(normalized, self.COMPOSITION_TERMS)
                and (self._contains_any(normalized, self.COMPARISON_TERMS) or has_encoder_decoder),
            ),
            (
                "spatial_connection_relationship",
                has_structure
                and self._contains_any(normalized, self.CONNECTION_TERMS)
                and (has_encoder_decoder or self._contains_any(normalized, ("입력", "출력", "어디"))),
            ),
            (
                "information_flow",
                has_structure and sum(term in normalized for term in self.FLOW_TERMS) >= 2,
            ),
            (
                "ordering_layout",
                has_structure and self._contains_any(normalized, self.LAYOUT_TERMS),
            ),
            (
                "architecture_context",
                "architecture" in normalized
                and self._contains_any(normalized, self.ARCHITECTURE_CONTEXT_TERMS),
            ),
        )
        matched = next((name for name, is_match in patterns if is_match), None)
        if matched is not None:
            return RoutingDecision(
                QueryRoute.VISUAL_REQUIRED,
                f"matched semantic pattern: {matched}",
            )
        return RoutingDecision(
            QueryRoute.TEXT_ONLY,
            "no explicit visual keyword or semantic pattern matched",
        )


class AdaptiveQueryRouter:
    """Shortcut explicit visual references and delegate everything else to an LLM."""

    ADDITIONAL_EXPLICIT_PATTERNS = (
        ("graph", re.compile(r"(?<!\w)graph(?!\w)", re.IGNORECASE)),
        ("표", re.compile(r"(?<![가-힣])표(?:에서|의|를|가|는|에|와|로|$|\s)")),
        ("도식", re.compile(r"도식")),
        ("그래프", re.compile(r"그래프")),
        ("차트", re.compile(r"차트")),
        ("구조도", re.compile(r"구조도")),
    )

    def __init__(self, llm_router: LLMQueryRouter) -> None:
        self._explicit_router = RuleBasedQueryRouter()
        self._llm_router = llm_router

    def route(self, question: str) -> RoutingDecision:
        explicit = self._explicit_router.route(question)
        if explicit.route is QueryRoute.VISUAL_REQUIRED:
            matched = explicit.reason.removeprefix("matched explicit visual keyword: ")
            return RoutingDecision(
                QueryRoute.VISUAL_REQUIRED,
                f"adaptive explicit visual shortcut: {matched}",
            )

        matched = next(
            (
                name
                for name, pattern in self.ADDITIONAL_EXPLICIT_PATTERNS
                if pattern.search(question)
            ),
            None,
        )
        if matched is not None:
            return RoutingDecision(
                QueryRoute.VISUAL_REQUIRED,
                f"adaptive explicit visual shortcut: {matched}",
            )

        decision = self._llm_router.route(question)
        return RoutingDecision(
            decision.route,
            f"adaptive LLM fallback: {decision.reason}",
            used_llm_router=True,
        )
