from __future__ import annotations

from pathlib import Path
from typing import Self
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest
from fastapi.testclient import TestClient

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.models import Chunk, RagAnswer, RetrievedChunk
from furiosa_rag.router import QueryRoute, RoutingDecision
from furiosa_rag.web.app import (
    HostedOnlyRagService,
    app,
    ensure_demo_pdf,
    get_service,
    parse_allowed_origins,
)


class FakeDownloadResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.content) - self.offset
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeTextPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def answer(self, pdf_path: Path, question: str) -> RagAnswer:
        self.calls.append((pdf_path, question))
        source = RetrievedChunk(Chunk("page-5-chunk-1", 5, "evidence"), 0.9, 0.8)
        return RagAnswer(
            "answer",
            (source,),
            {"query_embedding": 2.0, "answer_generation": 4.0, "total": 6.0},
        )


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"fake pdf")
    return path


@pytest.fixture(autouse=True)
def clear_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_health_is_immediate_and_does_not_resolve_service() -> None:
    app.dependency_overrides[get_service] = Mock(side_effect=AssertionError("must not initialize"))
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_does_not_trigger_pdf_download() -> None:
    with patch("furiosa_rag.web.app.urlopen") as download:
        response = TestClient(app).get("/health")
    assert response.status_code == 200
    download.assert_not_called()


def test_existing_demo_pdf_skips_download(pdf_path: Path) -> None:
    with patch("furiosa_rag.web.app.urlopen") as download:
        result = ensure_demo_pdf(
            pdf_path,
            "https://example.test/paper.pdf",
            timeout=3.0,
        )
    assert result == pdf_path
    download.assert_not_called()


def test_missing_demo_pdf_is_downloaded_and_atomically_saved(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "paper.pdf"
    response = FakeDownloadResponse(b"%PDF-1.7\nbody")
    with patch("furiosa_rag.web.app.urlopen", return_value=response) as download:
        result = ensure_demo_pdf(
            destination,
            "https://example.test/paper.pdf",
            timeout=7.0,
        )
    assert result == destination
    assert destination.read_bytes() == b"%PDF-1.7\nbody"
    assert list(destination.parent.glob(".*.tmp")) == []
    assert download.call_args.kwargs["timeout"] == 7.0


def test_html_download_is_rejected_without_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "paper.pdf"
    response = FakeDownloadResponse(b"<html>upstream error</html>")
    with patch("furiosa_rag.web.app.urlopen", return_value=response):
        result = ensure_demo_pdf(
            destination,
            "https://example.test/paper.pdf",
            timeout=3.0,
        )
    assert result is None
    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_download_failure_leaves_no_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "paper.pdf"
    with patch("furiosa_rag.web.app.urlopen", side_effect=URLError("offline")):
        result = ensure_demo_pdf(
            destination,
            "https://example.test/paper.pdf",
            timeout=3.0,
        )
    assert result is None
    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("question", ("", "   ", "\n\t"))
def test_ask_rejects_blank_question(question: str) -> None:
    response = TestClient(app).post("/ask", json={"question": question})
    assert response.status_code == 422


def test_text_only_uses_text_pipeline(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    pipeline = FakeTextPipeline()
    service = HostedOnlyRagService(router, pipeline, pdf_path)  # type: ignore[arg-type]
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).post("/ask", json={"question": "  normal question  "})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "normal question"
    assert body["route"] == "TEXT_ONLY"
    assert body["vision_used"] is False
    assert body["vision_available"] is False
    assert body["fallback_used"] is False
    assert body["sources"] == [{"page": 5, "chunk": "page-5-chunk-1"}]
    assert len(pipeline.calls) == 1


def test_hosted_only_visual_route_falls_back_without_vision_call(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(
        QueryRoute.VISUAL_REQUIRED, "adaptive explicit visual shortcut: figure"
    )
    pipeline = FakeTextPipeline()
    direct_npu_vision = Mock()
    service = HostedOnlyRagService(router, pipeline, pdf_path)  # type: ignore[arg-type]
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).post("/ask", json={"question": "Figure 1을 설명해줘"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "VISUAL_REQUIRED"
    assert body["vision_used"] is False
    assert body["vision_available"] is False
    assert body["fallback_used"] is True
    assert len(pipeline.calls) == 1
    direct_npu_vision.assert_not_called()


def test_missing_pdf_returns_safe_503() -> None:
    service = HostedOnlyRagService(Mock(), FakeTextPipeline(), "missing.pdf")  # type: ignore[arg-type]
    app.dependency_overrides[get_service] = lambda: service
    response = TestClient(app).post("/ask", json={"question": "question"})
    assert response.status_code == 503
    assert response.json() == {"detail": "demo document unavailable"}


def test_hosted_api_failure_returns_safe_502() -> None:
    service = Mock()
    service.ask.side_effect = FuriosaApiError("secret endpoint detail")
    app.dependency_overrides[get_service] = lambda: service

    response = TestClient(app).post("/ask", json={"question": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream model service unavailable"}
    assert "secret" not in response.text


def test_parse_allowed_origins_is_trimmed_deduplicated_and_safe() -> None:
    assert parse_allowed_origins(None) == []
    assert parse_allowed_origins("") == []
    assert parse_allowed_origins(
        " https://demo.example, http://localhost:5173,https://demo.example "
    ) == ["https://demo.example", "http://localhost:5173"]
