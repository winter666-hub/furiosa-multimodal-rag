from __future__ import annotations

import uuid
from pathlib import Path
from typing import Self
from unittest.mock import Mock, patch
from urllib.error import URLError

import pymupdf
import pytest
from fastapi.testclient import TestClient

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.models import Chunk, RagAnswer, RetrievedChunk
from furiosa_rag.router import QueryRoute, RoutingDecision
from furiosa_rag.web.app import (
    HostedOnlyRagService,
    app,
    ensure_demo_pdf,
    get_chat_log_repository,
    get_demo_pdf_path,
    get_service,
    parse_allowed_origins,
    render_page_png,
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
    def __init__(self, answer: str = "answer") -> None:
        self.calls: list[tuple[Path, str]] = []
        self.answer_text = answer

    def answer(self, pdf_path: Path, question: str) -> RagAnswer:
        self.calls.append((pdf_path, question))
        source = RetrievedChunk(Chunk("page-5-chunk-1", 5, "evidence"), 0.9, 0.8)
        return RagAnswer(
            self.answer_text,
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
    get_demo_pdf_path.cache_clear()
    render_page_png.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_demo_pdf_path.cache_clear()
    render_page_png.cache_clear()


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


def test_startup_initializes_schema_without_affecting_health() -> None:
    repository = Mock()
    with (
        patch("furiosa_rag.web.app.get_chat_log_repository", return_value=repository),
        TestClient(app) as client,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    repository.initialize.assert_called_once_with()


def test_startup_database_failure_is_safe_and_does_not_log_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = Mock()
    repository.initialize.side_effect = RuntimeError("postgres://secret:password@host/db")
    with (
        patch("furiosa_rag.web.app.get_chat_log_repository", return_value=repository),
        caplog.at_level("WARNING"),
        TestClient(app) as client,
    ):
        response = client.get("/health")

    assert response.status_code == 200
    assert "Failed to initialize chat log persistence" in caplog.text
    assert "password" not in caplog.text


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
    assert len(body["sources"]) == 1
    assert body["sources"][0]["page"] == 5
    assert body["sources"][0]["chunk"] == "page-5-chunk-1"
    assert body["sources"][0]["chunk_id"] == "page-5-chunk-1"
    assert body["sources"][0]["excerpt"] == "evidence"
    assert body["sources"][0]["retrieval_score"] == 0.9
    assert len(pipeline.calls) == 1


def test_successful_ask_persists_approved_conversation_payload(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(
        router,
        FakeTextPipeline("Hello [page 5, chunk page-5-chunk-1]."),
        pdf_path,
    )  # type: ignore[arg-type]
    repository = Mock()
    session_id = uuid.uuid4()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_chat_log_repository] = lambda: repository

    response = TestClient(app).post(
        "/ask", json={"question": "question", "session_id": str(session_id)}
    )

    assert response.status_code == 200
    repository.persist.assert_called_once()
    record = repository.persist.call_args.args[0]
    assert record.session_id == session_id
    assert record.document_id is None
    assert record.filename == "paper.pdf"
    assert record.question == "question"
    assert response.json()["answer"] == "Hello."
    assert record.answer == "Hello."
    assert record.route == "TEXT_ONLY"
    assert record.routing_reason == "text route"
    assert record.sources == [
        {
            "page": 5,
            "chunk_id": "page-5-chunk-1",
            "excerpt": "evidence",
            "retrieval_score": 0.9,
            "rerank_score": 0.8,
        }
    ]
    assert record.latency_ms["answer_generation"] == 4.0
    assert set(record.__dataclass_fields__) == {
        "id",
        "session_id",
        "document_id",
        "filename",
        "question",
        "answer",
        "route",
        "routing_reason",
        "vision_used",
        "vision_available",
        "fallback_used",
        "sources",
        "latency_ms",
        "created_at",
    }


def test_missing_session_id_gets_backend_generated_uuid(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(router, FakeTextPipeline(), pdf_path)  # type: ignore[arg-type]
    repository = Mock()
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_chat_log_repository] = lambda: repository

    response = TestClient(app).post("/ask", json={"question": "question"})

    assert response.status_code == 200
    assert isinstance(repository.persist.call_args.args[0].session_id, uuid.UUID)


def test_persistence_failure_does_not_fail_answer_or_log_details(
    pdf_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(router, FakeTextPipeline(), pdf_path)  # type: ignore[arg-type]
    repository = Mock()
    repository.persist.side_effect = RuntimeError("postgres://secret-user:secret-pass@host/db")
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_chat_log_repository] = lambda: repository

    with caplog.at_level("WARNING"):
        response = TestClient(app).post("/ask", json={"question": "question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "answer"
    assert "Failed to persist chat log" in caplog.text
    assert "secret-pass" not in caplog.text


def test_disabled_persistence_keeps_ask_behavior(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(router, FakeTextPipeline(), pdf_path)  # type: ignore[arg-type]
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_chat_log_repository] = lambda: None

    response = TestClient(app).post("/ask", json={"question": "question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "answer"


def test_sources_use_retrieved_text_and_deduplicate_exact_chunk(tmp_path: Path) -> None:
    pdf_path = _create_test_pdf(tmp_path / "sources.pdf", pages=1)
    first = RetrievedChunk(Chunk("page-1-chunk-1", 1, "actual retrieved first text"), 0.9, 0.8)
    duplicate = RetrievedChunk(
        Chunk("page-1-chunk-1", 1, "actual retrieved first text"), 0.7, 0.6
    )
    second = RetrievedChunk(
        Chunk("page-1-chunk-2", 1, "actual retrieved second text"), 0.5, 0.4
    )
    pipeline = Mock()
    pipeline.answer.return_value = RagAnswer("answer", (first, duplicate, second), {"total": 1.0})
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(router, pipeline, pdf_path)

    response = service.ask("question")

    assert [source.chunk_id for source in response.sources] == [
        "page-1-chunk-1",
        "page-1-chunk-2",
    ]
    assert [source.excerpt for source in response.sources] == [
        "actual retrieved first text",
        "actual retrieved second text",
    ]


def test_source_response_includes_highlight_coordinates_when_text_matches(tmp_path: Path) -> None:
    pdf_path = tmp_path / "highlight.pdf"
    document = pymupdf.open()
    document.new_page().insert_text(
        (72, 72), "BERT is designed to pre-train deep bidirectional representations."
    )
    document.save(pdf_path)
    document.close()
    source = RetrievedChunk(
        Chunk(
            "page-1-chunk-1",
            1,
            "BERT is designed to pre-train deep bidirectional representations.",
        ),
        0.9,
        0.8,
    )
    pipeline = Mock()
    pipeline.answer.return_value = RagAnswer("answer", (source,), {"total": 1.0})
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")

    response = HostedOnlyRagService(router, pipeline, pdf_path).ask("question")

    assert response.sources[0].page_width is not None
    assert response.sources[0].page_height is not None
    assert response.sources[0].highlights


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


def _create_test_pdf(path: Path, pages: int = 2) -> Path:
    document = pymupdf.open()
    for index in range(1, pages + 1):
        document.new_page().insert_text((72, 72), f"page {index}")
    document.save(path)
    document.close()
    return path


def test_document_page_returns_png_for_valid_one_based_page(tmp_path: Path) -> None:
    document = _create_test_pdf(tmp_path / "pages.pdf")
    app.dependency_overrides[get_demo_pdf_path] = lambda: document

    response = TestClient(app).get("/document/page/2")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_document_page_zero_is_rejected(pdf_path: Path) -> None:
    app.dependency_overrides[get_demo_pdf_path] = lambda: pdf_path
    response = TestClient(app).get("/document/page/0")
    assert response.status_code == 422


def test_document_page_above_total_pages_returns_404(tmp_path: Path) -> None:
    document = _create_test_pdf(tmp_path / "one-page.pdf", pages=1)
    app.dependency_overrides[get_demo_pdf_path] = lambda: document
    response = TestClient(app).get("/document/page/2")
    assert response.status_code == 404
    assert response.json() == {"detail": "document page not found"}


def test_document_page_missing_pdf_returns_existing_unavailable_error() -> None:
    app.dependency_overrides[get_demo_pdf_path] = lambda: None
    response = TestClient(app).get("/document/page/1")
    assert response.status_code == 503
    assert response.json() == {"detail": "demo document unavailable"}


def test_source_page_number_is_passed_unchanged_to_preview_endpoint(pdf_path: Path) -> None:
    router = Mock()
    router.route.return_value = RoutingDecision(QueryRoute.TEXT_ONLY, "text route")
    service = HostedOnlyRagService(
        router,
        FakeTextPipeline(),
        pdf_path,
    )  # type: ignore[arg-type]
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_demo_pdf_path] = lambda: pdf_path

    ask_response = TestClient(app).post("/ask", json={"question": "question"})
    source_page = ask_response.json()["sources"][0]["page"]
    with patch("furiosa_rag.web.app.render_page_png", return_value=b"\x89PNG") as render:
        preview_response = TestClient(app).get(f"/document/page/{source_page}")

    assert source_page == 5
    assert preview_response.status_code == 200
    render.assert_called_once_with(str(pdf_path), 5)
