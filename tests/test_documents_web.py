from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from furiosa_rag.models import Chunk, RagAnswer, RetrievedChunk
from furiosa_rag.router import QueryRoute, RoutingDecision
from furiosa_rag.web.app import (
    HostedOnlyRagService,
    app,
    get_document_store,
    get_service,
    render_page_png,
)
from furiosa_rag.web.documents import DocumentStore, RegisteredDocument


def _pdf_bytes(label: str = "paper", *, pages: int = 1) -> bytes:
    document = pymupdf.open()
    for index in range(pages):
        document.new_page().insert_text((72, 72), f"{label} page {index + 1}")
    content = document.tobytes()
    document.close()
    return content


def _upload(content: bytes, filename: str = "paper.pdf", content_type: str = "application/pdf"):
    return UploadFile(
        BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def _register(store: DocumentStore, content: bytes, filename: str = "paper.pdf"):
    return asyncio.run(store.register(_upload(content, filename)))


class FakeUploadService:
    def __init__(self, store: DocumentStore) -> None:
        self.document_store = store
        self.processing_calls = 0

    def prepare_document(self, document: RegisteredDocument) -> None:
        if document.cache_hit:
            return
        self.processing_calls += 1
        document.cache_dir.mkdir(parents=True, exist_ok=True)
        (document.cache_dir / "prepared.npz").write_bytes(b"cache")


@pytest.fixture(autouse=True)
def clear_web_state() -> None:
    app.dependency_overrides.clear()
    get_document_store.cache_clear()
    render_page_png.cache_clear()
    yield
    app.dependency_overrides.clear()
    get_document_store.cache_clear()
    render_page_png.cache_clear()


def test_valid_pdf_upload_and_metadata(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    service = FakeUploadService(store)
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_document_store] = lambda: store

    response = TestClient(app).post(
        "/documents",
        files={"file": ("bert.pdf", _pdf_bytes("bert", pages=2), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["document_id"]) == 64
    assert body["filename"] == "bert.pdf"
    assert body["pages"] == 2
    assert body["status"] == "ready"
    metadata = TestClient(app).get(f"/documents/{body['document_id']}")
    assert metadata.status_code == 200
    assert metadata.json()["document_id"] == body["document_id"]


@pytest.mark.parametrize(
    "filename, content_type, content",
    (
        ("paper.txt", "application/pdf", b"%PDF-bad"),
        ("paper.pdf", "text/plain", b"%PDF-bad"),
        ("paper.pdf", "application/pdf", b"not a pdf"),
        ("paper.pdf", "application/pdf", b"%PDF-broken"),
    ),
    ids=("extension", "content-type", "magic", "broken-pdf"),
)
def test_invalid_pdf_upload_is_rejected(
    tmp_path: Path, filename: str, content_type: str, content: bytes
) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    app.dependency_overrides[get_service] = lambda: FakeUploadService(store)
    response = TestClient(app).post(
        "/documents", files={"file": (filename, content, content_type)}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid PDF"}


def test_zero_page_pdf_is_rejected(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    app.dependency_overrides[get_service] = lambda: FakeUploadService(store)
    fake_document = Mock()
    fake_document.page_count = 0
    fake_document.__enter__ = Mock(return_value=fake_document)
    fake_document.__exit__ = Mock(return_value=None)
    with patch("furiosa_rag.web.documents.pymupdf.open", return_value=fake_document):
        response = TestClient(app).post(
            "/documents",
            files={"file": ("paper.pdf", b"%PDF-content", "application/pdf")},
        )
    assert response.status_code == 400


def test_upload_size_limit_returns_413(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=8)
    app.dependency_overrides[get_service] = lambda: FakeUploadService(store)
    response = TestClient(app).post(
        "/documents",
        files={"file": ("paper.pdf", b"%PDF-too-large", "application/pdf")},
    )
    assert response.status_code == 413
    assert list(tmp_path.glob(".upload-*.tmp")) == []


def test_duplicate_upload_has_stable_id_and_reuses_processing(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    service = FakeUploadService(store)
    app.dependency_overrides[get_service] = lambda: service
    content = _pdf_bytes("same")

    first = TestClient(app).post(
        "/documents", files={"file": ("first.pdf", content, "application/pdf")}
    )
    second = TestClient(app).post(
        "/documents", files={"file": ("renamed.pdf", content, "application/pdf")}
    )

    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert service.processing_calls == 1


def _set_access_time(document: RegisteredDocument, timestamp: float) -> None:
    metadata_path = document.pdf_path.parent / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["created_at"] = timestamp
    payload["last_accessed_at"] = timestamp
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")


def test_cleanup_removes_expired_document_directory(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024, document_ttl_hours=1)
    expired = _register(store, _pdf_bytes("expired"))
    current = _register(store, _pdf_bytes("current"))
    _set_access_time(expired, 1_000)
    _set_access_time(current, 10_000)

    removed = store.cleanup(now=10_000)

    assert removed == [expired.record.document_id]
    assert not expired.pdf_path.parent.exists()
    assert current.pdf_path.is_file()


def test_cleanup_enforces_document_count_oldest_first(tmp_path: Path) -> None:
    store = DocumentStore(
        tmp_path,
        max_upload_bytes=1024 * 1024,
        max_documents=2,
        document_ttl_hours=1_000_000_000,
    )
    first = _register(store, _pdf_bytes("first"))
    _set_access_time(first, 1_000)
    second = _register(store, _pdf_bytes("second"))
    _set_access_time(second, 2_000)
    third = _register(store, _pdf_bytes("third"))

    assert not first.pdf_path.parent.exists()
    assert second.pdf_path.is_file() and third.pdf_path.is_file()


def test_cleanup_enforces_storage_cap(tmp_path: Path) -> None:
    content_a = _pdf_bytes("storage-a")
    content_b = _pdf_bytes("storage-b")
    cap = max(len(content_a), len(content_b)) + 200
    store = DocumentStore(
        tmp_path,
        max_upload_bytes=1024 * 1024,
        max_storage_bytes=cap,
    )
    first = _register(store, content_a)
    _set_access_time(first, 1_000)
    second = _register(store, content_b)

    assert not first.pdf_path.parent.exists()
    assert second.pdf_path.is_file()


def test_cleanup_preserves_active_document(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024, document_ttl_hours=1)
    document = _register(store, _pdf_bytes("active"))
    _set_access_time(document, 1_000)

    with store.processing(document.record.document_id):
        assert store.cleanup(now=10_000) == []
        assert document.pdf_path.is_file()


def test_same_pdf_concurrent_registration_is_atomic(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    content = _pdf_bytes("concurrent")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda name: _register(store, content, name),
                ("first.pdf", "second.pdf"),
            )
        )

    assert results[0].record.document_id == results[1].record.document_id
    metadata_path = results[0].pdf_path.parent / "metadata.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["document_id"] == results[0].record.document_id
    assert results[0].pdf_path.read_bytes() == content
    assert list(results[0].pdf_path.parent.glob(".metadata.*.tmp")) == []


class IsolatedPipeline:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.paths: list[Path] = []

    def answer(self, pdf_path: Path, question: str) -> RagAnswer:
        self.paths.append(Path(pdf_path))
        source = RetrievedChunk(Chunk(f"{self.document_id}-page-1", 1, "text"), 1.0, 1.0)
        return RagAnswer(f"answer-{self.document_id}", (source,), {"total": 1.0})


class IsolatedFactory:
    def __init__(self) -> None:
        self.pipelines: dict[str, IsolatedPipeline] = {}
        self.documents: list[RegisteredDocument] = []

    def create(self, document: RegisteredDocument) -> IsolatedPipeline:
        self.documents.append(document)
        return self.pipelines.setdefault(
            document.record.document_id,
            IsolatedPipeline(document.record.document_id),
        )


def _uploaded_service(store: DocumentStore, factory: IsolatedFactory, route: QueryRoute):
    router = Mock()
    router.route.return_value = RoutingDecision(route, "test route")
    return HostedOnlyRagService(
        router,
        Mock(),
        "",
        document_store=store,
        pipeline_factory=factory,  # type: ignore[arg-type]
    )


def test_ask_isolates_two_uploaded_documents_and_caches(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    first = _register(store, _pdf_bytes("A"), "a.pdf")
    second = _register(store, _pdf_bytes("B"), "b.pdf")
    factory = IsolatedFactory()
    service = _uploaded_service(store, factory, QueryRoute.TEXT_ONLY)
    app.dependency_overrides[get_service] = lambda: service

    response_a = TestClient(app).post(
        "/ask", json={"document_id": first.record.document_id, "question": "question A"}
    )
    response_b = TestClient(app).post(
        "/ask", json={"document_id": second.record.document_id, "question": "question B"}
    )

    assert response_a.json()["answer"] == f"answer-{first.record.document_id}"
    assert response_b.json()["answer"] == f"answer-{second.record.document_id}"
    assert factory.pipelines[first.record.document_id].paths == [first.pdf_path]
    assert factory.pipelines[second.record.document_id].paths == [second.pdf_path]
    assert first.cache_dir != second.cache_dir


def test_uploaded_visual_route_keeps_hosted_only_fallback(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    document = _register(store, _pdf_bytes("visual"))
    service = _uploaded_service(store, IsolatedFactory(), QueryRoute.VISUAL_REQUIRED)
    app.dependency_overrides[get_service] = lambda: service
    response = TestClient(app).post(
        "/ask",
        json={"document_id": document.record.document_id, "question": "arrow location?"},
    )
    assert response.status_code == 200
    assert response.json()["vision_used"] is False
    assert response.json()["vision_available"] is False
    assert response.json()["fallback_used"] is True


def test_unknown_document_ask_returns_404(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    service = _uploaded_service(store, IsolatedFactory(), QueryRoute.TEXT_ONLY)
    app.dependency_overrides[get_service] = lambda: service
    response = TestClient(app).post(
        "/ask", json={"document_id": "0" * 64, "question": "question"}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "document not found"}


def test_uploaded_page_preview_and_same_page_cache_isolation(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    first = _register(store, _pdf_bytes("A"), "a.pdf")
    second = _register(store, _pdf_bytes("B"), "b.pdf")
    app.dependency_overrides[get_document_store] = lambda: store

    def fake_render(path: str, page: int) -> bytes:
        return b"PNG:" + Path(path).parent.name.encode() + f":{page}".encode()

    with patch("furiosa_rag.web.app.render_page_png", side_effect=fake_render) as render:
        response_a = TestClient(app).get(
            f"/documents/{first.record.document_id}/pages/1"
        )
        response_b = TestClient(app).get(
            f"/documents/{second.record.document_id}/pages/1"
        )
    assert response_a.status_code == response_b.status_code == 200
    assert response_a.headers["content-type"] == "image/png"
    assert response_a.content != response_b.content
    assert render.call_args_list[0].args[0] != render.call_args_list[1].args[0]


def test_uploaded_page_errors(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path, max_upload_bytes=1024 * 1024)
    document = _register(store, _pdf_bytes("one page"))
    app.dependency_overrides[get_document_store] = lambda: store
    client = TestClient(app)
    assert client.get(f"/documents/{document.record.document_id}/pages/0").status_code == 422
    assert client.get(f"/documents/{document.record.document_id}/pages/2").status_code == 404
    assert client.get(f"/documents/{'f' * 64}/pages/1").status_code == 404
