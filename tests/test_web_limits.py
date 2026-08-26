from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient
from starlette.requests import Request

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.web.app import (
    AskResponse,
    app,
    get_ask_concurrency_limiter,
    get_ask_rate_limiter,
    get_service,
    get_upload_concurrency_limiter,
    get_upload_rate_limiter,
)
from furiosa_rag.web.documents import DocumentStore
from furiosa_rag.web.limits import ConcurrencyLimiter, RateLimiter, client_ip

PROXY_SECRET = "test-proxy-secret"


def _request(*, host: str = "203.0.113.10", headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (name.casefold().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/ask",
            "headers": encoded_headers,
            "client": (host, 12345),
        }
    )


def test_client_ip_uses_valid_trusted_proxy_headers() -> None:
    request = _request(
        headers={
            "X-Paper-RAG-Proxy-Token": PROXY_SECRET,
            "X-Paper-RAG-Client-IP": "2001:db8::1",
        }
    )
    assert client_ip(request, PROXY_SECRET) == "2001:db8::1"


def test_client_ip_ignores_forwarded_ip_for_invalid_or_missing_secret() -> None:
    forwarded = {
        "X-Paper-RAG-Proxy-Token": "wrong-secret",
        "X-Paper-RAG-Client-IP": "192.0.2.50",
        "X-Forwarded-For": "192.0.2.60",
        "X-Real-IP": "192.0.2.70",
    }
    request = _request(headers=forwarded)

    assert client_ip(request, PROXY_SECRET) == "203.0.113.10"
    assert client_ip(request, None) == "203.0.113.10"


def test_client_ip_falls_back_for_malformed_trusted_proxy_ip() -> None:
    request = _request(
        headers={
            "X-Paper-RAG-Proxy-Token": PROXY_SECRET,
            "X-Paper-RAG-Client-IP": "not-an-ip, 192.0.2.2",
        }
    )
    assert client_ip(request, PROXY_SECRET) == "203.0.113.10"


def test_direct_render_request_uses_peer_host() -> None:
    assert client_ip(_request(host="198.51.100.25"), PROXY_SECRET) == "198.51.100.25"


def _answer() -> AskResponse:
    return AskResponse(
        question="question",
        answer="answer",
        route="TEXT_ONLY",
        routing_reason="test",
        vision_used=False,
        vision_available=False,
        fallback_used=False,
        sources=[],
        latency_ms={"total": 1.0},
    )


def test_upload_rate_limit_has_retry_after_and_separate_ip_buckets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PAPER_RAG_PROXY_SECRET", PROXY_SECRET)
    limiter = RateLimiter(1, 60)
    app.dependency_overrides[get_upload_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_service] = lambda: Mock(
        document_store=DocumentStore(tmp_path, max_upload_bytes=1024)
    )
    client = TestClient(app)
    files = {"file": ("bad.txt", b"bad", "text/plain")}

    proxy_headers_a = {
        "x-paper-rag-proxy-token": PROXY_SECRET,
        "x-paper-rag-client-ip": "192.0.2.1",
    }
    proxy_headers_b = {
        "x-paper-rag-proxy-token": PROXY_SECRET,
        "x-paper-rag-client-ip": "198.51.100.2",
    }
    first = client.post("/documents", files=files, headers=proxy_headers_a)
    limited = client.post("/documents", files=files, headers=proxy_headers_a)
    independent = client.post(
        "/documents", files=files, headers=proxy_headers_b
    )

    assert first.status_code == independent.status_code == 400
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.json() == {"detail": "Too many requests. Please try again later."}


def test_ask_rate_limit() -> None:
    limiter = RateLimiter(1, 60)
    service = Mock()
    service.ask.return_value = _answer()
    service.document_store = None
    app.dependency_overrides[get_ask_rate_limiter] = lambda: limiter
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    assert client.post("/ask", json={"question": "one"}).status_code == 200
    response = client.post("/ask", json={"question": "two"})
    assert response.status_code == 429
    assert "retry-after" in response.headers


def test_busy_concurrency_responses() -> None:
    upload_limiter = ConcurrencyLimiter(1)
    ask_limiter = ConcurrencyLimiter(1)
    assert upload_limiter.acquire() and ask_limiter.acquire()
    app.dependency_overrides[get_upload_concurrency_limiter] = lambda: upload_limiter
    app.dependency_overrides[get_ask_concurrency_limiter] = lambda: ask_limiter
    service = Mock(document_store=Mock())
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    upload = client.post(
        "/documents", files={"file": ("paper.pdf", b"%PDF-bad", "application/pdf")}
    )
    ask = client.post("/ask", json={"question": "question"})

    assert upload.status_code == ask.status_code == 503
    assert ask.json() == {
        "detail": "The demo is currently busy. Please try again shortly."
    }
    upload_limiter.release()
    ask_limiter.release()


def test_ask_concurrency_is_released_after_exception() -> None:
    limiter = ConcurrencyLimiter(1)
    service = Mock(document_store=None)
    service.ask.side_effect = FuriosaApiError("failure")
    app.dependency_overrides[get_ask_concurrency_limiter] = lambda: limiter
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    assert client.post("/ask", json={"question": "first"}).status_code == 502
    service.ask.side_effect = None
    service.ask.return_value = _answer()
    assert client.post("/ask", json={"question": "second"}).status_code == 200
