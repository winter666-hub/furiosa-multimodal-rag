from __future__ import annotations

import io
from unittest.mock import patch

from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import ModelEndpoint


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_check_connection_reads_openai_model_list() -> None:
    response = FakeResponse(b'{"object":"list","data":[{"id":"model-a"}]}')
    endpoint = ModelEndpoint("llm", "http://npu:8000/v1", "model-a")

    with patch("furiosa_rag.clients.furiosa.urlopen", return_value=response) as mocked:
        result = FuriosaClient(api_key="secret").check_connection(endpoint)

    assert result.ok is True
    assert result.available_models == ("model-a",)
    request = mocked.call_args.args[0]
    assert request.full_url == "http://npu:8000/v1/models"
    assert request.get_header("Authorization") == "Bearer secret"


def test_check_connection_adds_v1_and_reports_invalid_json() -> None:
    response = FakeResponse(io.BytesIO(b"not-json").read())
    endpoint = ModelEndpoint("embedding", "http://npu:8002/", "embedding-model")

    with patch("furiosa_rag.clients.furiosa.urlopen", return_value=response):
        result = FuriosaClient().check_connection(endpoint)

    assert result.ok is False
    assert result.url == "http://npu:8002/v1/models"
    assert result.error is not None
    assert "JSONDecodeError" in result.error

