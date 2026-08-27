from furiosa_rag.cli.smoke_test_vision import run_vision_smoke
from furiosa_rag.clients import ConnectionResult
from furiosa_rag.config import Settings


class FakeVisionClient:
    def __init__(self, *, connected: bool = True, content: str = "The image is white.") -> None:
        self.connected = connected
        self.content = content

    def check_connection(self, endpoint):
        return ConnectionResult(
            endpoint="vision",
            url="http://127.0.0.1:8001/v1/models",
            expected_model=endpoint.model,
            available_models=(endpoint.model,) if self.connected else (),
            latency_ms=1.0,
            ok=self.connected,
            error=None if self.connected else "model unavailable",
        )

    def post_json(self, base_url, path, payload):
        assert base_url == "http://127.0.0.1:8001/v1"
        assert path == "chat/completions"
        image = payload["messages"][1]["content"][0]["image_url"]["url"]
        assert image.startswith("data:image/png;base64,")
        return {"choices": [{"message": {"content": self.content}}]}


def _settings(monkeypatch, tmp_path) -> Settings:
    monkeypatch.setenv("FURIOSA_VISION_BASE_URL", "http://127.0.0.1:8001/v1")
    return Settings.from_env(tmp_path / "missing.env")


def test_vision_smoke_succeeds_with_nonempty_response(monkeypatch, tmp_path) -> None:
    assert run_vision_smoke(_settings(monkeypatch, tmp_path), FakeVisionClient()) is True


def test_vision_smoke_fails_when_endpoint_model_is_unavailable(monkeypatch, tmp_path) -> None:
    client = FakeVisionClient(connected=False)
    assert run_vision_smoke(_settings(monkeypatch, tmp_path), client) is False


def test_vision_smoke_fails_on_empty_response(monkeypatch, tmp_path) -> None:
    client = FakeVisionClient(content="")
    assert run_vision_smoke(_settings(monkeypatch, tmp_path), client) is False
