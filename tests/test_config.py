from furiosa_rag.config import Settings


def test_vision_endpoint_comes_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FURIOSA_VISION_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("FURIOSA_VISION_MODEL", "local-qwen-vl")
    monkeypatch.setenv("FURIOSA_VISION_MAX_TOKENS", "128")
    settings = Settings.from_env(tmp_path / "missing.env")
    vision = next(endpoint for endpoint in settings.endpoints if endpoint.name == "vision")
    assert vision.base_url == "http://localhost:8000/v1"
    assert vision.model == "local-qwen-vl"
    assert settings.vision_max_tokens == 128


def test_vision_max_tokens_defaults_to_256(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FURIOSA_VISION_MAX_TOKENS", raising=False)
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.vision_max_tokens == 256


def test_vision_request_timeout_defaults_to_60(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FURIOSA_VISION_REQUEST_TIMEOUT", raising=False)
    settings = Settings.from_env(tmp_path / "missing.env")
    assert settings.vision_request_timeout == 60


def test_vision_request_timeout_must_be_positive(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FURIOSA_VISION_REQUEST_TIMEOUT", "0")
    try:
        Settings.from_env(tmp_path / "missing.env")
    except ValueError as exc:
        assert "FURIOSA_VISION_REQUEST_TIMEOUT" in str(exc)
    else:
        raise AssertionError("Settings accepted a non-positive Vision request timeout")


def test_vision_max_tokens_must_be_positive(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FURIOSA_VISION_MAX_TOKENS", "0")
    try:
        Settings.from_env(tmp_path / "missing.env")
    except ValueError as exc:
        assert "FURIOSA_VISION_MAX_TOKENS" in str(exc)
    else:
        raise AssertionError("Settings accepted a non-positive Vision token limit")
