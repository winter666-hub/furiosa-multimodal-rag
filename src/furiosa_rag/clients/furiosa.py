"""Small dependency-free client for Furiosa-LLM health diagnostics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from furiosa_rag.config import ModelEndpoint


@dataclass(frozen=True, slots=True)
class ConnectionResult:
    endpoint: str
    url: str
    expected_model: str
    available_models: tuple[str, ...]
    latency_ms: float
    ok: bool
    error: str | None = None


class FuriosaApiError(RuntimeError):
    """Raised when a Furiosa-compatible endpoint cannot complete a request."""


class FuriosaClient:
    def __init__(self, api_key: str = "EMPTY", timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _models_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        return f"{normalized}/models"

    @staticmethod
    def api_url(base_url: str, path: str) -> str:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        return f"{normalized}/{path.lstrip('/')}"

    def post_json(self, base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON to an OpenAI-compatible API with normalized errors."""
        url = self.api_url(base_url, path)
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FuriosaApiError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise FuriosaApiError(f"POST {url} failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise FuriosaApiError(f"POST {url} returned invalid JSON") from exc

        if not isinstance(result, dict):
            raise FuriosaApiError(f"POST {url} returned a non-object JSON response")
        return result

    def check_connection(self, endpoint: ModelEndpoint) -> ConnectionResult:
        url = self._models_url(endpoint.base_url)
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="GET",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            models = tuple(
                item["id"]
                for item in payload.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            )
            model_found = endpoint.model in models
            return ConnectionResult(
                endpoint=endpoint.name,
                url=url,
                expected_model=endpoint.model,
                available_models=models,
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=model_found,
                error=None if model_found else f"Expected model not found: {endpoint.model}",
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return ConnectionResult(
                endpoint=endpoint.name,
                url=url,
                expected_model=endpoint.model,
                available_models=(),
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
