"""Environment based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    name: str
    base_url: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    request_timeout: float
    endpoints: tuple[ModelEndpoint, ...]

    @classmethod
    def from_env(cls, dotenv_path: str | Path = ".env") -> "Settings":
        load_dotenv(dotenv_path)
        timeout = float(os.getenv("FURIOSA_REQUEST_TIMEOUT", "10"))
        if timeout <= 0:
            raise ValueError("FURIOSA_REQUEST_TIMEOUT must be greater than zero")

        endpoints = (
            ModelEndpoint(
                "llm",
                os.getenv(
                    "FURIOSA_LLM_BASE_URL", "https://endpoint.access.furiosa.dev/v1"
                ),
                os.getenv("FURIOSA_LLM_MODEL", "furiosa-ai/Qwen3-32B-FP8"),
            ),
            ModelEndpoint(
                "vision",
                os.getenv(
                    "FURIOSA_VISION_BASE_URL", "https://endpoint.access.furiosa.dev/v1"
                ),
                os.getenv("FURIOSA_VISION_MODEL", "furiosa-ai/Qwen3-VL-32B-Instruct"),
            ),
            ModelEndpoint(
                "embedding",
                os.getenv(
                    "FURIOSA_EMBEDDING_BASE_URL", "https://endpoint.access.furiosa.dev/v1"
                ),
                os.getenv("FURIOSA_EMBEDDING_MODEL", "furiosa-ai/Qwen3-Embedding-8B"),
            ),
            ModelEndpoint(
                "reranker",
                os.getenv(
                    "FURIOSA_RERANKER_BASE_URL", "https://endpoint.access.furiosa.dev/v1"
                ),
                os.getenv("FURIOSA_RERANKER_MODEL", "furiosa-ai/Qwen3-Reranker-8B"),
            ),
        )
        return cls(
            api_key=os.getenv("FURIOSA_API_KEY", "EMPTY"),
            request_timeout=timeout,
            endpoints=endpoints,
        )
