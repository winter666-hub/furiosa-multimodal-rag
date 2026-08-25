"""Text generation backend interface and Furiosa implementation."""

from __future__ import annotations

from typing import Protocol

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint


class LlmBackend(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 64) -> str: ...


class FuriosaLlm:
    def __init__(self, endpoint: ModelEndpoint, client: FuriosaClient) -> None:
        self.endpoint = endpoint
        self.client = client

    def generate(self, prompt: str, *, max_tokens: int = 64) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        payload = self.client.post_json(
            self.endpoint.base_url,
            "chat/completions",
            {
                "model": self.endpoint.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FuriosaApiError("LLM response is missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise FuriosaApiError("LLM response content is not a string")
        return content
