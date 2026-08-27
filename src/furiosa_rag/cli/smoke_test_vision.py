"""Send one tiny synthetic PNG to the Direct Vision endpoint."""

from __future__ import annotations

import time

from furiosa_rag.cli.run_rag import _endpoint
from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import Settings
from furiosa_rag.vision import FuriosaVision

# A generated 32x32 white PNG. It is large enough for common VL preprocessors but only 95 bytes.
_TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJklEQVR4nO3NMQ0AAAwDoPo33arY"
    "sQQMkB6LQCAQCAQCgUAg+BIMi1X0pjxKe0gAAAAASUVORK5CYII="
)


def run_vision_smoke(settings: Settings, client: FuriosaClient) -> bool:
    endpoint = _endpoint(settings, "vision")
    connection = client.check_connection(endpoint)
    print(f"vision_endpoint_reachable={str(connection.ok).lower()}")
    print(f"model={endpoint.model}")
    if not connection.ok:
        print(f"error={connection.error or 'endpoint check failed'}")
        return False

    started = time.perf_counter()
    try:
        content = FuriosaVision(endpoint, client).analyze(
            "What color is this single-pixel image?",
            _TINY_PNG_DATA_URL,
            max_tokens=settings.vision_max_tokens,
        )
    except (FuriosaApiError, ValueError) as exc:
        print("vision_response_nonempty=false")
        print(f"error={exc}")
        return False
    print("vision_response_nonempty=true")
    print(f"latency_ms={(time.perf_counter() - started) * 1000:.1f}")
    print(f"response_preview={content[:120]!r}")
    return True


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2
    client = FuriosaClient(settings.api_key, settings.vision_request_timeout)
    return 0 if run_vision_smoke(settings, client) else 1


if __name__ == "__main__":
    raise SystemExit(main())
