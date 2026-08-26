from __future__ import annotations

import pytest

from furiosa_rag.web.app import (
    app,
    get_ask_concurrency_limiter,
    get_ask_rate_limiter,
    get_chat_log_repository,
    get_upload_concurrency_limiter,
    get_upload_rate_limiter,
)


@pytest.fixture(autouse=True)
def reset_demo_abuse_controls() -> None:
    cached_factories = (
        get_upload_rate_limiter,
        get_ask_rate_limiter,
        get_upload_concurrency_limiter,
        get_ask_concurrency_limiter,
        get_chat_log_repository,
    )
    for factory in cached_factories:
        factory.cache_clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    for factory in cached_factories:
        factory.cache_clear()
