"""Small process-local abuse controls for the single-instance demo deployment.

These controls are intentionally process-local. They are not sufficient for
multi-instance production deployments; use a shared limiter such as Cloudflare
Rate Limiting, Durable Objects, or Redis for that environment.
"""

from __future__ import annotations

import hmac
import ipaddress
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request


def client_ip(request: Request, proxy_secret: str | None = None) -> str:
    """Return a trusted proxy IP or the direct Render peer address."""

    supplied_token = request.headers.get("x-paper-rag-proxy-token", "")
    supplied_ip = request.headers.get("x-paper-rag-client-ip", "")
    trusted_proxy = bool(
        proxy_secret
        and supplied_token
        and len(supplied_token) <= 1024
        and hmac.compare_digest(supplied_token, proxy_secret)
    )
    if trusted_proxy and len(supplied_ip) <= 128:
        try:
            return ipaddress.ip_address(supplied_ip.strip()).compressed
        except ValueError:
            pass
    host = request.client.host if request.client else "unknown"
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        return "unknown"


class RateLimiter:
    """Thread-safe sliding-window limiter with bounded stale-key retention."""

    def __init__(
        self,
        requests: int,
        window_seconds: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests <= 0 or window_seconds <= 0:
            raise ValueError("rate limit values must be greater than zero")
        self.requests = requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                return False, max(1, math.ceil(events[0] + self.window_seconds - now))
            events.append(now)
            if len(self._events) > 4096:
                self._events = defaultdict(
                    deque,
                    {item_key: item for item_key, item in self._events.items() if item and item[-1] > cutoff},
                )
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


class ConcurrencyLimiter:
    def __init__(self, maximum: int) -> None:
        if maximum <= 0:
            raise ValueError("maximum concurrency must be greater than zero")
        self._semaphore = threading.BoundedSemaphore(maximum)

    def acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()
