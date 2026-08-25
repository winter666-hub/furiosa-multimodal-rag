"""Check connectivity to all configured Furiosa model servers."""

from __future__ import annotations

from furiosa_rag.clients import FuriosaClient
from furiosa_rag.config import Settings


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    endpoints = tuple(endpoint for endpoint in settings.endpoints if endpoint.enabled)
    if not endpoints:
        print("No Furiosa endpoints configured.")
        return 2

    client = FuriosaClient(settings.api_key, settings.request_timeout)
    results = [client.check_connection(endpoint) for endpoint in endpoints]

    for result in results:
        status = "OK" if result.ok else "FAIL"
        models = ", ".join(result.available_models) or "-"
        print(
            f"[{status}] {result.endpoint:<9} {result.latency_ms:8.1f} ms "
            f"url={result.url} models={models}"
        )
        if result.error:
            print(f"       error={result.error}")

    passed = sum(result.ok for result in results)
    print(f"\nConnection check: {passed}/{len(results)} endpoints reachable")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
