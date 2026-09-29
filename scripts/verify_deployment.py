#!/usr/bin/env python3
"""Check a running Kronos web deployment without changing server state."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def get_response(url: str) -> tuple[int, bytes]:
    try:
        with urlopen(url, timeout=10) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, b""
    except URLError as exc:
        raise SystemExit(f"deployment is unreachable: {exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Kronos FastAPI deployment")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health_status, health_body = get_response(f"{base_url}/health")
    try:
        health = json.loads(health_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit("health check did not return JSON") from exc
    if health_status != 200 or health.get("status") != "ok":
        raise SystemExit(f"health check failed: HTTP {health_status} {health}")

    root_status, _ = get_response(f"{base_url}/")
    if root_status != 200:
        raise SystemExit(f"SPA root check failed: HTTP {root_status}")

    api_status, api_body = get_response(f"{base_url}/api/v1/index/list")
    try:
        symbols = json.loads(api_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit("API check did not return JSON") from exc
    if api_status != 200 or not isinstance(symbols, (dict, list)):
        raise SystemExit(f"API check failed: HTTP {api_status}")

    research_status, _ = get_response(f"{base_url}/api/v1/admin/research/summary")
    if research_status != 401:
        raise SystemExit(f"admin research access-control check failed: HTTP {research_status}")

    print("deployment verification OK")
    print(f"health version={health.get('version')} uptime={health.get('uptime_seconds')}s")
    print("admin research API access control=OK")


if __name__ == "__main__":
    main()
