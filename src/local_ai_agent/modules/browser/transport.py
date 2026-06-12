"""Concrete httpx transports behind the seam.

  * **`HttpxFetcher`** — a `Fetcher` doing ONE request, **pinned to the IP `contain_url` already
    validated** (URL host → IP, `Host` header + TLS SNI = original host), with auto-redirect OFF so
    `GuardedFetcher` re-contains every hop. Pinning closes the DNS-rebinding TOCTOU window (the host
    cannot re-resolve to an internal IP between the containment check and the connection).
  * **`SearxngBackend`** — a `SearchBackend` querying an operator-configured SearXNG (`format=json`).
    The endpoint is operator config (trusted, often localhost) so it is deliberately NOT contained.

httpx is a core dependency. Both transports accept an injected, duck-typed client so tests run with
no real network. All failures surface as typed `BrowserError`s carrying only the exception type name
(never an internal IP/host/url/secret).
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

import httpx

from .fetcher import (
    _REDIRECT_STATUS,
    BrowserError,
    BrowserTimeout,
    BrowserUnavailable,
    BrowserTooLarge,
    ContainedUrl,
    RawFetch,
    SearchHit,
)

_DEFAULT_UA = "local-ai-agent/0.0 (+https://localhost) browser-capability"


def _pin_target(contained: ContainedUrl) -> tuple[str, str, str]:
    """Rewrite the contained URL's host to its validated IP, returning (target_url, host_header,
    sni_hostname). The Host header + SNI keep vhost routing and TLS cert verification correct."""
    parts = urlsplit(contained.url)
    ip = contained.ips[0]
    ip_host = f"[{ip}]" if ":" in ip else ip
    port = parts.port
    ip_netloc = f"{ip_host}:{port}" if port else ip_host
    host_header = parts.netloc # original host[:port]; userinfo already rejected by contain_url
    target = urlunsplit((parts.scheme, ip_netloc, parts.path or "/", parts.query, ""))
    return target, host_header, parts.hostname or ""


class HttpxFetcher:
    """A single-request, IP-pinned, non-redirecting `Fetcher`. Inject `client` (duck-typed
    `build_request`/`send`/`aclose`) in tests; otherwise a default `httpx.AsyncClient` is created."""

    def __init__(self, client=None, *, user_agent: str = _DEFAULT_UA) -> None:
        self._client = client
        self._own_client = client is None
        self._ua = user_agent

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False, verify=True)
        return self._client

    async def get(self, contained: ContainedUrl, max_bytes: int, timeout: float) -> RawFetch:
        client = self._ensure_client()
        target, host_header, sni = _pin_target(contained)
        headers = {"Host": host_header, "User-Agent": self._ua, "Accept": "text/html,*/*;q=0.8"}
        extensions = {"sni_hostname": sni} if contained.scheme == "https" else {}
        try:
            request = client.build_request(
                "GET", target, headers=headers, timeout=httpx.Timeout(timeout), extensions=extensions
            )
            response = await client.send(request, stream=True, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise BrowserTimeout(f"fetch timed out: {type(exc).__name__}") from exc
        except BrowserError:
            raise
        except Exception as exc: # noqa: BLE001 — never echo target IP/host/url
            raise BrowserUnavailable(f"fetch failed: {type(exc).__name__}") from exc

        try:
            status = int(response.status_code)
            ctype = response.headers.get("content-type", "") or ""
            location = response.headers.get("location")
            if status in _REDIRECT_STATUS and location:
                return RawFetch(status=status, content_type=ctype, body=b"", location=location)
            body = await self._read_capped(response, max_bytes)
            return RawFetch(status=status, content_type=ctype, body=body, location=None)
        except httpx.TimeoutException as exc:
            raise BrowserTimeout(f"read timed out: {type(exc).__name__}") from exc
        except BrowserError:
            raise
        except Exception as exc: # noqa: BLE001
            raise BrowserUnavailable(f"read failed: {type(exc).__name__}") from exc
        finally:
            await response.aclose()

    @staticmethod
    async def _read_capped(response, max_bytes: int) -> bytes:
        total = 0
        chunks: list[bytes] = []
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise BrowserTooLarge(f"response exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    async def close(self) -> None:
        if self._client is not None and self._own_client:
            try:
                await self._client.aclose()
            except Exception as exc: # noqa: BLE001
                raise BrowserUnavailable(f"close failed: {type(exc).__name__}") from exc


class SearxngBackend:
    """Query an operator-configured SearXNG instance (`format=json`). The endpoint is trusted config
    (not agent-controlled), so it is not contained. Inject `client` in tests."""

    def __init__(self, base_url: str, *, client=None, timeout: float = 10.0) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("SearXNG base_url must be a non-empty string")
        self._url = base_url.strip()
        self._client = client
        self._own_client = client is None
        self._timeout = float(timeout)

    def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False, verify=True)
        return self._client

    async def search(self, query: str, count: int) -> list[SearchHit]:
        if count <= 0:
            return []
        client = self._ensure_client()
        params = {"q": query, "format": "json"}
        try:
            response = await client.get(self._url, params=params, timeout=httpx.Timeout(self._timeout))
            status = int(response.status_code)
            if status != 200:
                raise BrowserUnavailable(f"search backend returned status {status}")
            data = response.json()
        except httpx.TimeoutException as exc:
            raise BrowserTimeout(f"search timed out: {type(exc).__name__}") from exc
        except BrowserError:
            raise
        except Exception as exc: # noqa: BLE001 — invalid JSON / connect error → typed, no leak
            raise BrowserUnavailable(f"search failed: {type(exc).__name__}") from exc

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        hits: list[SearchHit] = []
        for item in results[:count]:
            if not isinstance(item, dict):
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title", "") or ""),
                    url=str(item.get("url", "") or ""),
                    snippet=str(item.get("content", "") or ""),
                )
            )
        return hits

    async def close(self) -> None:
        if self._client is not None and self._own_client:
            try:
                await self._client.aclose()
            except Exception as exc: # noqa: BLE001
                raise BrowserUnavailable(f"close failed: {type(exc).__name__}") from exc
