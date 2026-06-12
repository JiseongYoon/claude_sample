"""Fetcher/search seam + the GuardedFetcher policy layer + URL/SSRF containment.

Two layers, so SSRF policy can never be forgotten per fetcher:

  * **`Fetcher`** — the narrow *raw* interface a concrete fetcher implements (httpx in ). It
    performs ONE request against an already-contained URL and does **not** follow redirects (it
    returns the 3xx status + `location` so the policy layer can re-check the next hop). It raises the
    typed `BrowserError`s — never a raw library exception that could embed an internal host/IP.
  * **`GuardedFetcher`** — wraps a `Fetcher` and enforces policy on every call: `contain_url` the
    target, run the raw `get`, and on a redirect **re-contain the next hop** and loop (bounded). It
    enforces the response size cap and maps any non-typed exception to `BrowserUnavailable`. Tools
    talk only to this; they never hold a raw fetcher, so there is no un-contained path.

The 2026-06-01 verification established that with a real browser SSRF cannot be enforced in
Python (chromium owns DNS/redirects/subresources). v1 uses an HTTP client, where we DO own every
request and redirect — so `contain_url` is the load-bearing control and the redirect loop lives here
in the policy layer, re-checking every hop. `contain_url` checks the **resolved IP** (not the
hostname string), defeating alternative IP encodings; it returns the resolved IPs so can pin
the connection (defeating DNS rebinding).
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit


# --------------------------------------------------------------------------- #
# value types + typed errors (never leak an internal host/IP in a message)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContainedUrl:
    """A URL that passed `contain_url`. `ips` are the resolved, validated-public IPs — pins
    the connection to one of them so a re-resolve cannot redirect to an internal address."""

    url: str
    scheme: str
    host: str
    ips: tuple[str, ...]


@dataclass(frozen=True)
class RawFetch:
    """A single raw response from a `Fetcher` (no redirect following). On a 3xx, `location` carries
    the (possibly relative) redirect target so the policy layer can re-contain it."""

    status: int
    content_type: str = ""
    body: bytes = b""
    location: str | None = None


@dataclass(frozen=True)
class FetchResult:
    """A fully fetched, contained, size-checked response returned by `GuardedFetcher.fetch`."""

    url: str
    status: int
    content_type: str
    body: bytes


@dataclass(frozen=True)
class SearchHit:
    """One search result. `url` is NOT yet contained — it is contained when fetched."""

    title: str
    url: str
    snippet: str = ""


class BrowserError(Exception):
    """Base for all browser/web failures."""


class BrowserBlocked(BrowserError):
    """URL refused by containment (bad scheme, userinfo, allowlist miss, or a non-public address)."""


class BrowserNotFound(BrowserError):
    """The resource does not exist (e.g. HTTP 404 / NXDOMAIN), surfaced typed by a concrete fetcher."""


class BrowserTooLarge(BrowserError):
    """A response exceeds the configured byte cap."""


class BrowserTimeout(BrowserError):
    """A fetch exceeded its time budget."""


class BrowserUnavailable(BrowserError):
    """The fetcher/search backend is unreachable or errored (message carries only the exception type)."""


# --------------------------------------------------------------------------- #
# the raw seams (implemented by concrete fetchers / search backends)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Fetcher(Protocol):
    """One request against an already-contained URL; does NOT follow redirects. Concrete impls raise
    the typed `BrowserError`s (never a raw lib exception that could embed internal host/IP detail)."""

    async def get(self, contained: ContainedUrl, max_bytes: int, timeout: float) -> RawFetch: ...
    async def close(self) -> None: ...


@runtime_checkable
class SearchBackend(Protocol):
    """A search query against an operator-configured (trusted) endpoint → contained-later hits."""

    async def search(self, query: str, count: int) -> list[SearchHit]: ...
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# URL / SSRF containment — the security crux
# --------------------------------------------------------------------------- #
Resolver = Callable[[str], list[str]] # hostname -> list of IP strings (injectable; default = socket)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


def _socket_resolve(host: str) -> list[str]:
    """Default resolver. `getaddrinfo` canonicalises alternative IP encodings (octal/hex/decimal/
    dword) and resolves DNS names, returning real numeric IPs we then range-check."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


# Extra non-public ranges Python's `ipaddress` flags do NOT classify as private/reserved but which
# are not safe public-internet targets: RFC 6598 CGNAT shared address space, and the RFC 2544
# benchmarking range. (Flagged by the verification.)
_EXTRA_BLOCKED_NETS = (
    ipaddress.ip_network("100.64.0.0/10"), # RFC 6598 carrier-grade NAT
    ipaddress.ip_network("198.18.0.0/15"), # RFC 2544 benchmarking
)


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any non-public address. IPv4-mapped IPv6 is unwrapped so `::ffff:127.0.0.1` is caught
    via its IPv4 form. Covers loopback / link-local (incl. 169.254.169.254) / private / multicast /
    reserved / unspecified, plus CGNAT + benchmarking ranges the stdlib flags miss."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _ip_is_blocked(ip.ipv4_mapped)
    if ip.version == 4 and any(ip in net for net in _EXTRA_BLOCKED_NETS):
        return True
    return bool(
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def contain_url(
    url: str,
    *,
    allow_hosts: Iterable[str] | None = None,
    resolve: Resolver | None = None,
) -> ContainedUrl:
    """Validate `url` for SSRF safety. http/https only; no URL userinfo; if `allow_hosts` is given the
    host must match exactly; then the host is resolved (IP literals checked directly) and EVERY
    resolved IP must be public — else `BrowserBlocked`. Returns the resolved IPs for connection pinning.
    Default-deny: a malformed/unparseable URL or an unresolvable host is refused."""
    if not isinstance(url, str) or not url.strip():
        raise BrowserBlocked("url must be a non-empty string")
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise BrowserBlocked(f"malformed url: {type(exc).__name__}") from exc

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise BrowserBlocked(f"scheme not allowed: {scheme!r}")
    if "@" in parts.netloc:
        raise BrowserBlocked("userinfo in URL is not allowed")
    try:
        _ = parts.port # raises ValueError on a non-numeric/out-of-range port
    except ValueError as exc:
        raise BrowserBlocked(f"invalid port: {type(exc).__name__}") from exc

    host = parts.hostname # lowercased + de-bracketed by urlsplit
    if not host:
        raise BrowserBlocked("url has no host")

    if allow_hosts is not None:
        allowed = {h.lower() for h in allow_hosts}
        if allowed and host not in allowed:
            raise BrowserBlocked(f"host not in allowlist: {host!r}")

    # candidate IPs: a literal is checked directly; otherwise resolve (also canonicalises encodings)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    else:
        resolver = resolve or _socket_resolve
        try:
            raw_ips = resolver(host)
        except Exception as exc: # noqa: BLE001 — DNS/resolver failure → typed, no internal detail
            raise BrowserUnavailable(f"dns resolution failed: {type(exc).__name__}") from exc
        if not raw_ips:
            raise BrowserBlocked(f"host did not resolve: {host!r}")
        candidates = []
        for s in raw_ips:
            try:
                candidates.append(ipaddress.ip_address(s))
            except ValueError as exc:
                raise BrowserBlocked(f"resolver returned a non-IP: {type(exc).__name__}") from exc

    for ip in candidates:
        if _ip_is_blocked(ip):
            raise BrowserBlocked("host resolves to a non-public address")

    return ContainedUrl(
        url=url.strip(),
        scheme=scheme,
        host=host,
        ips=tuple(str(ip) for ip in candidates),
    )


# --------------------------------------------------------------------------- #
# the policy layer — the single contained path to a fetcher
# --------------------------------------------------------------------------- #
class GuardedFetcher:
    """Enforces URL containment + the per-hop redirect re-check + size cap around a `Fetcher`. Tools
    call only `fetch(url)`; they never hold a raw fetcher, so no request escapes containment."""

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        max_bytes: int = 5_000_000,
        per_fetch_timeout: float = 10.0,
        max_redirects: int = 5,
        allow_hosts: Iterable[str] | None = None,
        resolve: Resolver | None = None,
        name: str = "browser",
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if per_fetch_timeout <= 0:
            raise ValueError("per_fetch_timeout must be > 0")
        if max_redirects < 0:
            raise ValueError("max_redirects must be >= 0")
        self.name = name
        self._f = fetcher
        self._max_bytes = int(max_bytes)
        self._timeout = float(per_fetch_timeout)
        self._max_redirects = int(max_redirects)
        self._allow = frozenset(h.lower() for h in allow_hosts) if allow_hosts else None
        self._resolve = resolve

    def contain(self, url: str) -> ContainedUrl:
        """Public so can pre-validate a search-result URL before deciding to fetch it."""
        return contain_url(url, allow_hosts=self._allow, resolve=self._resolve)

    async def _guarded(self, fn, *args):
        try:
            return await fn(*args)
        except BrowserError:
            raise
        except Exception as exc: # noqa: BLE001 — never echo an internal host/IP from a lib exception
            raise BrowserUnavailable(f"fetch error: {type(exc).__name__}") from exc

    async def fetch(self, url: str) -> FetchResult:
        """Contain `url`, fetch it, and follow redirects in the policy layer — re-containing EVERY hop
        — bounded by `max_redirects`. Enforces the size cap on the final body."""
        current = url
        hops = 0
        while True:
            contained = self.contain(current) # re-contained on every hop (incl. redirects)
            resp = await self._guarded(self._f.get, contained, self._max_bytes, self._timeout)
            if resp.status in _REDIRECT_STATUS and resp.location:
                hops += 1
                if hops > self._max_redirects:
                    raise BrowserError(f"too many redirects (> {self._max_redirects})")
                current = urljoin(contained.url, resp.location)
                continue
            body = resp.body or b""
            if len(body) > self._max_bytes:
                raise BrowserTooLarge(f"response exceeds {self._max_bytes} bytes")
            return FetchResult(
                url=contained.url, status=resp.status, content_type=resp.content_type, body=body
            )

    async def close(self) -> None:
        await self._guarded(self._f.close)
