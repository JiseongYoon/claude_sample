"""— browser fetcher/search seam + URL/SSRF containment (network-free).

A fake `Fetcher` records every `ContainedUrl` it is asked to `get`, so tests can assert that a
blocked URL never reaches the raw fetcher. A fake resolver maps hostnames → IPs deterministically
(no real DNS). Covers `contain_url` (scheme/userinfo/IP-category/encoding/allowlist), the
`GuardedFetcher` per-hop redirect re-check + size cap + error mapping, and config.
"""
from __future__ import annotations

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.browser.config import BrowserConfig
from local_ai_agent.modules.browser.fetcher import (
    BrowserBlocked,
    BrowserError,
    BrowserTooLarge,
    BrowserUnavailable,
    ContainedUrl,
    FetchResult,
    Fetcher,
    GuardedFetcher,
    RawFetch,
    SearchHit,
    contain_url,
)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeFetcher:
    """Records the ContainedUrls it was asked to get; replays a scripted list of RawFetch (by call
    order) or a single RawFetch for all calls. `boom` makes get() raise (to test error mapping)."""

    def __init__(self, *, script=None, single=None, boom: Exception | None = None):
        self.calls: list[ContainedUrl] = []
        self._script = list(script) if script is not None else None
        self._single = single
        self._boom = boom
        self.closed = False

    async def get(self, contained: ContainedUrl, max_bytes: int, timeout: float) -> RawFetch:
        self.calls.append(contained)
        if self._boom is not None:
            raise self._boom
        if self._script is not None:
            return self._script[len(self.calls) - 1]
        return self._single if self._single is not None else RawFetch(status=200, body=b"ok")

    async def close(self) -> None:
        self.closed = True


def resolver(mapping: dict[str, list[str]]):
    """Build a fake resolver from host → IP list. Unknown host → empty (host did not resolve)."""

    def _resolve(host: str) -> list[str]:
        return list(mapping.get(host, []))

    return _resolve


PUBLIC = resolver({"example.com": ["93.184.216.34"], "good.test": ["8.8.8.8"]})

# Settings requires the model dirs; mirror the established test pattern (no real .env / files).
_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


# --------------------------------------------------------------------------- #
# contain_url — normal class
# --------------------------------------------------------------------------- #
def test_contain_public_ip_literal():
    c = contain_url("http://93.184.216.34/path?q=1")
    assert isinstance(c, ContainedUrl)
    assert c.scheme == "http" and c.host == "93.184.216.34"
    assert c.ips == ("93.184.216.34",)


def test_contain_public_hostname_resolved():
    c = contain_url("https://example.com/a", resolve=PUBLIC)
    assert c.scheme == "https" and c.host == "example.com"
    assert c.ips == ("93.184.216.34",)


def test_contain_https_and_http_both_ok():
    assert contain_url("http://good.test", resolve=PUBLIC).host == "good.test"
    assert contain_url("https://good.test", resolve=PUBLIC).host == "good.test"


def test_contain_allowlist_admits_listed_host():
    c = contain_url("https://example.com", allow_hosts=["example.com"], resolve=PUBLIC)
    assert c.host == "example.com"


def test_contain_public_ipv6_literal():
    c = contain_url("http://[2606:2800:220:1:248:1893:25c8:1946]/x")
    assert c.host == "2606:2800:220:1:248:1893:25c8:1946"


# --------------------------------------------------------------------------- #
# contain_url — error / security class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://127.0.0.1/",
        "data:text/html,<b>x</b>",
        "ftp://example.com/f",
        "ws://example.com/s",
    ],
)
def test_contain_rejects_non_http_scheme(url):
    with pytest.raises(BrowserBlocked):
        contain_url(url, resolve=PUBLIC)


def test_contain_rejects_userinfo():
    with pytest.raises(BrowserBlocked):
        contain_url("http://user:pass@example.com/", resolve=PUBLIC)


@pytest.mark.parametrize("url", ["", " ", "not a url", "http://", "https:///path"])
def test_contain_rejects_malformed(url):
    with pytest.raises(BrowserBlocked):
        contain_url(url, resolve=PUBLIC)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1", # loopback v4
        "10.0.0.5", # private
        "192.168.1.1", # private
        "172.16.0.1", # private
        "169.254.169.254", # link-local / cloud metadata
        "0.0.0.0", # unspecified
        "224.0.0.1", # multicast
    ],
)
def test_contain_blocks_internal_ip_literals(host):
    with pytest.raises(BrowserBlocked):
        contain_url(f"http://{host}/", resolve=PUBLIC)


@pytest.mark.parametrize("host", ["::1", "[::1]", "::ffff:127.0.0.1", "[::ffff:7f00:1]", "fc00::1", "fe80::1"])
def test_contain_blocks_internal_ipv6(host):
    # urlsplit needs IPv6 literals bracketed in a URL; accept both forms in the param for clarity
    h = host if host.startswith("[") else f"[{host}]"
    with pytest.raises(BrowserBlocked):
        contain_url(f"http://{h}/", resolve=PUBLIC)


@pytest.mark.parametrize("host", ["100.64.0.1", "100.127.255.254", "198.18.0.1", "198.19.255.1"])
def test_contain_blocks_cgnat_and_benchmark_ranges(host):
    # stdlib ipaddress does not flag these; the explicit extra-net check must catch them
    with pytest.raises(BrowserBlocked):
        contain_url(f"http://{host}/", resolve=PUBLIC)


def test_contain_blocks_hostname_resolving_internal():
    rb = resolver({"evil.test": ["127.0.0.1"]})
    with pytest.raises(BrowserBlocked):
        contain_url("http://evil.test/", resolve=rb)


def test_contain_blocks_dns_rebind_any_internal_ip():
    # multiple A records, one internal → blocked (all must be public)
    rb = resolver({"mix.test": ["8.8.8.8", "10.0.0.9"]})
    with pytest.raises(BrowserBlocked):
        contain_url("http://mix.test/", resolve=rb)


@pytest.mark.parametrize("host", ["2130706433", "0x7f000001", "0177.0.0.1"])
def test_contain_blocks_encoded_loopback_via_resolver(host):
    # getaddrinfo canonicalises these to 127.0.0.1; the fake resolver mirrors that
    rb = resolver({host: ["127.0.0.1"]})
    with pytest.raises(BrowserBlocked):
        contain_url(f"http://{host}/", resolve=rb)


def test_contain_unresolvable_host_blocked():
    with pytest.raises(BrowserBlocked):
        contain_url("http://nx.test/", resolve=resolver({}))


def test_contain_resolver_failure_unavailable():
    def boom(host: str):
        raise OSError("resolver down")

    with pytest.raises(BrowserUnavailable):
        contain_url("http://x.test/", resolve=boom)


def test_contain_allowlist_miss_blocked():
    with pytest.raises(BrowserBlocked):
        contain_url("https://other.test", allow_hosts=["example.com"], resolve=resolver({"other.test": ["8.8.8.8"]}))


# --------------------------------------------------------------------------- #
# GuardedFetcher — normal class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_fetch_200_returns_result():
    f = FakeFetcher(single=RawFetch(status=200, content_type="text/html", body=b"<h1>hi</h1>"))
    g = GuardedFetcher(f, resolve=PUBLIC)
    r = await g.fetch("https://example.com/p")
    assert isinstance(r, FetchResult)
    assert r.status == 200 and r.body == b"<h1>hi</h1>" and r.url == "https://example.com/p"
    assert len(f.calls) == 1 and f.calls[0].ips == ("93.184.216.34",)


@pytest.mark.asyncio
async def test_fetch_follows_bounded_redirect_to_public():
    script = [
        RawFetch(status=302, location="https://good.test/final"),
        RawFetch(status=200, body=b"done"),
    ]
    f = FakeFetcher(script=script)
    g = GuardedFetcher(f, resolve=PUBLIC, max_redirects=5)
    r = await g.fetch("https://example.com/start")
    assert r.body == b"done"
    # both hops were contained before being fetched
    assert [c.host for c in f.calls] == ["example.com", "good.test"]


@pytest.mark.asyncio
async def test_close_delegates():
    f = FakeFetcher()
    await GuardedFetcher(f, resolve=PUBLIC).close()
    assert f.closed is True


# --------------------------------------------------------------------------- #
# GuardedFetcher — error / security class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_fetch_blocked_url_never_calls_raw():
    f = FakeFetcher()
    g = GuardedFetcher(f, resolve=PUBLIC)
    with pytest.raises(BrowserBlocked):
        await g.fetch("http://127.0.0.1/admin")
    assert f.calls == [] # raw fetcher never touched


@pytest.mark.asyncio
async def test_redirect_to_internal_refused_on_hop():
    rb = resolver({"example.com": ["93.184.216.34"]}) # internal target won't resolve public
    f = FakeFetcher(script=[RawFetch(status=302, location="http://169.254.169.254/latest/meta-data/")])
    g = GuardedFetcher(f, resolve=rb)
    with pytest.raises(BrowserBlocked):
        await g.fetch("https://example.com/redir")
    # only the first (public) hop reached the raw fetcher; the internal hop was refused before get()
    assert len(f.calls) == 1 and f.calls[0].host == "example.com"


@pytest.mark.asyncio
async def test_too_many_redirects():
    # always redirect back to a public host → exceeds max_redirects
    f = FakeFetcher(single=RawFetch(status=307, location="https://example.com/loop"))
    g = GuardedFetcher(f, resolve=PUBLIC, max_redirects=3)
    with pytest.raises(BrowserError):
        await g.fetch("https://example.com/loop")
    assert len(f.calls) == 4 # initial + 3 redirects, then refused


@pytest.mark.asyncio
async def test_oversize_body_too_large():
    f = FakeFetcher(single=RawFetch(status=200, body=b"x" * 100))
    g = GuardedFetcher(f, resolve=PUBLIC, max_bytes=10)
    with pytest.raises(BrowserTooLarge):
        await g.fetch("https://example.com/big")


@pytest.mark.asyncio
async def test_raw_exception_mapped_to_unavailable_no_leak():
    f = FakeFetcher(boom=ConnectionResetError("connect to 10.1.2.3:6379 failed: secret-host"))
    g = GuardedFetcher(f, resolve=PUBLIC)
    with pytest.raises(BrowserUnavailable) as ei:
        await g.fetch("https://example.com/x")
    msg = str(ei.value)
    assert "10.1.2.3" not in msg and "secret-host" not in msg and "ConnectionResetError" in msg


@pytest.mark.asyncio
async def test_typed_error_passes_through():
    f = FakeFetcher(boom=BrowserTooLarge("nope"))
    g = GuardedFetcher(f, resolve=PUBLIC)
    with pytest.raises(BrowserTooLarge):
        await g.fetch("https://example.com/x")


@pytest.mark.parametrize(
    "kw",
    [{"max_bytes": 0}, {"max_bytes": -1}, {"per_fetch_timeout": 0}, {"per_fetch_timeout": -2.0}, {"max_redirects": -1}],
)
def test_guarded_fetcher_rejects_bad_bounds(kw):
    with pytest.raises(ValueError):
        GuardedFetcher(FakeFetcher(), resolve=PUBLIC, **kw)


# --------------------------------------------------------------------------- #
# config + data types
# --------------------------------------------------------------------------- #
def test_browser_config_from_settings_maps_fields():
    s = Settings(
        **_DIRS,
        browser_searxng_url="http://searx.local/search",
        browser_allow_hosts=["example.com", "good.test"],
        browser_max_bytes=1234,
        browser_per_fetch_timeout=7.5,
        browser_total_timeout=20.0,
        browser_max_redirects=2,
        browser_top_n=3,
    )
    c = BrowserConfig.from_settings(s)
    assert c.searxng_url == "http://searx.local/search"
    assert c.allow_hosts == ("example.com", "good.test")
    assert c.max_bytes == 1234 and c.per_fetch_timeout == 7.5 and c.total_timeout == 20.0
    assert c.max_redirects == 2 and c.top_n == 3


def test_browser_config_defaults_blocklist_mode():
    c = BrowserConfig.from_settings(Settings(**_DIRS))
    assert c.allow_hosts == () and c.searxng_url is None


@pytest.mark.parametrize(
    "kw",
    [
        {"browser_max_bytes": 0},
        {"browser_per_fetch_timeout": 0},
        {"browser_total_timeout": -1.0},
        {"browser_top_n": 0},
        {"browser_max_redirects": -1},
    ],
)
def test_settings_browser_validators_reject(kw):
    with pytest.raises(ValueError):
        Settings(**_DIRS, **kw)


def test_search_hit_and_protocols_present():
    h = SearchHit(title="T", url="https://x.test", snippet="s")
    assert h.title == "T" and h.url == "https://x.test"
    assert isinstance(FakeFetcher(), Fetcher) # runtime_checkable structural match
