"""— httpx Fetcher (IP-pinned) + trafilatura extractor + SearXNG backend (network-free).

A fake httpx client records the request target / headers / extensions and replays canned responses
(incl. chunked bodies for the size cap). SearXNG is tested with canned JSON. Extraction runs against
real HTML strings. No real network / DNS.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import httpx
import pytest

from local_ai_agent.modules.browser.extract import ExtractResult, extract_main_text
from local_ai_agent.modules.browser.fetcher import (
    BrowserTimeout,
    BrowserTooLarge,
    BrowserUnavailable,
    ContainedUrl,
    RawFetch,
    SearchHit,
)
from local_ai_agent.modules.browser.transport import HttpxFetcher, SearxngBackend


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, *, status_code=200, headers=None, chunks=None, json_data=None, json_error=None):
        self.status_code = status_code
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._chunks = chunks if chunks is not None else [b""]
        self._json = json_data
        self._json_error = json_error
        self.closed = False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json

    async def aclose(self):
        self.closed = True


class FakeFetchClient:
    def __init__(self, *, response=None, send_error=None):
        self.requests = []
        self.send_args = None
        self._response = response
        self._send_error = send_error
        self.closed = False

    def build_request(self, method, url, *, headers=None, timeout=None, extensions=None):
        req = SimpleNamespace(
            method=method, url=url, headers=headers or {}, extensions=extensions or {}, timeout=timeout
        )
        self.requests.append(req)
        return req

    async def send(self, request, *, stream=False, follow_redirects=False):
        self.send_args = {"stream": stream, "follow_redirects": follow_redirects}
        if self._send_error is not None:
            raise self._send_error
        return self._response

    async def aclose(self):
        self.closed = True


class FakeSearchClient:
    def __init__(self, *, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error
        self.closed = False

    async def get(self, url, *, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        if self._error is not None:
            raise self._error
        return self._response

    async def aclose(self):
        self.closed = True


def contained(url="https://example.com/p", scheme="https", host="example.com", ips=("93.184.216.34",)):
    return ContainedUrl(url=url, scheme=scheme, host=host, ips=ips)


# --------------------------------------------------------------------------- #
# HttpxFetcher — normal class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_pins_ip_keeps_host_and_sni():
    resp = FakeResponse(status_code=200, headers={"Content-Type": "text/html"}, chunks=[b"<h1>", b"hi</h1>"])
    client = FakeFetchClient(response=resp)
    f = HttpxFetcher(client)
    r = await f.get(contained(), max_bytes=1000, timeout=5.0)
    assert isinstance(r, RawFetch) and r.status == 200 and r.body == b"<h1>hi</h1>"
    assert r.content_type == "text/html"
    req = client.requests[0]
    assert "93.184.216.34" in req.url and "example.com" not in req.url # URL host is the pinned IP
    assert req.headers["Host"] == "example.com"
    assert req.extensions.get("sni_hostname") == "example.com" # TLS SNI = original host
    assert client.send_args == {"stream": True, "follow_redirects": False}
    assert resp.closed is True


@pytest.mark.asyncio
async def test_get_http_sets_no_sni():
    client = FakeFetchClient(response=FakeResponse(status_code=200, chunks=[b"x"]))
    f = HttpxFetcher(client)
    await f.get(contained(url="http://example.com/", scheme="http"), max_bytes=10, timeout=5.0)
    assert client.requests[0].extensions == {}


@pytest.mark.asyncio
async def test_get_ipv6_pin_is_bracketed():
    ip = "2606:2800:220:1:248:1893:25c8:1946"
    client = FakeFetchClient(response=FakeResponse(status_code=200, chunks=[b"x"]))
    f = HttpxFetcher(client)
    await f.get(contained(ips=(ip,)), max_bytes=10, timeout=5.0)
    assert f"[{ip}]" in client.requests[0].url


@pytest.mark.asyncio
async def test_get_redirect_returns_location_no_body():
    resp = FakeResponse(status_code=302, headers={"Location": "https://good.test/next"}, chunks=[b"IGN"])
    client = FakeFetchClient(response=resp)
    r = await HttpxFetcher(client).get(contained(), max_bytes=1000, timeout=5.0)
    assert r.status == 302 and r.location == "https://good.test/next" and r.body == b""


@pytest.mark.asyncio
async def test_get_preserves_path_and_query_on_pin():
    client = FakeFetchClient(response=FakeResponse(status_code=200, chunks=[b"x"]))
    await HttpxFetcher(client).get(
        contained(url="https://example.com/a/b?x=1&y=2"), max_bytes=10, timeout=5.0
    )
    assert "/a/b" in client.requests[0].url and "x=1" in client.requests[0].url


# --------------------------------------------------------------------------- #
# HttpxFetcher — error / security class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_oversize_streamed_body_too_large():
    resp = FakeResponse(status_code=200, chunks=[b"a" * 6, b"b" * 6]) # 12 bytes
    f = HttpxFetcher(FakeFetchClient(response=resp))
    with pytest.raises(BrowserTooLarge):
        await f.get(contained(), max_bytes=10, timeout=5.0)
    assert resp.closed is True # still closed on the error path


@pytest.mark.asyncio
async def test_get_timeout_mapped():
    client = FakeFetchClient(send_error=httpx.ConnectTimeout("slow"))
    with pytest.raises(BrowserTimeout):
        await HttpxFetcher(client).get(contained(), max_bytes=10, timeout=1.0)


@pytest.mark.asyncio
async def test_get_connect_error_unavailable_no_leak():
    client = FakeFetchClient(send_error=httpx.ConnectError("connect to 10.1.2.3:6379 secret-host"))
    with pytest.raises(BrowserUnavailable) as ei:
        await HttpxFetcher(client).get(contained(), max_bytes=10, timeout=1.0)
    msg = str(ei.value)
    assert "10.1.2.3" not in msg and "secret-host" not in msg and "ConnectError" in msg


@pytest.mark.asyncio
async def test_close_owned_default_client_safe():
    # injected client is NOT owned → close() does not touch it
    client = FakeFetchClient(response=FakeResponse())
    f = HttpxFetcher(client)
    await f.close()
    assert client.closed is False


# --------------------------------------------------------------------------- #
# extract_main_text
# --------------------------------------------------------------------------- #
_HTML = (
    "<html><head><title>Doc Title</title></head><body><article>"
    "<h1>Main Heading</h1><p>This is the principal body paragraph that trafilatura should keep, "
    "with enough words to be treated as real content rather than boilerplate noise.</p>"
    "</article><nav>menu junk</nav><footer>copyright junk</footer></body></html>"
)


def test_extract_returns_title_and_text():
    r = extract_main_text(_HTML)
    assert isinstance(r, ExtractResult)
    assert "principal body paragraph" in r.text
    assert "menu junk" not in r.text and "copyright junk" not in r.text
    assert r.title # some non-empty title


def test_extract_from_bytes_with_charset():
    r = extract_main_text(_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
    assert "principal body paragraph" in r.text


def test_extract_empty_and_garbage_no_raise():
    assert extract_main_text("") == ExtractResult(title="", text="")
    assert extract_main_text(" ") == ExtractResult(title="", text="")
    r = extract_main_text("<<<not really html>>> &&&")
    assert isinstance(r, ExtractResult) # no crash


def test_extract_unknown_charset_falls_back():
    r = extract_main_text(_HTML.encode("utf-8"), content_type="text/html; charset=bogus-enc")
    assert "principal body paragraph" in r.text


def test_extract_missing_trafilatura_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", None) # import trafilatura → ImportError
    with pytest.raises(BrowserUnavailable):
        extract_main_text(_HTML)


# --------------------------------------------------------------------------- #
# SearxngBackend
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_search_maps_results_and_caps_count():
    data = {
        "results": [
            {"title": "A", "url": "https://a.test", "content": "snippet a"},
            {"title": "B", "url": "https://b.test", "content": "snippet b"},
            {"title": "C", "url": "https://c.test", "content": "snippet c"},
        ]
    }
    client = FakeSearchClient(response=FakeResponse(status_code=200, json_data=data))
    hits = await SearxngBackend("http://searx.local/search", client=client).search("q", count=2)
    assert len(hits) == 2 and hits[0] == SearchHit(title="A", url="https://a.test", snippet="snippet a")
    assert client.calls[0]["params"]["format"] == "json" and client.calls[0]["params"]["q"] == "q"


@pytest.mark.asyncio
async def test_search_tolerates_missing_fields():
    data = {"results": [{"url": "https://x.test"}, {"title": "T"}, "not-a-dict"]}
    hits = await SearxngBackend("http://s/search", client=FakeSearchClient(response=FakeResponse(json_data=data))).search("q", 10)
    assert hits[0] == SearchHit(title="", url="https://x.test", snippet="")
    assert hits[1] == SearchHit(title="T", url="", snippet="")
    assert len(hits) == 2 # the non-dict result is skipped


@pytest.mark.asyncio
async def test_search_count_zero_returns_empty():
    client = FakeSearchClient(response=FakeResponse(json_data={"results": [{"title": "A"}]}))
    assert await SearxngBackend("http://s", client=client).search("q", 0) == []
    assert client.calls == [] # never even queried


@pytest.mark.asyncio
async def test_search_non_200_unavailable():
    client = FakeSearchClient(response=FakeResponse(status_code=503, json_data={}))
    with pytest.raises(BrowserUnavailable):
        await SearxngBackend("http://s", client=client).search("q", 5)


@pytest.mark.asyncio
async def test_search_invalid_json_unavailable():
    client = FakeSearchClient(response=FakeResponse(status_code=200, json_error=ValueError("bad json")))
    with pytest.raises(BrowserUnavailable):
        await SearxngBackend("http://s", client=client).search("q", 5)


@pytest.mark.asyncio
async def test_search_timeout_mapped():
    client = FakeSearchClient(error=httpx.ReadTimeout("slow"))
    with pytest.raises(BrowserTimeout):
        await SearxngBackend("http://s", client=client).search("q", 5)


@pytest.mark.asyncio
async def test_search_connect_error_no_leak():
    client = FakeSearchClient(error=httpx.ConnectError("connect to 10.9.9.9 failed"))
    with pytest.raises(BrowserUnavailable) as ei:
        await SearxngBackend("http://s", client=client).search("q", 5)
    assert "10.9.9.9" not in str(ei.value)


@pytest.mark.asyncio
async def test_search_results_not_list_returns_empty():
    client = FakeSearchClient(response=FakeResponse(json_data={"results": "nope"}))
    assert await SearxngBackend("http://s", client=client).search("q", 5) == []


def test_searxng_rejects_empty_base_url():
    with pytest.raises(ValueError):
        SearxngBackend(" ")
