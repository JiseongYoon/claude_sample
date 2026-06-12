"""— browser gated tools (network-free, fakes injected).

`WebSearchTool` over a fake `SearchBackend`; `OpenUrlTool` over a fake `GuardedFetcher`-like object
(returns a `FetchResult`). Covers arg validation, count cap, text truncation, and graceful error
mapping (every `BrowserError` → `{"ok": False, ...}`).
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.browser.config import BrowserConfig
from local_ai_agent.modules.browser.fetcher import (
    BrowserBlocked,
    BrowserUnavailable,
    FetchResult,
    SearchHit,
)
from local_ai_agent.modules.browser.tools import (
    BROWSER_SAFE_TOOL_NAMES,
    OpenUrlTool,
    WebSearchTool,
    build_tools,
)

_HTML = (
    b"<html><head><title>T</title></head><body><article><p>"
    + b"Main body content paragraph with sufficient words to be kept by trafilatura. " * 3
    + b"</p></article></body></html>"
)


class FakeSearch:
    def __init__(self, *, hits=None, boom=None):
        self.hits = hits or []
        self.boom = boom
        self.calls = []

    async def search(self, query, count):
        self.calls.append((query, count))
        if self.boom is not None:
            raise self.boom
        return self.hits[:count]

    async def close(self): ...


class FakeFetcher:
    """Stands in for a GuardedFetcher: fetch(url) → FetchResult or raises a BrowserError."""

    def __init__(self, *, result=None, boom=None):
        self.result = result
        self.boom = boom
        self.fetched = []

    async def fetch(self, url):
        self.fetched.append(url)
        if self.boom is not None:
            raise self.boom
        return self.result


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_web_search_returns_results_capped():
    hits = [SearchHit(f"t{i}", f"https://x{i}.test", f"s{i}") for i in range(5)]
    t = WebSearchTool(FakeSearch(hits=hits), default_count=3)
    out = await t.run({"query": "hello"})
    assert out["ok"] is True and len(out["results"]) == 3
    assert out["results"][0] == {"title": "t0", "url": "https://x0.test", "snippet": "s0"}


@pytest.mark.asyncio
async def test_web_search_count_never_exceeds_default():
    fake = FakeSearch(hits=[SearchHit("a", "u", "s")])
    t = WebSearchTool(fake, default_count=2)
    await t.run({"query": "q", "count": 99})
    assert fake.calls[0][1] == 2 # capped to default_count


@pytest.mark.asyncio
async def test_web_search_bad_query():
    t = WebSearchTool(FakeSearch(), default_count=3)
    for bad in ({}, {"query": ""}, {"query": 5}):
        out = await t.run(bad)
        assert out["ok"] is False


@pytest.mark.asyncio
async def test_web_search_no_backend():
    out = await WebSearchTool(None, default_count=3).run({"query": "q"})
    assert out["ok"] is False and "not configured" in out["error"]


@pytest.mark.asyncio
async def test_web_search_error_graceful():
    t = WebSearchTool(FakeSearch(boom=BrowserUnavailable("search failed: ConnectError")), default_count=3)
    out = await t.run({"query": "q"})
    assert out["ok"] is False and "BrowserUnavailable" in out["error"]


# --------------------------------------------------------------------------- #
# open_url
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_open_url_fetches_and_extracts():
    fr = FetchResult(url="https://x.test/p", status=200, content_type="text/html", body=_HTML)
    t = OpenUrlTool(FakeFetcher(result=fr), max_text_chars=10_000)
    out = await t.run({"url": "https://x.test/p"})
    assert out["ok"] is True and out["status"] == 200 and out["url"] == "https://x.test/p"
    assert "Main body content" in out["text"] and out["truncated"] is False


@pytest.mark.asyncio
async def test_open_url_truncates_text():
    fr = FetchResult(url="https://x.test", status=200, content_type="text/html", body=_HTML)
    out = await OpenUrlTool(FakeFetcher(result=fr), max_text_chars=20).run({"url": "https://x.test"})
    assert out["ok"] is True and len(out["text"]) == 20 and out["truncated"] is True


@pytest.mark.asyncio
async def test_open_url_bad_arg():
    t = OpenUrlTool(FakeFetcher(), max_text_chars=100)
    for bad in ({}, {"url": ""}, {"url": 7}):
        assert (await t.run(bad))["ok"] is False


@pytest.mark.asyncio
async def test_open_url_blocked_graceful():
    t = OpenUrlTool(FakeFetcher(boom=BrowserBlocked("host resolves to a non-public address")), max_text_chars=100)
    out = await t.run({"url": "http://127.0.0.1/"})
    assert out["ok"] is False and "BrowserBlocked" in out["error"]


# --------------------------------------------------------------------------- #
# build_tools + safe-list
# --------------------------------------------------------------------------- #
def test_build_tools_names_and_safe_list():
    cfg = BrowserConfig(
        searxng_url=None, allow_hosts=(), max_bytes=1000, per_fetch_timeout=5.0,
        total_timeout=10.0, max_redirects=3, top_n=4, max_text_chars=500,
    )
    tools = build_tools(FakeFetcher(), FakeSearch(), config=cfg)
    names = {t.name for t in tools}
    assert names == {"web_search", "open_url"}
    assert "web_search" in BROWSER_SAFE_TOOL_NAMES
    assert "open_url" not in BROWSER_SAFE_TOOL_NAMES # gated via _NAV_TOOLS, never safe-listed
