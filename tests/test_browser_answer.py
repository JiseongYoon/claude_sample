"""— web_answer (search→fetch top-N→synthesize w/ citations), network-free.

Fake search + a per-URL fake fetcher (returns `FetchResult`s) + a fake `ChatModel`. Covers the
grounded-answer path with URL citations, zero-retrieval short-circuit (no model call), graceful
degrade (no chat / serving down / all pages blocked), and bounded skipping of failed pages.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.browser.config import BrowserConfig
from local_ai_agent.modules.browser.fetcher import (
    BrowserBlocked,
    BrowserError,
    FetchResult,
    SearchHit,
)
from local_ai_agent.modules.browser.tools import WebAnswerTool, build_tools
from local_ai_agent.modules.llm_serving import EngineNotReady


def _cfg(**over):
    base = dict(
        searxng_url="http://searx.local", allow_hosts=(), max_bytes=10_000, per_fetch_timeout=5.0,
        total_timeout=30.0, max_redirects=3, top_n=3, max_text_chars=5000,
        qa_top_k=5, qa_max_context_chars=12000, qa_answer_max_tokens=256, qa_chunk_chars=2000,
        qa_max_chunks=64,
    )
    base.update(over)
    return BrowserConfig(**base)


def _page(words: str) -> bytes:
    return f"<html><body><article><p>{words}</p></article></body></html>".encode("utf-8")


class FakeSearch:
    def __init__(self, hits=None, boom=None):
        self.hits = hits or []
        self.boom = boom
        self.calls = []

    async def search(self, query, count):
        self.calls.append((query, count))
        if self.boom:
            raise self.boom
        return self.hits[:count]

    async def close(self): ...


class MultiFetcher:
    """fetch(url) → FetchResult from a url→result map; a mapped BrowserError is raised; missing → blocked."""

    def __init__(self, by_url):
        self.by_url = by_url
        self.fetched = []

    async def fetch(self, url):
        self.fetched.append(url)
        r = self.by_url.get(url)
        if r is None:
            raise BrowserBlocked("not allowed")
        if isinstance(r, BrowserError):
            raise r
        return r

    async def close(self): ...


class FakeChat:
    def __init__(self, content="Paris is the capital of France.", boom=None):
        self.content = content
        self.boom = boom
        self.calls = []

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self.boom:
            raise self.boom
        return {"choices": [{"message": {"content": self.content}}]}


_Q = "what is the capital of France"
_RELEVANT = "The capital of France is Paris, a major European city on the Seine."
_IRRELEVANT = "Bananas are a yellow tropical fruit rich in potassium and grown in warm climates."


# --------------------------------------------------------------------------- #
# normal class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_web_answer_grounded_with_citations():
    hits = [SearchHit("A", "https://a.test", "sa"), SearchHit("B", "https://b.test", "sb")]
    fetcher = MultiFetcher({
        "https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT)),
        "https://b.test": FetchResult("https://b.test", 200, "text/html", _page(_IRRELEVANT)),
    })
    chat = FakeChat()
    tool = WebAnswerTool(fetcher, FakeSearch(hits=hits), chat, config=_cfg())
    out = await tool.run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is True
    assert "Paris" in out["answer"]
    assert any(c["url"] == "https://a.test" for c in out["citations"]) # the relevant page is cited
    assert len(out["pages"]) == 2 and chat.calls # both pages fetched; model was called
    assert fetcher.fetched == ["https://a.test", "https://b.test"]


@pytest.mark.asyncio
async def test_web_answer_accepts_question_alias():
    fetcher = MultiFetcher({"https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT))})
    tool = WebAnswerTool(fetcher, FakeSearch(hits=[SearchHit("A", "https://a.test", "")]), FakeChat(), config=_cfg())
    out = await tool.run({"question": _Q})
    assert out["ok"] is True and out["answer_found"] is True


@pytest.mark.asyncio
async def test_web_answer_no_answer_reply():
    fetcher = MultiFetcher({"https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT))})
    chat = FakeChat(content="NO_ANSWER")
    tool = WebAnswerTool(fetcher, FakeSearch(hits=[SearchHit("A", "https://a.test", "")]), chat, config=_cfg())
    out = await tool.run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is False and out["answer"] == ""


# --------------------------------------------------------------------------- #
# error / degrade class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_web_answer_no_chat_graceful():
    tool = WebAnswerTool(MultiFetcher({}), FakeSearch(), None, config=_cfg())
    out = await tool.run({"query": _Q})
    assert out["ok"] is False and "not available" in out["error"]


@pytest.mark.asyncio
async def test_web_answer_no_search_backend():
    tool = WebAnswerTool(MultiFetcher({}), None, FakeChat(), config=_cfg())
    assert (await tool.run({"query": _Q}))["ok"] is False


@pytest.mark.asyncio
async def test_web_answer_bad_query():
    tool = WebAnswerTool(MultiFetcher({}), FakeSearch(), FakeChat(), config=_cfg())
    for bad in ({}, {"query": ""}, {"query": 5}):
        assert (await tool.run(bad))["ok"] is False


@pytest.mark.asyncio
async def test_web_answer_no_hits_no_model_call():
    chat = FakeChat()
    tool = WebAnswerTool(MultiFetcher({}), FakeSearch(hits=[]), chat, config=_cfg())
    out = await tool.run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is False and out["pages"] == []
    assert chat.calls == []


@pytest.mark.asyncio
async def test_web_answer_all_pages_blocked_no_model_call():
    chat = FakeChat()
    hits = [SearchHit("A", "https://a.test", ""), SearchHit("B", "https://b.test", "")]
    fetcher = MultiFetcher({}) # every fetch → BrowserBlocked
    tool = WebAnswerTool(fetcher, FakeSearch(hits=hits), chat, config=_cfg())
    out = await tool.run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is False and chat.calls == []
    assert len(fetcher.fetched) == 2 # both attempted, both skipped


@pytest.mark.asyncio
async def test_web_answer_skips_failed_page_keeps_others():
    hits = [SearchHit("A", "https://a.test", ""), SearchHit("B", "https://b.test", "")]
    fetcher = MultiFetcher({
        "https://a.test": BrowserBlocked("internal"), # skipped
        "https://b.test": FetchResult("https://b.test", 200, "text/html", _page(_RELEVANT)),
    })
    out = await WebAnswerTool(fetcher, FakeSearch(hits=hits), FakeChat(), config=_cfg()).run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is True
    assert [p["url"] for p in out["pages"]] == ["https://b.test"]


@pytest.mark.asyncio
async def test_web_answer_zero_retrieval_no_model_call():
    # page has no overlap with the question → select_chunks empty → no model call, answer_found False
    chat = FakeChat()
    fetcher = MultiFetcher({"https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_IRRELEVANT))})
    out = await WebAnswerTool(fetcher, FakeSearch(hits=[SearchHit("A", "https://a.test", "")]), chat, config=_cfg()).run({"query": _Q})
    assert out["ok"] is True and out["answer_found"] is False and chat.calls == []


@pytest.mark.asyncio
async def test_web_answer_serving_unavailable_graceful():
    fetcher = MultiFetcher({"https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT))})
    chat = FakeChat(boom=EngineNotReady("engine not ready"))
    out = await WebAnswerTool(fetcher, FakeSearch(hits=[SearchHit("A", "https://a.test", "")]), chat, config=_cfg()).run({"query": _Q})
    assert out["ok"] is False # ServingUnavailable (QAError) → graceful


# --------------------------------------------------------------------------- #
# build_tools gating
# --------------------------------------------------------------------------- #
def test_build_tools_adds_web_answer_only_with_chat():
    fetcher, search, cfg = MultiFetcher({}), FakeSearch(), _cfg()
    assert {t.name for t in build_tools(fetcher, search, config=cfg)} == {"web_search", "open_url"}
    assert {t.name for t in build_tools(fetcher, search, config=cfg, chat=FakeChat())} == {
        "web_search", "open_url", "web_answer"}
