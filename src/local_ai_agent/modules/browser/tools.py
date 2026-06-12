"""Browser gated tools.

Two `Tool` adapters the agent reaches through the dispatcher:

  * **`web_search`** — query the configured SearXNG → result hits. Read-only + side-effect-free, so
    it is **safe-listed** at wiring time (`BROWSER_SAFE_TOOL_NAMES`).
  * **`open_url`** — fetch ONE URL (through `GuardedFetcher` → URL/SSRF contained) and return the
    extracted main text. The name matches the gate's `_NAV_TOOLS`, so it is auto-`needs_confirmation`
    (HITL on every outbound fetch) WITHOUT touching the security core.

Every known failure → a graceful `{"ok": False, "error": ...}`; the `BrowserError` message is already
internal-detail-scrubbed by the fetcher/containment layer. The agent never holds a raw fetcher or the
search backend — only these tools, behind the gate.
"""
from __future__ import annotations

import time
from typing import Any

from ..docqa.chunker import chunk_text
from ..docqa.qa import QAError, answer_question, sourced_chunks
from ..docqa.summarizer import ChatModel
from .config import BrowserConfig
from .extract import extract_main_text
from .fetcher import BrowserError, GuardedFetcher, SearchBackend

# read-only / side-effect-free tools → safe to allowlist on the gate ( wiring).
# `open_url` is intentionally NOT here: it is in the gate's `_NAV_TOOLS` → needs_confirmation.
BROWSER_SAFE_TOOL_NAMES = frozenset({"web_search"})


def _err(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bad(msg: str) -> dict:
    return {"ok": False, "error": f"ValueError: {msg}"}


class WebSearchTool:
    """`web_search` — SearXNG query → hits. Safe-listed (no side effects, no egress to a fetched page)."""

    name = "web_search"
    description = "Search the web via the configured search backend; returns result titles + URLs."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "count": {"type": "integer", "description": "Max number of results."},
        },
        "required": ["query"],
    }

    def __init__(self, search: SearchBackend | None, *, default_count: int) -> None:
        self._search = search
        self._default = int(default_count)

    async def run(self, args: dict) -> Any:
        if self._search is None:
            return _bad("search backend not configured")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _bad("'query' must be a non-empty string")
        count = args.get("count", self._default)
        if not isinstance(count, int) or count <= 0:
            count = self._default
        count = min(count, self._default) # never exceed the configured top_n
        try:
            hits = await self._search.search(query.strip(), count)
        except BrowserError as exc:
            return _err(exc)
        return {"ok": True, "results": [
            {"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits]}


class OpenUrlTool:
    """`open_url` — fetch one URL (contained) + extract main text. Gated (`_NAV_TOOLS` → confirm)."""

    name = "open_url"
    description = ("Fetch ONE web page by URL (contained) and extract its main text. Egress; "
                   "requires approval.")
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The http(s) URL to fetch."}},
        "required": ["url"],
    }

    def __init__(self, fetcher: GuardedFetcher, *, max_text_chars: int) -> None:
        self._f = fetcher
        self._max = int(max_text_chars)

    async def run(self, args: dict) -> Any:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return _bad("'url' must be a non-empty string")
        try:
            result = await self._f.fetch(url.strip())
            extracted = extract_main_text(result.body, content_type=result.content_type, url=result.url)
        except BrowserError as exc:
            return _err(exc)
        full = extracted.text
        text = full[: self._max]
        return {
            "ok": True,
            "url": result.url,
            "status": result.status,
            "title": extracted.title,
            "text": text,
            "truncated": len(full) > self._max,
        }


class WebAnswerTool:
    """`web_answer` — search → fetch top-N (contained) → synthesize a grounded answer with URL
    citations via llm-serving. Egress-heavy, so it is gated via
    the gate's `_NAV_TOOLS` (→ needs_confirmation): ONE confirm covers the whole bounded operation.
    Bounded by `top_n` (pages), `max_text_chars` (per page), and `total_timeout` (wall-clock)."""

    name = "web_answer"
    description = ("Search the web, fetch the top results (contained), and synthesize a grounded "
                   "answer with URL citations. Egress-heavy; requires approval.")
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The question / search query to answer."}},
        "required": ["query"],
    }

    def __init__(
        self,
        fetcher: GuardedFetcher,
        search: SearchBackend | None,
        chat: ChatModel | None,
        *,
        config: BrowserConfig,
    ) -> None:
        self._f = fetcher
        self._search = search
        self._chat = chat
        self._c = config

    async def run(self, args: dict) -> Any:
        if self._chat is None:
            return _bad("answer synthesis not available (llm-serving off)")
        if self._search is None:
            return _bad("search backend not configured")
        question = args.get("query", args.get("question"))
        if not isinstance(question, str) or not question.strip():
            return _bad("'query' must be a non-empty string")
        question = question.strip()
        try:
            hits = await self._search.search(question, self._c.top_n)
        except BrowserError as exc:
            return _err(exc)

        # fetch + extract each hit (contained), bounded by total wall-clock; failed pages are skipped.
        pages: list[tuple[str, str, str]] = [] # (url, title, text)
        deadline = time.monotonic() + self._c.total_timeout
        for hit in hits:
            if time.monotonic() > deadline:
                break
            try:
                result = await self._f.fetch(hit.url)
                extracted = extract_main_text(result.body, content_type=result.content_type, url=result.url)
            except BrowserError:
                continue # skip a page that is blocked/unreachable/too-large; keep the rest
            text = extracted.text.strip()
            if text:
                pages.append((result.url, extracted.title or hit.title, text[: self._c.max_text_chars]))

        if not pages:
            return {"ok": True, "answer_found": False, "answer": "", "citations": [], "pages": []}

        chunks = []
        for url, _title, text in pages:
            cr = chunk_text(text, chunk_chars=self._c.qa_chunk_chars, max_chunks=self._c.qa_max_chunks)
            chunks += sourced_chunks(url, cr)

        try:
            qa = await answer_question(
                question, chunks, chat=self._chat,
                top_k=self._c.qa_top_k, max_context_chars=self._c.qa_max_context_chars,
                max_answer_tokens=self._c.qa_answer_max_tokens,
            )
        except (QAError, ValueError) as exc: # ServingUnavailable (llm down) is a QAError → graceful
            return _err(exc)

        title_by_url = {url: title for url, title, _ in pages}
        citations, seen = [], set()
        for c in qa.citations:
            if c.source not in seen:
                seen.add(c.source)
                citations.append({"url": c.source, "title": title_by_url.get(c.source, "")})
        return {
            "ok": True,
            "answer_found": qa.answer_found,
            "answer": qa.answer,
            "citations": citations,
            "pages": [{"url": url, "title": title} for url, title, _ in pages],
        }


def build_tools(
    fetcher: GuardedFetcher,
    search: SearchBackend | None,
    *,
    config: BrowserConfig,
    chat: ChatModel | None = None,
) -> list:
    """The browser tools, in registration order (search → fetch → answer). `web_answer` is added
    only when a chat model is available (llm-serving on)."""
    tools = [
        WebSearchTool(search, default_count=config.top_n),
        OpenUrlTool(fetcher, max_text_chars=config.max_text_chars),
    ]
    if chat is not None:
        tools.append(WebAnswerTool(fetcher, search, chat, config=config))
    return tools
