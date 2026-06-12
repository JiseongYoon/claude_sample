"""— hermetic regression (the phase completion gate).

Exercises the browser capability as an INTEGRATED whole against the user-chosen behavioral
mechanism (the "권장 기본 세트", 2026-06-01):

  * gate routing — web_search→safe, open_url→confirm, web_answer→confirm;
  * SSRF end-to-end — an internal URL is refused with NO real connection (literal IP, network-free);
  * web_answer grounded + citations (citations ⊆ fetched URLs) through confirm→approve→execute HITL;
  * zero-retrieval → not-found with no model call; graceful degrade (serving down) → {ok:False};
  * bounded — top_n caps fetched pages;
  * flag isolation — enable_browser off → no browser capability/tools.

Network-free: the real `GuardedFetcher`/`HttpxFetcher` only ever sees a literal internal IP (blocked
before any socket), and the grounded path uses injected fakes. No real model, no SearXNG.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.browser.config import BrowserConfig
from local_ai_agent.modules.browser.fetcher import GuardedFetcher, SearchHit
from local_ai_agent.modules.browser.tools import BROWSER_SAFE_TOOL_NAMES, build_tools
from local_ai_agent.modules.browser.transport import HttpxFetcher
from local_ai_agent.modules.llm_serving import EngineNotReady
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from tests.test_browser_answer import FakeChat, FakeSearch, MultiFetcher, _page, _Q, _RELEVANT
from local_ai_agent.modules.browser.fetcher import FetchResult

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


async def _approve_run(disp, action):
    res = await disp.dispatch(action, "stage5")
    if res.outcome is Outcome.pending:
        disp.approvals.approve(res.approval_id, action, _Approver())
        res = await disp.execute_approved(res.approval_id, action)
    return res


def _cfg(**over):
    base = dict(
        searxng_url="http://searx.local", allow_hosts=(), max_bytes=10_000, per_fetch_timeout=5.0,
        total_timeout=30.0, max_redirects=3, top_n=3, max_text_chars=5000,
        qa_top_k=5, qa_max_context_chars=12000, qa_answer_max_tokens=256, qa_chunk_chars=2000,
        qa_max_chunks=64,
    )
    base.update(over)
    return BrowserConfig(**base)


def _composed(*, fetcher, search, chat):
    tools = build_tools(fetcher, search, config=_cfg(), chat=chat)
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | BROWSER_SAFE_TOOL_NAMES)
    return ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=tools)


# --------------------------------------------------------------------------- #
# 1. gate routing over the full tool set (real build_application)
# --------------------------------------------------------------------------- #
def test_gate_routing_full_set():
    gate = build_application(Settings(**_DIRS, enable_agent=True, enable_browser=True)).agent_runtime.gate
    assert gate.classify(Action("web_search", {"query": "q"})).verdict is Verdict.safe
    assert gate.classify(Action("open_url", {"url": "https://x.test"})).verdict is Verdict.needs_confirmation
    assert gate.classify(Action("web_answer", {"query": "q"})).verdict is Verdict.needs_confirmation
    # the gate's own rules still bite on a secret/internal path arg
    assert gate.classify(Action("web_search", {"query": "x", "path": "/etc/shadow"})).verdict is Verdict.blocked


# --------------------------------------------------------------------------- #
# 2. SSRF end-to-end through the REAL dispatcher (literal IP → no connection)
# --------------------------------------------------------------------------- #
async def test_ssrf_open_url_blocked_end_to_end():
    disp = build_application(Settings(**_DIRS, enable_agent=True, enable_browser=True)).agent_runtime.dispatcher
    for url in ("http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/"):
        res = await _approve_run(disp, Action("open_url", {"url": url}))
        assert res.outcome is Outcome.executed and res.result["ok"] is False # contained, no real connection


# --------------------------------------------------------------------------- #
# 3. web_answer grounded + citations through confirm→approve→execute (fakes)
# --------------------------------------------------------------------------- #
async def test_web_answer_grounded_citations_hitl():
    hits = [SearchHit("A", "https://a.test", "sa"), SearchHit("B", "https://b.test", "sb")]
    fetcher = MultiFetcher({
        "https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT)),
        "https://b.test": FetchResult("https://b.test", 200, "text/html", _page("unrelated potassium fruit text")),
    })
    disp = _composed(fetcher=fetcher, search=FakeSearch(hits=hits), chat=FakeChat())
    res = await _approve_run(disp, Action("web_answer", {"query": _Q}))
    assert res.outcome is Outcome.executed and res.result["ok"] is True
    assert res.result["answer_found"] is True and "Paris" in res.result["answer"]
    fetched = {p["url"] for p in res.result["pages"]}
    assert {c["url"] for c in res.result["citations"]} <= fetched # citations ⊆ fetched URLs


# --------------------------------------------------------------------------- #
# 4. web_answer SSRF — internal search hit refused, no connection, graceful
# --------------------------------------------------------------------------- #
async def test_web_answer_internal_hit_skipped_no_connection():
    # REAL GuardedFetcher: a literal internal IP is blocked before any socket
    real_fetcher = GuardedFetcher(HttpxFetcher(), max_bytes=10_000, per_fetch_timeout=2.0, max_redirects=2)
    search = FakeSearch(hits=[SearchHit("evil", "http://127.0.0.1/secret", "")])
    disp = _composed(fetcher=real_fetcher, search=search, chat=FakeChat())
    res = await _approve_run(disp, Action("web_answer", {"query": _Q}))
    assert res.outcome is Outcome.executed and res.result["ok"] is True
    assert res.result["answer_found"] is False and res.result["pages"] == [] # internal page skipped


# --------------------------------------------------------------------------- #
# 5. graceful degrade — serving down → {ok:False}; search/fetch still work
# --------------------------------------------------------------------------- #
async def test_web_answer_serving_down_graceful_but_search_ok():
    fetcher = MultiFetcher({"https://a.test": FetchResult("https://a.test", 200, "text/html", _page(_RELEVANT))})
    chat = FakeChat(boom=EngineNotReady("down"))
    disp = _composed(fetcher=fetcher, search=FakeSearch(hits=[SearchHit("A", "https://a.test", "")]), chat=chat)
    # web_answer degrades gracefully...
    res = await _approve_run(disp, Action("web_answer", {"query": _Q}))
    assert res.outcome is Outcome.executed and res.result["ok"] is False
    # ...but web_search (llm-independent) still serves
    s = await disp.dispatch(Action("web_search", {"query": "q"}), "stage5")
    assert s.outcome is Outcome.executed and s.result["ok"] is True


# --------------------------------------------------------------------------- #
# 6. bounded — top_n caps fetched pages
# --------------------------------------------------------------------------- #
async def test_web_answer_bounded_by_top_n():
    hits = [SearchHit(f"t{i}", f"https://x{i}.test", "") for i in range(10)]
    fetcher = MultiFetcher({f"https://x{i}.test": FetchResult(f"https://x{i}.test", 200, "text/html", _page(_RELEVANT)) for i in range(10)})
    search = FakeSearch(hits=hits)
    disp = _composed(fetcher=fetcher, search=search, chat=FakeChat())
    await _approve_run(disp, Action("web_answer", {"query": _Q}))
    assert search.calls[0][1] == 3 # asked for top_n=3
    assert len(fetcher.fetched) <= 3 # never fetched more than top_n


# --------------------------------------------------------------------------- #
# 7. flag isolation — browser off → no capability/tools
# --------------------------------------------------------------------------- #
def test_flag_off_no_browser():
    app = build_application(Settings(**_DIRS, enable_agent=True))
    known = {c for m in app.registry.modules for c in m.spec.capabilities}
    assert "browser" not in known
    disp = app.agent_runtime.dispatcher
    for name in ("web_search", "open_url", "web_answer"):
        assert name not in disp._tools
