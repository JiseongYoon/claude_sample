"""Integration: browser wired into the composition root.

Group A — real `build_application` under `enable_browser`: tool registration, gate classification
(`web_search` safe / `open_url` needs_confirmation / secret-or-internal path interplay), flag
independence, capability registration, and fault isolation (a real fetch to a blocked URL → graceful).
Group B — composed execution through the dispatcher with injected fakes: search→safe→runs,
open_url→confirm, blocked URL → graceful, default safe tools unchanged.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.browser.config import BrowserConfig
from local_ai_agent.modules.browser.fetcher import FetchResult, SearchHit
from local_ai_agent.modules.browser.tools import build_tools
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from tests.test_browser_tools import FakeFetcher, FakeSearch

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


def _app(**kw):
    return build_application(Settings(**_DIRS, enable_agent=True, enable_browser=True, **kw))


# --------------------------------------------------------------------------- #
# Group A — real build_application wiring
# --------------------------------------------------------------------------- #
def test_web_search_safe_listed():
    gate = _app().agent_runtime.gate
    assert gate.classify(Action("web_search", {"query": "hello"})).verdict is Verdict.safe


def test_open_url_needs_confirmation():
    gate = _app().agent_runtime.gate
    assert gate.classify(Action("open_url", {"url": "https://example.com"})).verdict is Verdict.needs_confirmation


def test_browser_tools_registered_on_dispatcher():
    disp = _app().agent_runtime.dispatcher
    # enable_browser builds serving (ChatModel) → web_answer is also present
    assert {"web_search", "open_url", "web_answer"}.issubset(set(disp._tools.keys()))


def test_web_answer_needs_confirmation_and_not_safe_listed():
    gate = _app().agent_runtime.gate
    assert gate.classify(Action("web_answer", {"query": "hi"})).verdict is Verdict.needs_confirmation


def test_browser_without_agent_registers_module_no_runtime():
    app = build_application(Settings(**_DIRS, enable_agent=False, enable_browser=True))
    assert app.agent_runtime is None
    known = {c for m in app.registry.modules for c in m.spec.capabilities}
    assert "browser" in known


def test_default_safe_tools_unchanged_with_browser():
    gate = _app().agent_runtime.gate
    assert gate.classify(Action("read_file", {"path": "x"})).verdict is Verdict.safe
    assert gate.classify(Action("write_file", {"path": "x"})).verdict is Verdict.needs_confirmation
    # browser flag did not accidentally safe-list open_url
    assert gate.classify(Action("open_url", {"url": "https://x.test"})).verdict is Verdict.needs_confirmation


async def test_fault_isolation_blocked_url_graceful():
    # real GuardedFetcher + HttpxFetcher; an internal URL is refused by containment → graceful, no crash
    disp = _app().agent_runtime.dispatcher
    action = Action("open_url", {"url": "http://127.0.0.1/admin"}) # open_url is gated → confirm first
    decision = await disp.dispatch(action, "t")
    assert decision.outcome is Outcome.pending
    disp.approvals.approve(decision.approval_id, action, _Approver())
    approved = await disp.execute_approved(decision.approval_id, action)
    assert approved.outcome is Outcome.executed
    assert approved.result["ok"] is False # BrowserBlocked → graceful {ok: False}


def test_browser_flag_off_no_browser_capability():
    app = build_application(Settings(**_DIRS, enable_agent=True))
    known = {c for m in app.registry.modules for c in m.spec.capabilities}
    assert "browser" not in known


# --------------------------------------------------------------------------- #
# Group B — composed execution through the dispatcher with injected fakes
# --------------------------------------------------------------------------- #
def _cfg():
    return BrowserConfig(
        searxng_url="http://searx.local", allow_hosts=(), max_bytes=1000, per_fetch_timeout=5.0,
        total_timeout=10.0, max_redirects=3, top_n=3, max_text_chars=500,
    )


def _composed(*, search_hits=None, fetch_result=None, fetch_boom=None):
    from local_ai_agent.modules.browser.tools import BROWSER_SAFE_TOOL_NAMES

    tools = build_tools(
        FakeFetcher(result=fetch_result, boom=fetch_boom),
        FakeSearch(hits=search_hits or []),
        config=_cfg(),
    )
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | BROWSER_SAFE_TOOL_NAMES)
    return ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=tools)


async def test_search_executes_via_dispatcher():
    disp = _composed(search_hits=[SearchHit("A", "https://a.test", "s")])
    res = await disp.dispatch(Action("web_search", {"query": "q"}), "t")
    assert res.outcome is Outcome.executed and res.result["ok"] is True
    assert res.result["results"][0]["url"] == "https://a.test"


async def test_open_url_confirm_then_execute():
    fr = FetchResult(url="https://a.test", status=200, content_type="text/html", body=b"<p>hi there world</p>")
    disp = _composed(fetch_result=fr)
    action = Action("open_url", {"url": "https://a.test"})
    decision = await disp.dispatch(action, "t")
    assert decision.outcome is Outcome.pending
    disp.approvals.approve(decision.approval_id, action, _Approver())
    approved = await disp.execute_approved(decision.approval_id, action)
    assert approved.outcome is Outcome.executed and approved.result["ok"] is True
