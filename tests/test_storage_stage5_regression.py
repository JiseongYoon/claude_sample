"""regression — storage capability as an integrated whole (hermetic).

The phase-completion gate: drive the FULL composed tool set (6 file tools + 2 remote→DocQA tools)
through one gated dispatcher — the same composition `build_application` wires under
`enable_agent + enable_storage + enable_docqa` — over a fake `RemoteTransport` connector + a fake
`ChatModel` (no network, no model; real-protocol/real-model are the SSH loopback + real-model
smokes in `scripts/smoke_storage.py`). Confirms gate routing (read→safe, mutation→confirm,
secret→blocked), containment, read-only, remote→DocQA, and fault isolation across the whole set.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from local_ai_agent.modules.docqa.tools import DocQAConfig
from local_ai_agent.modules.llm_serving import EngineNotReady
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from local_ai_agent.modules.storage.connector import GuardedConnector
from local_ai_agent.modules.storage.remote_docqa import REMOTE_DOCQA_SAFE_TOOL_NAMES, build_remote_tools
from local_ai_agent.modules.storage.tools import STORAGE_SAFE_TOOL_NAMES, build_tools
from tests.test_docqa_loaders import _make_pdf
from tests.test_storage_connector import FakeTransport

_ROOT = "/srv/share"


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


def _resp(c):
    return {"choices": [{"message": {"content": c}}]}


class FakeChat:
    def __init__(self, *, answer="SUMMARY", raise_exc=None):
        self.calls, self._a, self._raise = [], answer, raise_exc

    async def chat(self, messages, **p):
        self.calls.append((messages, p))
        if self._raise:
            raise self._raise
        return _resp(self._a)


def _cfg():
    return DocQAConfig(
        docs_root=Path("/unused"), max_doc_bytes=10_000_000, summarize_chunk_chars=2000,
        qa_chunk_chars=500, overlap_ratio=0.1, max_chunks=64, summary_max_tokens=256,
        reduce_max_passes=5, qa_top_k=5, qa_max_context_chars=4000, answer_max_tokens=256,
        max_docs_per_query=20, max_total_chunks=128,
    )


def _composed(*, read_only=False, chat=None):
    """Mirror build_application's storage+remote branch with injectable fakes."""
    t = FakeTransport(
        files={f"{_ROOT}/doc.txt": b"alpha remote content discussing revenue figures.",
               f"{_ROOT}/report.pdf": _make_pdf("Hello PDF revenue")},
        dirs={_ROOT},
    )
    connectors = {"nas": GuardedConnector(t, allowed_root=_ROOT, read_only=read_only, max_bytes=10_000_000)}
    tools = build_tools(connectors) + build_remote_tools(connectors, chat or FakeChat(), _cfg())
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | STORAGE_SAFE_TOOL_NAMES | REMOTE_DOCQA_SAFE_TOOL_NAMES)
    return ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=tools), gate


# --------------------------------------------------------------------------- #
# the whole tool set classifies + routes correctly
# --------------------------------------------------------------------------- #
def test_full_toolset_gate_classification():
    _, gate = _composed()
    for name in ("storage_list", "storage_stat", "storage_read", "summarize_remote", "answer_remote"):
        assert gate.classify(Action(name, {"connector": "nas", "path": "x.txt"})).verdict is Verdict.safe, name
    for name in ("storage_write", "storage_delete", "storage_move"):
        assert gate.classify(Action(name, {"connector": "nas", "path": "x"})).verdict is Verdict.needs_confirmation, name


async def test_read_and_remote_execute_via_dispatcher():
    disp, _ = _composed()
    listed = await disp.dispatch(Action("storage_list", {"connector": "nas"}), "s5")
    assert listed.outcome is Outcome.executed and {"doc.txt", "report.pdf"} <= {e["path"] for e in listed.result["entries"]}
    summ = await disp.dispatch(Action("summarize_remote", {"connector": "nas", "path": "report.pdf"}), "s5")
    assert summ.outcome is Outcome.executed and summ.result["ok"] is True # PDF format-aware
    ans = await disp.dispatch(Action("answer_remote", {"question": "revenue", "connector": "nas", "path": "doc.txt"}), "s5")
    assert ans.outcome is Outcome.executed and ans.result["answer_found"] is True
    assert ans.result["citations"][0]["source"] == "doc.txt"


async def test_mutation_confirm_then_execute():
    disp, _ = _composed(read_only=False)
    action = Action("storage_write", {"connector": "nas", "path": "new.txt", "content": "hi"})
    pending = await disp.dispatch(action, "s5")
    assert pending.outcome is Outcome.pending
    disp.approvals.approve(pending.approval_id, action, _Approver())
    done = await disp.execute_approved(pending.approval_id, action)
    assert done.outcome is Outcome.executed and done.result["ok"] is True


# --------------------------------------------------------------------------- #
# security holds across the whole set
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tool", ["storage_read", "summarize_remote"])
async def test_secret_path_blocked_across_set(tool):
    disp, _ = _composed()
    args = {"connector": "nas", "path": "/etc/shadow"}
    if tool == "summarize_remote":
        args = {"connector": "nas", "path": "/etc/shadow"}
    res = await disp.dispatch(Action(tool, args), "s5")
    assert res.outcome is Outcome.refused and res.result is None


async def test_read_only_blocks_mutation_after_approval():
    disp, _ = _composed(read_only=True)
    action = Action("storage_delete", {"connector": "nas", "path": "doc.txt"})
    pending = await disp.dispatch(action, "s5")
    disp.approvals.approve(pending.approval_id, action, _Approver())
    done = await disp.execute_approved(pending.approval_id, action)
    assert done.result["ok"] is False and "StorageReadOnly" in done.result["error"]


async def test_remote_fault_isolation_llm_down():
    disp, _ = _composed(chat=FakeChat(raise_exc=EngineNotReady("no model")))
    res = await disp.dispatch(Action("summarize_remote", {"connector": "nas", "path": "doc.txt"}), "s5")
    assert res.outcome is Outcome.executed and res.result["ok"] is False
    assert "ServingUnavailable" in res.result["error"]
