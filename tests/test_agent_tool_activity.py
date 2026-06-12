"""live tool-activity events.

The loop fires `tool_call` (pre-dispatch) and `tool_result` (after the INV-6 audit record) events
through an injected `emit` seam so a live client can show tool activity. emit is ADDITIVE
OBSERVATION ONLY — a read-only side-channel that never feeds back into the loop's control flow, so
INV-1/4/6 stay byte-for-byte. A faulty/dead emit must never crash or alter the run (mirrors the
approval-seam guard). Run in conda `local-ai-agent-env-1`: `pytest tests/test_agent_tool_activity.py`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from local_ai_agent.modules.orchestrator.loop import AssistantTurn, Orchestrator, ToolCall
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class ScriptedModel:
    def __init__(self, turns):
        self.turns = turns
        self.i = 0

    async def complete(self, messages, tools):
        t = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return t


class RecordingTool:
    def __init__(self, name, result="ok"):
        self.name = name
        self.result = result
        self.calls = []

    async def run(self, args):
        self.calls.append(args)
        return self.result


class RecordingEmit:
    def __init__(self, raises: bool = False):
        self.events: list[dict] = []
        self._raises = raises

    async def __call__(self, event: dict) -> None:
        self.events.append(event)
        if self._raises:
            raise RuntimeError("emit boom")


class RecordingSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"agent:run", "approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


class FakeChannel:
    def __init__(self, *, auto=None):
        self.sent: list[dict] = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._auto = auto

    async def send_json(self, data):
        self.sent.append(data)
        if self._auto and data.get("event") == "approval_request":
            resp = self._auto(data)
            if resp in ("approve", "deny"):
                await self._inbox.put({"action": resp, "approval_id": data["approval_id"]})

    async def receive_json(self):
        return await self._inbox.get()

    def names(self):
        return [s.get("event") for s in self.sent]


SAFE = ToolCall("c1", "read_file", {"path": "workspace/a.txt"})
CONFIRM = ToolCall("c2", "delete_file", {"path": "workspace/old.txt"})


def _orch(model, tools, *, emit=None, audit=None, **kw):
    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools)
    return Orchestrator(model=model, dispatcher=disp, emit=emit, audit=audit, **kw)


# =========================== loop: emit firing (NORMAL) ===================== #
async def test_emit_fires_tool_call_then_result():
    emit = RecordingEmit()
    tool = RecordingTool("read_file", {"ok": True})
    orch = _orch(ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]),
                 [tool], emit=emit)
    await orch.run("read", "operator")
    kinds = [e["event"] for e in emit.events]
    assert kinds == ["tool_call", "tool_result"] # call before result
    tc, tr = emit.events
    assert tc == {"event": "tool_call", "id": "c1", "tool": "read_file",
                  "args": {"path": "workspace/a.txt"}}
    assert tr["event"] == "tool_result" and tr["id"] == "c1" and tr["tool"] == "read_file"
    assert tr["outcome"] == "executed" and "result" in tr # capped observation included


async def test_emit_omitted_is_noop():
    tool = RecordingTool("read_file")
    orch = _orch(ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]), [tool])
    res = await orch.run("read", "operator") # no emit → must not crash
    assert res.status.value == "completed" and tool.calls == [SAFE.args]


async def test_blocked_tool_still_emits_and_is_refused():
    # a catastrophic/blocked action: the gate refuses; emit observes (no authority added)
    emit = RecordingEmit()
    bad = ToolCall("c9", "delete_file", {"path": "/etc/passwd", "recursive": True})
    orch = _orch(ScriptedModel([AssistantTurn(tool_calls=[bad]), AssistantTurn(text="ok")]),
                 [RecordingTool("delete_file")], emit=emit)
    await orch.run("rm", "operator")
    outcomes = [e.get("outcome") for e in emit.events if e["event"] == "tool_result"]
    assert outcomes and outcomes[0] in ("refused", "aborted", "pending", "error")
    # tool_call was still announced for the blocked attempt
    assert any(e["event"] == "tool_call" for e in emit.events)


# =========================== loop: emit isolation (ERROR/SECURITY) ========== #
async def test_faulty_emit_does_not_crash_or_alter_run():
    emit = RecordingEmit(raises=True) # every emit raises
    tool = RecordingTool("read_file", "data")
    audit = RecordingSink()
    orch = _orch(ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]),
                 [tool], emit=emit, audit=audit)
    res = await orch.run("read", "operator")
    assert res.status.value == "completed" # run survived the raising emit
    assert tool.calls == [SAFE.args] # tool still ran (control unaltered)
    # INV-6: the dispatch audit record is present despite the emit failure (emit fires AFTER audit)
    assert any(e.get("kind") == "dispatch" for e in audit.events)


async def test_emit_does_not_change_outcome_vs_no_emit():
    turns = lambda: [AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]
    a = await _orch(ScriptedModel(turns()), [RecordingTool("read_file")]).run("x", "op")
    b = await _orch(ScriptedModel(turns()), [RecordingTool("read_file")],
                    emit=RecordingEmit()).run("x", "op")
    assert (a.status, a.steps, a.tool_calls_made) == (b.status, b.steps, b.tool_calls_made)


async def test_emit_result_fires_after_audit_record():
    # ordering proof: the audit sink records the dispatch BEFORE the tool_result emit observes it
    order: list[str] = []

    class OrderAudit:
        def record(self, event):
            if event.get("kind") == "dispatch":
                order.append("audit")

    async def emit(event):
        if event["event"] == "tool_result":
            order.append("emit_result")

    orch = _orch(ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]),
                 [RecordingTool("read_file")], emit=emit, audit=OrderAudit())
    await orch.run("x", "op")
    assert order == ["audit", "emit_result"] # audit first, emit after


# =========================== session wiring ================================= #
def _runtime(model, tools):
    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools)
    return AgentRuntime(gate=SafetyGate(), approvals=approvals, dispatcher=disp,
                        model=model, decision_timeout=2.0)


async def test_session_streams_tool_activity_to_channel():
    rt = _runtime(ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="done")]),
                  [RecordingTool("read_file")])
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "read"})
    names = ch.names()
    assert "tool_call" in names and "tool_result" in names and "task_result" in names
    assert names.index("tool_call") < names.index("tool_result") < names.index("task_result")


async def test_session_confirm_tool_event_ordering():
    # tool_call → approval_request → (approve) → tool_result → task_result
    rt = _runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]), AssistantTurn(text="done")]),
                  [RecordingTool("delete_file", "deleted")])
    ch = FakeChannel(auto=lambda req: "approve")
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    names = ch.names()
    assert (names.index("tool_call") < names.index("approval_request")
            < names.index("tool_result") < names.index("task_result"))
