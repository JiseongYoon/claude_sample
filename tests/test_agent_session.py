"""Tests for the interactive AgentSession WS round-trip.

Hermetic — a fake `Channel` (auto-responding to approval_request per a policy, or
raising to simulate disconnect) drives the session over the real runtime/orchestrator/
dispatcher/approvals with a scripted model + stub tools. Asserts scope enforcement,
server-side action binding, deny/timeout/stale/disconnect handling, and audit events.

Run in conda `local-ai-agent-env-1`: `pytest` (asyncio_mode=auto).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"agent:run", "approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


class RecordingSink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


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


class FakeChannel:
    """Captures sent events; `auto(request)->'approve'|'deny'|dict|None` responds to each
    approval_request. `raise_on_receive` makes receive raise (simulated disconnect)."""

    def __init__(self, *, auto=None, raise_on_receive=None):
        self.sent: list[dict] = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._auto = auto
        self._raise = raise_on_receive

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)
        if self._auto and data.get("event") == "approval_request":
            resp = self._auto(data)
            if resp in ("approve", "deny"):
                await self._inbox.put({"action": resp, "approval_id": data["approval_id"]})
            elif isinstance(resp, dict):
                await self._inbox.put(resp)

    async def receive_json(self):
        if self._raise is not None:
            raise self._raise
        return await self._inbox.get()

    def push(self, msg):
        self._inbox.put_nowait(msg)

    def events(self, name):
        return [s for s in self.sent if s.get("event") == name]


SAFE = ToolCall("c1", "read_file", {"path": "workspace/a.txt"})
CONFIRM = ToolCall("c2", "delete_file", {"path": "workspace/old.txt"})


def build_runtime(model, tools, *, decision_timeout=2.0, audit=None):
    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    dispatcher = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools)
    return AgentRuntime(gate=SafetyGate(), approvals=approvals, dispatcher=dispatcher,
                        model=model, audit=audit or RecordingSink(),
                        decision_timeout=decision_timeout)


# --------------------------------------------------------------------------- #
# NORMAL
# --------------------------------------------------------------------------- #
async def test_safe_tool_run_completes():
    tool = RecordingTool("read_file", "data")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[SAFE]),
                                       AssistantTurn(text="done")]), [tool])
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "read"})
    res = ch.events("task_result")
    assert res and res[0]["status"] == "completed"
    assert tool.calls == [SAFE.args]


async def test_needs_confirmation_auto_approve_runs():
    tool = RecordingTool("delete_file", "deleted")
    audit = RecordingSink()
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool], audit=audit)
    ch = FakeChannel(auto=lambda req: "approve")
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert ch.events("approval_request") # prompt was sent
    assert ch.events("task_result")[0]["status"] == "completed"
    assert tool.calls == [CONFIRM.args] # ran after approval
    # audit recorded an approve decision with the principal
    appr = [e for e in audit.events if e.get("kind") == "approval" and e.get("decision") == "approve"]
    assert appr and appr[0]["principal"] == "operator"


# --------------------------------------------------------------------------- #
# ERROR / security
# --------------------------------------------------------------------------- #
async def test_missing_agent_run_scope_forbidden():
    tool = RecordingTool("read_file")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[SAFE])]), [tool])
    weak = FakePrincipal(subject="bob", scopes=frozenset({"read"})) # no agent:run
    ch = FakeChannel()
    await AgentSession(ch, weak, rt).run({"action": "run_task", "task": "x"})
    assert ch.events("error") and ch.events("error")[0]["reason"] == "forbidden"
    assert tool.calls == [] # orchestrator never built


async def test_approve_without_approve_scope_rejected():
    tool = RecordingTool("delete_file")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool], decision_timeout=0.5)
    runner = FakePrincipal(subject="carol", scopes=frozenset({"agent:run"})) # no approve
    ch = FakeChannel(auto=lambda req: "approve")
    await AgentSession(ch, runner, rt).run({"action": "run_task", "task": "del"})
    assert ch.events("approval_error") # approve rejected (NotAuthorized)
    assert tool.calls == [] # never ran


async def test_server_side_action_binding_no_substitution():
    # client tries to substitute a different tool/args in the decision message → ignored;
    # the server-stored CONFIRM action is what gets approved/run.
    tool = RecordingTool("delete_file", "deleted")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool])

    def forge(req):
        # respond with an approve carrying a forged tool/args alongside the real id
        return {"action": "approve", "approval_id": req["approval_id"],
                "tool": "read_file", "args": {"path": "/etc/shadow"}}

    ch = FakeChannel(auto=forge)
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert tool.calls == [CONFIRM.args] # the ORIGINAL action ran, not the forgery


async def test_deny_aborts_tool():
    tool = RecordingTool("delete_file")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="ok aborted")]), [tool])
    ch = FakeChannel(auto=lambda req: "deny")
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert ch.events("task_result")[0]["status"] == "completed"
    assert tool.calls == [] # denied → never ran


async def test_decision_timeout_denies_and_aborts():
    tool = RecordingTool("delete_file")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="ok")]), [tool], decision_timeout=0.05)
    ch = FakeChannel(auto=lambda req: None) # client never answers
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert ch.events("approval_timeout") # timed out
    assert tool.calls == [] # not run
    # B5: the approval was denied on timeout (so a late approve can't run it)
    assert rt.approvals.get("appr-1") is None # purged at task end (terminal)


async def test_unknown_approval_id_decision_errors():
    tool = RecordingTool("delete_file", "deleted")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool], decision_timeout=0.5)

    def bogus_then_real(req):
        # push a decision for a non-existent id first, then the real approve
        return {"action": "approve", "approval_id": "does-not-exist"}

    ch = FakeChannel(auto=bogus_then_real)
    sess = AgentSession(ch, FakePrincipal(), rt)
    # also queue the real approve so the task can still complete
    ch.push({"action": "approve", "approval_id": "appr-1"})
    await sess.run({"action": "run_task", "task": "del"})
    assert ch.events("approval_error") # the bogus id errored
    assert tool.calls == [CONFIRM.args] # the real one still ran


async def test_junk_decision_message_no_crash():
    tool = RecordingTool("delete_file", "deleted")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool], decision_timeout=0.5)

    def junk_then_real(req):
        return "not-a-dict" # malformed → enqueued? no (not approve/deny/dict)

    ch = FakeChannel(auto=junk_then_real)
    ch.push({"garbage": True}) # malformed dict
    ch.push({"action": "approve", "approval_id": "appr-1"}) # then the real one
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert ch.events("approval_error") # junk handled gracefully
    assert tool.calls == [CONFIRM.args] # real approve still ran


async def test_disconnect_mid_approval_cancels_run():
    tool = RecordingTool("delete_file")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool])
    ch = FakeChannel(raise_on_receive=ConnectionError("client gone"))
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    assert tool.calls == [] # tool never ran (run cancelled)
    assert ch.events("task_result") == [] # no completion sent on a dead channel


async def test_not_run_task_first_message():
    rt = build_runtime(ScriptedModel([AssistantTurn(text="x")]), [])
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "something_else"})
    assert ch.events("error")[0]["reason"] == "expected run_task"


async def test_missing_task_field():
    rt = build_runtime(ScriptedModel([AssistantTurn(text="x")]), [])
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task"})
    assert ch.events("error")[0]["reason"] == "missing task"


async def test_purge_called_at_task_end():
    tool = RecordingTool("delete_file", "deleted")
    rt = build_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                       AssistantTurn(text="done")]), [tool])
    ch = FakeChannel(auto=lambda req: "approve")
    await AgentSession(ch, FakePrincipal(), rt).run({"action": "run_task", "task": "del"})
    # the consumed approval was purged at task end
    assert rt.approvals.get("appr-1") is None
