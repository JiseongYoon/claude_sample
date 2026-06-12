"""Tests for the audit sink + orchestrator audit hook + purge_terminal.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from local_ai_agent.modules.orchestrator.loop import AssistantTurn, Orchestrator, ToolCall
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.audit import (
    LoggingAuditSink,
    NullAuditSink,
    safe_record,
)
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import Action, Decision, SafetyGate, Verdict


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


class RaisingSink:
    def record(self, event: dict) -> None:
        raise RuntimeError("sink down")


class RecordingTool:
    def __init__(self, name, result="ok"):
        self.name = name
        self.result = result
        self.calls = []

    async def run(self, args):
        self.calls.append(args)
        return self.result


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


class ScriptedModel:
    def __init__(self, turns):
        self.turns = turns
        self.i = 0

    async def complete(self, messages, tools):
        t = self.turns[self.i]
        self.i += 1
        return t


# --- safe_record ----------------------------------------------------------- #
def test_null_sink_noop():
    NullAuditSink().record({"x": 1}) # no raise


def test_safe_record_swallows_sink_failure():
    safe_record(RaisingSink(), {"x": 1}) # must not raise


def test_logging_sink_stamps_ts(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="local_ai_agent.audit"):
        LoggingAuditSink().record({"kind": "task", "phase": "start"})
    assert any("audit" in r.message for r in caplog.records)


# --- orchestrator audit hook ----------------------------------------------- #
async def test_orchestrator_emits_task_and_dispatch_events():
    clock = FakeClock()
    approvals = PendingApprovals(timeout_seconds=300, now=clock, id_factory=lambda: "appr-1")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals,
                          tools=[RecordingTool("read_file", "data")])
    sink = RecordingSink()
    model = ScriptedModel([
        AssistantTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "workspace/a"})]),
        AssistantTurn(text="done"),
    ])
    orch = Orchestrator(model=model, dispatcher=disp, now=clock, audit=sink)
    res = await orch.run("go", requested_by="alice")
    kinds = [e["kind"] for e in sink.events]
    assert kinds[0] == "task" and sink.events[0]["phase"] == "start"
    assert kinds[-1] == "task" and sink.events[-1]["phase"] == "end"
    dispatch = [e for e in sink.events if e["kind"] == "dispatch"]
    assert len(dispatch) == 1
    assert dispatch[0]["principal"] == "alice"
    assert dispatch[0]["tool"] == "read_file"
    assert dispatch[0]["verdict"] == "safe"
    assert dispatch[0]["outcome"] == "executed"
    assert sink.events[-1]["status"] == res.status.value


async def test_orchestrator_audit_default_is_null():
    # no audit injected → no crash, runs fine
    clock = FakeClock()
    approvals = PendingApprovals(timeout_seconds=300, now=clock, id_factory=lambda: "a")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    orch = Orchestrator(model=ScriptedModel([AssistantTurn(text="hi")]),
                        dispatcher=disp, now=clock)
    res = await orch.run("x", requested_by="bob")
    assert res.status.value == "completed"


async def test_faulty_sink_does_not_break_run():
    clock = FakeClock()
    approvals = PendingApprovals(timeout_seconds=300, now=clock, id_factory=lambda: "a")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals,
                          tools=[RecordingTool("read_file")])
    model = ScriptedModel([
        AssistantTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "workspace/a"})]),
        AssistantTurn(text="done"),
    ])
    orch = Orchestrator(model=model, dispatcher=disp, now=clock, audit=RaisingSink())
    res = await orch.run("x", requested_by="alice") # must not raise
    assert res.status.value == "completed"


# --- purge_terminal -------------------------------------------------------- #
def _confirm():
    return Decision(Verdict.needs_confirmation, "need ok", "r")


def test_purge_terminal_drops_only_terminal():
    clock = FakeClock()
    seq = {"n": 0}

    def ids():
        seq["n"] += 1
        return f"a{seq['n']}"

    reg = PendingApprovals(timeout_seconds=300, now=clock, id_factory=ids)
    approver = FakePrincipal()
    pend = reg.request(Action("delete_file", {"path": "x"}), _confirm(), "agent") # pending
    cons = reg.request(Action("delete_file", {"path": "y"}), _confirm(), "agent")
    reg.approve(cons.approval_id, Action("delete_file", {"path": "y"}), approver)
    reg.consume(cons.approval_id, Action("delete_file", {"path": "y"})) # consumed
    den = reg.request(Action("delete_file", {"path": "z"}), _confirm(), "agent")
    reg.deny(den.approval_id) # denied

    removed = reg.purge_terminal()
    assert removed == 2 # consumed + denied
    assert reg.get(pend.approval_id) is not None # pending kept
    assert reg.get(cons.approval_id) is None
    assert reg.get(den.approval_id) is None


def test_purge_keeps_approved():
    clock = FakeClock()
    reg = PendingApprovals(timeout_seconds=300, now=clock, id_factory=lambda: "a1")
    rec = reg.request(Action("delete_file", {"path": "x"}), _confirm(), "agent")
    reg.approve(rec.approval_id, Action("delete_file", {"path": "x"}), FakePrincipal())
    assert reg.purge_terminal() == 0
    assert reg.get(rec.approval_id).status.value == "approved"
