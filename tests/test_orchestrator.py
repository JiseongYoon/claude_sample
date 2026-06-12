"""Tests for the bounded ReAct Orchestrator loop.

Hermetic: a scripted ToolCallingModel + stub tools behind the real ToolDispatcher +
PendingApprovals, an injected clock, and fake approval seams. Asserts boundedness,
crash-safety, and that no tool runs outside the dispatcher.

Run in conda `local-ai-agent-env-1`: `pytest` (asyncio_mode=auto).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.modules.orchestrator.loop import (
    AssistantTurn,
    Orchestrator,
    RunStatus,
    ToolCall,
)
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import Action, Decision, SafetyGate


# --------------------------------------------------------------------------- #
# test doubles
# --------------------------------------------------------------------------- #
class ScriptedModel:
    """Returns a predetermined sequence of turns. `loop_forever` repeats the last turn
    so we can probe the step bound. A turn that is an Exception instance is raised."""

    def __init__(self, turns: list, loop_forever: bool = False) -> None:
        self.turns = turns
        self.loop_forever = loop_forever
        self.calls = 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.calls += 1
        idx = self.calls - 1
        if idx >= len(self.turns):
            if self.loop_forever:
                last = self.turns[-1]
            else:
                return AssistantTurn(text="(no more script)")
        else:
            last = self.turns[idx]
        if isinstance(last, BaseException):
            raise last
        return last


class RecordingTool:
    def __init__(self, name: str, result="ok") -> None:
        self.name = name
        self.result = result
        self.calls: list[dict] = []

    async def run(self, args: dict):
        self.calls.append(args)
        return self.result


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


SAFE = ToolCall("c1", "read_file", {"path": "workspace/a.txt"})
CONFIRM = ToolCall("c2", "delete_file", {"path": "workspace/old.txt"})
BLOCKED = ToolCall("c3", "read_file", {"path": "/etc/shadow"})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def approvals(clock) -> PendingApprovals:
    seq = {"n": 0}

    def ids() -> str:
        seq["n"] += 1
        return f"appr-{seq['n']:04d}"

    return PendingApprovals(timeout_seconds=300.0, now=clock, id_factory=ids)


@pytest.fixture
def tools():
    return {
        "read_file": RecordingTool("read_file", result="file-data"),
        "delete_file": RecordingTool("delete_file", result="deleted"),
    }


@pytest.fixture
def dispatcher(approvals, tools) -> ToolDispatcher:
    return ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools.values())


def make_orch(model, dispatcher, approvals, clock, *, seam=None, **kw):
    return Orchestrator(model=model, dispatcher=dispatcher, obtain_approval=seam,
                        now=clock, **kw)


def auto_approve_seam(approvals, principal=None):
    principal = principal or FakePrincipal()

    async def seam(action: Action, decision: Decision, approval_id: str) -> None:
        approvals.approve(approval_id, action, principal)

    return seam


def auto_deny_seam(approvals):
    async def seam(action: Action, decision: Decision, approval_id: str) -> None:
        approvals.deny(approval_id)

    return seam


# --------------------------------------------------------------------------- #
# NORMAL class
# --------------------------------------------------------------------------- #
async def test_final_answer_no_tools(dispatcher, approvals, clock, tools):
    model = ScriptedModel([AssistantTurn(text="hello, done")])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("hi", requested_by="agent")
    assert res.status is RunStatus.completed
    assert res.answer == "hello, done"
    assert res.tool_calls_made == 0
    assert tools["read_file"].calls == []


async def test_safe_tool_then_finalize(dispatcher, approvals, clock, tools):
    model = ScriptedModel([
        AssistantTurn(tool_calls=[SAFE]),
        AssistantTurn(text="used the file"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("read it", requested_by="agent")
    assert res.status is RunStatus.completed
    assert res.answer == "used the file"
    assert tools["read_file"].calls == [SAFE.args] # ran exactly once, via dispatcher
    # the tool result was observed and fed back
    assert any(m["role"] == "tool" and "file-data" in m["content"] for m in res.transcript)


async def test_needs_confirmation_auto_approved(dispatcher, approvals, clock, tools):
    model = ScriptedModel([
        AssistantTurn(tool_calls=[CONFIRM]),
        AssistantTurn(text="deleted it"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock,
                     seam=auto_approve_seam(approvals))
    res = await orch.run("delete old", requested_by="agent")
    assert res.status is RunStatus.completed
    assert tools["delete_file"].calls == [CONFIRM.args] # ran after approval


async def test_multiple_tool_calls_one_turn(dispatcher, approvals, clock, tools):
    two = AssistantTurn(tool_calls=[SAFE, ToolCall("c1b", "read_file", {"path": "workspace/b.txt"})])
    model = ScriptedModel([two, AssistantTurn(text="both read")])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("read both", requested_by="agent")
    assert res.status is RunStatus.completed
    assert res.tool_calls_made == 2
    assert len(tools["read_file"].calls) == 2
    tool_msgs = [m for m in res.transcript if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c1b"} # correlated by id


# --------------------------------------------------------------------------- #
# ERROR class — bounded + never-crash
# --------------------------------------------------------------------------- #
async def test_bounded_by_max_steps(dispatcher, approvals, clock, tools):
    # a model that always wants a safe tool → must halt at max_steps
    model = ScriptedModel([AssistantTurn(tool_calls=[SAFE])], loop_forever=True)
    orch = make_orch(model, dispatcher, approvals, clock, max_steps=5, max_tool_calls=999)
    res = await orch.run("loop", requested_by="agent")
    assert res.status is RunStatus.max_steps
    assert res.steps == 5
    assert model.calls == 5


async def test_max_steps_zero_no_model_call(dispatcher, approvals, clock):
    model = ScriptedModel([AssistantTurn(text="never reached")])
    orch = make_orch(model, dispatcher, approvals, clock, max_steps=0)
    res = await orch.run("x", requested_by="agent")
    assert res.status is RunStatus.max_steps
    assert model.calls == 0


async def test_bounded_by_max_tool_calls(dispatcher, approvals, clock, tools):
    model = ScriptedModel([AssistantTurn(tool_calls=[SAFE])], loop_forever=True)
    orch = make_orch(model, dispatcher, approvals, clock, max_steps=999, max_tool_calls=3)
    res = await orch.run("loop", requested_by="agent")
    assert res.status is RunStatus.budget_exhausted
    assert res.tool_calls_made == 3
    assert len(tools["read_file"].calls) == 3


async def test_timed_out_by_deadline(dispatcher, approvals, clock, tools):
    class TickingModel:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            clock.advance(10) # each turn burns 10s
            return AssistantTurn(tool_calls=[SAFE])

    orch = make_orch(TickingModel(), dispatcher, approvals, clock,
                     max_steps=999, deadline_seconds=25)
    res = await orch.run("slow", requested_by="agent")
    assert res.status is RunStatus.timed_out


async def test_model_exception_is_graceful(dispatcher, approvals, clock):
    model = ScriptedModel([RuntimeError("model blew up")])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("x", requested_by="agent") # must NOT raise
    assert res.status is RunStatus.error
    assert "RuntimeError" in res.reason


async def test_blocked_action_fed_back_loop_continues(dispatcher, approvals, clock, tools):
    model = ScriptedModel([
        AssistantTurn(tool_calls=[BLOCKED]), # gate → refused
        AssistantTurn(text="ok i won't"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("read shadow", requested_by="agent")
    assert res.status is RunStatus.completed # loop continued past refusal
    assert tools["read_file"].calls == [] # blocked action never ran
    assert any("[refused]" in m["content"] for m in res.transcript if m["role"] == "tool")


async def test_denied_approval_fed_back(dispatcher, approvals, clock, tools):
    model = ScriptedModel([
        AssistantTurn(tool_calls=[CONFIRM]),
        AssistantTurn(text="ok aborted"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock, seam=auto_deny_seam(approvals))
    res = await orch.run("delete", requested_by="agent")
    assert res.status is RunStatus.completed
    assert tools["delete_file"].calls == [] # not run (denied)
    assert any("[aborted]" in m["content"] for m in res.transcript if m["role"] == "tool")


async def test_default_seam_never_approves(dispatcher, approvals, clock, tools):
    # no seam injected → needs_confirmation never approved → aborts (fail-closed)
    model = ScriptedModel([
        AssistantTurn(tool_calls=[CONFIRM]),
        AssistantTurn(text="done"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock) # seam=None
    res = await orch.run("delete", requested_by="agent")
    assert res.status is RunStatus.completed
    assert tools["delete_file"].calls == []


async def test_faulty_seam_does_not_crash(dispatcher, approvals, clock, tools):
    async def boom(action, decision, approval_id):
        raise ValueError("seam exploded")

    model = ScriptedModel([
        AssistantTurn(tool_calls=[CONFIRM]),
        AssistantTurn(text="survived"),
    ])
    orch = make_orch(model, dispatcher, approvals, clock, seam=boom)
    res = await orch.run("delete", requested_by="agent") # must NOT raise
    assert res.status is RunStatus.completed
    assert tools["delete_file"].calls == [] # not approved → not run


async def test_result_is_size_capped(approvals, clock):
    big = RecordingTool("read_file", result="X" * 10_000)
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[big])
    model = ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="ok")])
    orch = make_orch(model, disp, approvals, clock, result_char_cap=100)
    res = await orch.run("read big", requested_by="agent")
    tool_msg = next(m for m in res.transcript if m["role"] == "tool")
    assert len(tool_msg["content"]) <= 100 + len("…[truncated]")
    assert tool_msg["content"].endswith("…[truncated]")


async def test_non_string_result_does_not_crash(approvals, clock):
    odd = RecordingTool("read_file", result={"nested": [1, 2, 3], "x": object()})
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[odd])
    model = ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="ok")])
    orch = make_orch(model, disp, approvals, clock)
    res = await orch.run("read odd", requested_by="agent") # str() it, no crash
    assert res.status is RunStatus.completed


async def test_unknown_tool_fed_back(approvals, clock):
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[]) # nothing registered
    model = ScriptedModel([AssistantTurn(tool_calls=[SAFE]), AssistantTurn(text="ok")])
    orch = make_orch(model, disp, approvals, clock)
    res = await orch.run("read", requested_by="agent")
    assert res.status is RunStatus.completed
    assert any("[error]" in m["content"] for m in res.transcript if m["role"] == "tool")


async def test_negative_bounds_rejected(dispatcher, approvals, clock):
    model = ScriptedModel([AssistantTurn(text="x")])
    with pytest.raises(ValueError):
        make_orch(model, dispatcher, approvals, clock, max_steps=-1)
    with pytest.raises(ValueError):
        make_orch(model, dispatcher, approvals, clock, deadline_seconds=0)


async def test_run_result_is_frozen(dispatcher, approvals, clock):
    model = ScriptedModel([AssistantTurn(text="done")])
    orch = make_orch(model, dispatcher, approvals, clock)
    res = await orch.run("x", requested_by="agent")
    with pytest.raises(Exception): # FrozenInstanceError
        res.status = RunStatus.error
