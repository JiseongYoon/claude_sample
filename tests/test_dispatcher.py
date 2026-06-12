"""Tests for the gated ToolDispatcher.

The dispatcher is the only path from a proposed action to tool execution: classify →
safe→run · needs_confirmation→enrol→approve→consume→run · blocked→refuse. No tool runs
without passing the gate, and a blocked action never runs even with a valid approval.

Run in conda `local-ai-agent-env-1`: `pytest` (asyncio_mode=auto).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import (
    DispatchResult,
    Outcome,
    Tool,
    ToolDispatcher,
)
from local_ai_agent.modules.safety.gate import Action, SafetyGate, Verdict


# --------------------------------------------------------------------------- #
# test doubles
# --------------------------------------------------------------------------- #
class RecordingTool:
    """Records every invocation; returns a fixed result."""

    def __init__(self, name: str, result="ok") -> None:
        self.name = name
        self.result = result
        self.calls: list[dict] = []

    async def run(self, args: dict):
        self.calls.append(args)
        return self.result


class FailingTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    async def run(self, args: dict):
        self.calls.append(args)
        # message embeds a "secret" to prove it does NOT leak into the result
        raise RuntimeError("boom /etc/shadow internal-detail")


class SlowTool:
    """Yields control inside run() so a concurrent execute can interleave."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict] = []

    async def run(self, args: dict):
        self.calls.append(args)
        await asyncio.sleep(0)
        return "done"


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# realistic gate-classified actions
SAFE = Action("read_file", {"path": "workspace/notes.txt"}) # → safe
CONFIRM = Action("delete_file", {"path": "workspace/old.txt"}) # → needs_confirmation
BLOCKED = Action("read_file", {"path": "/etc/shadow"}) # → blocked (secret)
SHELL_BLOCKED = Action("shell", {"cmd": "rm -rf /"}) # → blocked (command)


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
        "read_file": RecordingTool("read_file", result="file-contents"),
        "delete_file": RecordingTool("delete_file", result="deleted"),
    }


@pytest.fixture
def disp(approvals, tools) -> ToolDispatcher:
    return ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools.values())


@pytest.fixture
def approver() -> FakePrincipal:
    return FakePrincipal()


# --------------------------------------------------------------------------- #
# sanity: the fixtures classify as intended (guards the test premises)
# --------------------------------------------------------------------------- #
def test_premise_classifications():
    g = SafetyGate()
    assert g.classify(SAFE).verdict is Verdict.safe
    assert g.classify(CONFIRM).verdict is Verdict.needs_confirmation
    assert g.classify(BLOCKED).verdict is Verdict.blocked
    assert g.classify(SHELL_BLOCKED).verdict is Verdict.blocked


# --------------------------------------------------------------------------- #
# NORMAL class
# --------------------------------------------------------------------------- #
async def test_safe_runs_immediately(disp, tools):
    res = await disp.dispatch(SAFE, requested_by="agent")
    assert res.outcome is Outcome.executed
    assert res.result == "file-contents"
    assert tools["read_file"].calls == [SAFE.args]


async def test_needs_confirmation_then_approve_then_execute(disp, approvals, approver, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    assert res.outcome is Outcome.pending
    assert res.approval_id is not None
    assert tools["delete_file"].calls == [] # not run yet

    approvals.approve(res.approval_id, CONFIRM, approver) # gateway does this
    run = await disp.execute_approved(res.approval_id, CONFIRM)
    assert run.outcome is Outcome.executed
    assert run.result == "deleted"
    assert tools["delete_file"].calls == [CONFIRM.args]


async def test_blocked_refused_secret(disp, tools):
    res = await disp.dispatch(BLOCKED, requested_by="agent")
    assert res.outcome is Outcome.refused
    assert tools["read_file"].calls == [] # never invoked


async def test_blocked_refused_command(disp):
    res = await disp.dispatch(SHELL_BLOCKED, requested_by="agent")
    assert res.outcome is Outcome.refused


# --------------------------------------------------------------------------- #
# ERROR class — fail-closed everywhere
# --------------------------------------------------------------------------- #
async def test_blocked_never_runs_even_with_approval(disp, approvals, approver, tools):
    # enrol+approve a legitimate confirm action...
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    approvals.approve(res.approval_id, CONFIRM, approver)
    # ...then try to launder a blocked action through that approval id
    run = await disp.execute_approved(res.approval_id, BLOCKED)
    assert run.outcome is Outcome.refused # re-classify blocks before consume/run
    assert tools["read_file"].calls == [] # the blocked tool never ran
    # the legitimate approval is untouched (not consumed) and can still run its own action
    ok = await disp.execute_approved(res.approval_id, CONFIRM)
    assert ok.outcome is Outcome.executed


async def test_double_execute_sequential_runs_once(disp, approvals, approver, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    approvals.approve(res.approval_id, CONFIRM, approver)
    first = await disp.execute_approved(res.approval_id, CONFIRM)
    second = await disp.execute_approved(res.approval_id, CONFIRM)
    assert first.outcome is Outcome.executed
    assert second.outcome is Outcome.aborted # replay rejected by consume
    assert len(tools["delete_file"].calls) == 1


async def test_double_execute_concurrent_runs_once(approvals, approver):
    # a slow tool that awaits inside run; two concurrent executes must still run once
    slow = SlowTool("delete_file") # name maps to a needs_confirmation action
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[slow])
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    approvals.approve(res.approval_id, CONFIRM, approver)
    a, b = await asyncio.gather(
        disp.execute_approved(res.approval_id, CONFIRM),
        disp.execute_approved(res.approval_id, CONFIRM),
    )
    outcomes = sorted([a.outcome, b.outcome], key=lambda o: o.value)
    assert outcomes == [Outcome.aborted, Outcome.executed]
    assert len(slow.calls) == 1


async def test_denied_then_execute_aborted(disp, approvals, approver, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    approvals.approve(res.approval_id, CONFIRM, approver)
    approvals.deny(res.approval_id) # revoke before run
    run = await disp.execute_approved(res.approval_id, CONFIRM)
    assert run.outcome is Outcome.aborted
    assert tools["delete_file"].calls == []


async def test_expired_then_execute_aborted(disp, approvals, approver, clock, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    # never approved; let it expire, then try to execute
    clock.advance(301)
    run = await disp.execute_approved(res.approval_id, CONFIRM)
    assert run.outcome is Outcome.aborted
    assert tools["delete_file"].calls == []


async def test_execute_without_approval_aborted(disp, approvals, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    # skip approve entirely → consume sees `pending`, not `approved`
    run = await disp.execute_approved(res.approval_id, CONFIRM)
    assert run.outcome is Outcome.aborted
    assert tools["delete_file"].calls == []


async def test_action_mismatch_at_execute_aborted(disp, approvals, approver, tools):
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    approvals.approve(res.approval_id, CONFIRM, approver)
    other = Action("delete_file", {"path": "workspace/SOMETHING_ELSE.txt"})
    run = await disp.execute_approved(res.approval_id, other)
    assert run.outcome is Outcome.aborted
    assert tools["delete_file"].calls == []


async def test_unknown_tool_safe_verdict_errors(approvals):
    # a safe-classified action whose tool is NOT registered → error, never executed
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    res = await disp.dispatch(SAFE, requested_by="agent")
    assert res.outcome is Outcome.error
    assert "unknown tool" in (res.error or "")


async def test_blocked_precedes_unknown_tool(approvals):
    # blocked must win even when the tool is unregistered
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    res = await disp.dispatch(BLOCKED, requested_by="agent")
    assert res.outcome is Outcome.refused # not "error: unknown tool"


async def test_dispatch_unknown_tool_needs_confirmation_does_not_enrol(approvals):
    # a needs_confirmation action whose tool is unregistered → error, NO approval enrolled
    # (don't create approvals that can never run — fail fast, no registry bloat).
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    res = await disp.dispatch(CONFIRM, requested_by="agent")
    assert res.outcome is Outcome.error
    assert res.approval_id is None
    assert approvals.pending() == []


async def test_unknown_tool_at_execute_does_not_consume(approvals, approver):
    # an approval that exists (e.g. tool was present at dispatch) but whose tool is absent
    # now → execute_approved errors BEFORE consume, so the approval is not wasted.
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    dec = SafetyGate().classify(CONFIRM)
    rec = approvals.request(CONFIRM, dec, "agent") # enrol directly (bypass dispatch)
    approvals.approve(rec.approval_id, CONFIRM, approver)
    run = await disp.execute_approved(rec.approval_id, CONFIRM)
    assert run.outcome is Outcome.error
    # approval was NOT consumed (still approved) — it wasn't wasted by a missing tool
    assert approvals.get(rec.approval_id).status.value == "approved"


async def test_tool_exception_caught_and_no_leak(approvals):
    failing = RecordingTool # placeholder to keep import tidy
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[FailingTool("read_file")])
    res = await disp.dispatch(SAFE, requested_by="agent")
    assert res.outcome is Outcome.error
    assert res.error == "tool raised RuntimeError" # type only
    assert "/etc/shadow" not in (res.error or "") # tool-internal string did NOT leak
    assert "internal-detail" not in (res.error or "")
    # dispatcher survives — a subsequent safe dispatch on a good tool still works
    disp.register(RecordingTool("read_file", result="recovered"))
    ok = await disp.dispatch(SAFE, requested_by="agent")
    assert ok.outcome is Outcome.executed and ok.result == "recovered"


async def test_unknown_approval_id_aborted(disp, tools):
    run = await disp.execute_approved("forged-id", CONFIRM)
    assert run.outcome is Outcome.aborted
    assert tools["delete_file"].calls == []


async def test_malformed_action_never_raises(disp):
    # malformed → gate returns blocked → refused; dispatcher must not raise
    malformed = Action("", {})
    res = await disp.dispatch(malformed, requested_by="agent")
    assert isinstance(res, DispatchResult)
    assert res.outcome is Outcome.refused


async def test_result_is_immutable(disp):
    res = await disp.dispatch(SAFE, requested_by="agent")
    with pytest.raises(Exception): # FrozenInstanceError
        res.outcome = Outcome.refused
