"""— safety-coverage audit: INV-1..6 over the FULLY COMPOSED graph.

Where proved the six invariants per-component, this re-establishes them with **every capability
loaded at once** (docqa+storage+browser+exec+mcp+agent), against the REAL `build_application` wiring
(via the `BuildOverrides` seam, hermetic), plus adversarial escape probes spanning capabilities and the
composed re-verification of the S1 (argv block) / S2 (MCP literal-env secret) hardening.

INV-1 single gated chokepoint · INV-2 deterministic · INV-3 fail-closed · INV-4 approval integrity
(single-use, action-bound, scope-checked, blocked-never-approvable) · INV-5 bounded loop · INV-6 audit.

Run in conda `local-ai-agent-env-1`: `pytest tests/test_phase10_safety_audit.py`.
"""
from __future__ import annotations

import json

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.orchestrator.loop import AssistantTurn
from local_ai_agent.modules.safety.approval import ApprovalError, ApprovalState
from local_ai_agent.modules.safety.dispatcher import Outcome
from local_ai_agent.modules.safety.gate import Action, Verdict
from tests.test_phase10_integration import (
    FakePrincipal,
    ScriptedModel,
    _compose,
    _dispatcher_tools,
    _run_task,
    _tc,
)

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


async def _started_app(tmp_path, turns=None, **kw):
    app, h = _compose(tmp_path, turns or [AssistantTurn(text="idle")], **kw)
    await app.startup()
    return app, h


# --------------------------------------------------------------------------- #
# INV-1 — single gated chokepoint over the composed graph (no bypass)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inv1_all_tools_on_one_dispatcher_and_blocked_never_runs(tmp_path):
    app, h = await _started_app(tmp_path)
    try:
        disp = app.agent_runtime.dispatcher
        tools = _dispatcher_tools(app)
        # every capability's tools (incl. dynamically-registered MCP) live on the ONE dispatcher
        assert {"summarize_document", "answer_question", "storage_read", "storage_write",
                "summarize_remote", "web_search", "open_url", "run_command",
                "read_workspace_file", "write_workspace_file", "mcp__fs__read"} <= tools

        # a catastrophic argv is refused outright (blocked), never enrolled for approval, never run
        blocked = Action("run_command", {"command": ["rm", "-rf", "/"]})
        r = await disp.dispatch(blocked, "operator")
        assert r.outcome is Outcome.refused and r.approval_id is None

        # blocked wins even over a VALID approval for a different (gated) action:
        gated = Action("run_command", {"command": ["ls"]})
        dec = app.agent_runtime.gate.classify(gated)
        rec = app.agent_runtime.approvals.request(gated, dec, "operator")
        app.agent_runtime.approvals.approve(rec.approval_id, gated, FakePrincipal())
        # try to spend that approval on the BLOCKED action → re-classified blocked → refused, not consumed
        r2 = await disp.execute_approved(rec.approval_id, blocked)
        assert r2.outcome is Outcome.refused
        assert app.agent_runtime.approvals.get(rec.approval_id).status is ApprovalState.approved # untouched
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# INV-2 / INV-3 — deterministic + fail-closed across the full composed tool set
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inv2_inv3_deterministic_and_fail_closed(tmp_path):
    app, h = await _started_app(tmp_path)
    try:
        gate = app.agent_runtime.gate
        samples = [
            Action("web_search", {"query": "x"}), # safe
            Action("answer_question", {"path": "doc.txt", "question": "q"}), # safe
            Action("run_command", {"command": ["ls"]}), # confirm
            Action("storage_write", {"connector": "nas", "path": "a", "content": "x"}), # confirm
            Action("mcp__fs__read", {"path": "p"}), # confirm (untrusted)
            Action("run_command", {"command": ["rm", "-rf", "/"]}), # blocked
        ]
        # INV-2 determinism: identical verdict+rule across repeated classification
        for a in samples:
            d1, d2 = gate.classify(a), gate.classify(a)
            assert (d1.verdict, d1.rule) == (d2.verdict, d2.rule)

        # INV-3 fail-closed: malformed / unknown never classify `safe`
        for bad in [Action("", {}), Action("run_command", None), Action(123, {}), # type: ignore[arg-type]
                    Action("totally_unknown_tool", {"x": 1})]:
            assert gate.classify(bad).verdict is not Verdict.safe
        # an external MCP tool is NEVER safe (untrusted) even though it is registered
        assert gate.classify(Action("mcp__fs__read", {})).verdict is Verdict.needs_confirmation
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# INV-4 — approval integrity, concurrent across two capabilities
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inv4_approval_integrity_composed(tmp_path):
    app, h = await _started_app(tmp_path)
    try:
        gate, appr = app.agent_runtime.gate, app.agent_runtime.approvals
        a_exec = Action("run_command", {"command": ["ls"]})
        a_store = Action("storage_write", {"connector": "nas", "path": "f", "content": "x"})
        r_exec = appr.request(a_exec, gate.classify(a_exec), "operator")
        r_store = appr.request(a_store, gate.classify(a_store), "operator")
        assert r_exec.approval_id != r_store.approval_id # distinct ids, no cross-binding

        # scope: a principal lacking the `approve` scope cannot approve
        no_scope = FakePrincipal(scopes=frozenset({"agent:run"}))
        with pytest.raises(ApprovalError):
            appr.approve(r_exec.approval_id, a_exec, no_scope)

        # action-binding: approving id_exec with a DIFFERENT action's fingerprint is rejected
        with pytest.raises(ApprovalError):
            appr.approve(r_exec.approval_id, a_store, FakePrincipal())

        # proper approve + single-use consume; replay rejected
        appr.approve(r_exec.approval_id, a_exec, FakePrincipal())
        appr.consume(r_exec.approval_id, a_exec)
        with pytest.raises(ApprovalError):
            appr.consume(r_exec.approval_id, a_exec) # single-use

        # cross-id: spending id_store with the exec action is rejected (binding)
        appr.approve(r_store.approval_id, a_store, FakePrincipal())
        with pytest.raises(ApprovalError):
            appr.consume(r_store.approval_id, a_exec)

        # blocked is never enrollable
        blk = Action("run_command", {"command": ["rm", "-rf", "/"]})
        with pytest.raises(ApprovalError):
            appr.request(blk, gate.classify(blk), "operator")
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# INV-5 — the composed loop stays bounded on an endless multi-tool model
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inv5_loop_bounded_composed(tmp_path):
    # a model that NEVER finishes — always proposes another safe tool call
    endless = ScriptedModel([AssistantTurn(tool_calls=[_tc(1, "web_search", query="loop")])])
    app, h = _compose(tmp_path, [AssistantTurn(text="x")]) # placeholder; replace the model
    app.agent_runtime.model = endless # inject the endless model
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="spin forever")
        tr = ch.events("task_result")[0]
        # it TERMINATED on a BOUND (didn't hang, not a final answer) and stayed within the ceilings
        assert tr["tool_calls_made"] <= 32
        assert tr["status"] != "completed" # not a model-produced final answer
        assert tr["status"] in {"max_steps", "max_tool_calls", "budget_exhausted"} # a real bound tripped
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# INV-6 — audit completeness with principal, over a composed multi-cap task
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inv6_audit_complete_with_principal(tmp_path):
    sink = RecordingSink()
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "web_search", query="capital")]), # safe
        AssistantTurn(tool_calls=[_tc(2, "run_command", command=["ls"])]), # gated → approve
        AssistantTurn(text="done"),
    ]
    app, h = _compose(tmp_path, turns, audit=sink)
    await app.startup()
    try:
        await _run_task(app, auto="approve", task="search then run")
        assert sink.events # something was audited
        assert all(e.get("principal") == "operator" for e in sink.events if "principal" in e)
        assert any("principal" in e for e in sink.events) # principal recorded
        assert any("run_command" in json.dumps(e, default=str) for e in sink.events) # the gated tool logged
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# S1 / S2 hardening re-verified in the composed wiring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_S1_argv_block_in_composed_dispatcher(tmp_path):
    app, h = await _started_app(tmp_path)
    try:
        r = await app.agent_runtime.dispatcher.dispatch(
            Action("run_command", {"command": ["rm", "-rf", "/"]}), "operator")
        assert r.outcome is Outcome.refused # blocked, not pending/executed
        assert r.approval_id is None # never enrolled for human approval
    finally:
        await app.shutdown()


def test_S2_secret_keyed_mcp_env_fails_composition(tmp_path):
    # a servers.json with a secret-looking LITERAL env key → McpModule build (inside build_application)
    # fails closed; the secret VALUE never leaks in the raised error.
    f = tmp_path / "servers.json"
    f.write_text(json.dumps({"servers": [{"name": "fs", "command": "srv-x",
                                          "env": {"API_KEY": "REALSECRET"}}]}), encoding="utf-8")
    settings = Settings(**_DIRS, enable_mcp=True, mcp_servers_file=f)
    with pytest.raises(ValueError) as ei:
        build_application(settings)
    assert "REALSECRET" not in str(ei.value)


# --------------------------------------------------------------------------- #
# adversarial escape probes — composed surface, repeated to confirm no escape
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_adversarial_escape_probes_composed(tmp_path):
    app, h = await _started_app(tmp_path)
    try:
        disp, gate, appr = (app.agent_runtime.dispatcher, app.agent_runtime.gate,
                            app.agent_runtime.approvals)
        for _ in range(3): # repeat — the gate is deterministic, no probe may ever escape
            # 1) unknown / laundered tool name → error, never runs
            r = await disp.dispatch(Action("mcp__evil__exfiltrate", {"x": 1}), "operator")
            assert r.outcome in (Outcome.error, Outcome.aborted) # not registered → never executes
            # a fake "native-looking" unknown tool
            assert (await disp.dispatch(Action("delete_everything", {}), "operator")).outcome \
                in (Outcome.error, Outcome.aborted, Outcome.refused)

            # 2) argv catastrophic across the exec tool → blocked
            assert (await disp.dispatch(Action("run_command", {"command": ["rm", "-rf", "/"]}),
                                        "operator")).outcome is Outcome.refused

            # 3) a registered external MCP tool is ALWAYS gated (never auto-runs)
            assert (await disp.dispatch(Action("mcp__fs__read", {"path": "p"}), "operator")).outcome \
                is Outcome.pending

            # 4) spend a never-approved id → aborted, never runs
            gated = Action("run_command", {"command": ["ls"]})
            rec = appr.request(gated, gate.classify(gated), "operator")
            assert (await disp.execute_approved(rec.approval_id, gated)).outcome is Outcome.aborted
    finally:
        await app.shutdown()
