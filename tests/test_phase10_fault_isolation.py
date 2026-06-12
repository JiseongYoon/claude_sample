"""— fault-isolation + health-gating audit over the FULLY COMPOSED graph (G4).

With ALL `enable_*` on, prove the acceptance-critical property at full scale: a broken/absent module
disables only its own (and explicitly dependent) features while every unrelated capability — and the
registry's health-based gating (DocQA `depends_on=("llm-serving",)`) — keeps working. No cascade, no
global crash, and the gate never weakens under a module fault.

Drives the REAL `build_application` (via `BuildOverrides`/`_compose`); faults are injected through the
serving leaf (down / raises-at-start / flips-unhealthy / raises-at-chat). Run in conda
`local-ai-agent-env-1`: `pytest tests/test_phase10_fault_isolation.py`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.core.module import Health, HealthStatus
from local_ai_agent.modules.llm_serving import EngineNotReady
from local_ai_agent.modules.orchestrator.loop import AssistantTurn
from local_ai_agent.modules.safety.dispatcher import Outcome
from local_ai_agent.modules.safety.gate import Action, Verdict
from tests.test_phase10_integration import FakeServing, _compose, _run_task, _tc

_INDEP = ("storage", "browser", "exec", "mcp") # capabilities with depends_on=() — never gated by serving


# -- serving fakes that fault in different ways (subclass the healthy FakeServing) ------------------ #
class DownServing(FakeServing):
    def health(self) -> Health:
        return Health(HealthStatus.down, "engine crashed")


class RaiseStartServing(FakeServing):
    async def start(self) -> None:
        raise RuntimeError("serving failed to start")


class FlipServing(FakeServing):
    def __init__(self, content: str = "ok") -> None:
        super().__init__(content)
        self._down = False

    def go_down(self) -> None:
        self._down = True

    def health(self) -> Health:
        return Health(HealthStatus.down, "flipped") if self._down else super().health()


class RaiseChatServing(FakeServing):
    async def chat(self, messages, **params):
        raise EngineNotReady("no model is loaded/ready")


# --------------------------------------------------------------------------- #
# normal — composed enumeration + everything available when healthy
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_all_capabilities_available_when_healthy(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")])
    await app.startup()
    try:
        reg = app.registry
        assert reg.start_errors == {}
        caps = reg.available_capabilities()
        assert {"docqa", "storage", "browser", "exec", "mcp", "chat"} <= caps
        # every module healthy + dependency satisfied
        for m in ("llm-serving", "docqa", "storage", "browser", "exec", "mcp"):
            assert reg.is_module_available(m), m
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# health-gating crux — a down dependency gates ONLY its dependents
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_serving_down_gates_only_docqa_not_independents(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")], serving_override=DownServing())
    await app.startup()
    try:
        reg = app.registry
        # the dependency (llm-serving) is down → DocQA (depends_on it) is gated off
        assert not reg.is_module_available("llm-serving")
        assert not reg.is_capability_available("docqa")
        assert "docqa" not in reg.available_capabilities()
        # independents (depends_on=()) stay available — NO cascade
        for cap in _INDEP:
            assert reg.is_capability_available(cap), cap
        # the gated dispatcher still functions for an unrelated capability's tool
        r = await app.agent_runtime.dispatcher.dispatch(Action("web_search", {"query": "x"}), "op")
        assert r.outcome is Outcome.executed
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# a module that RAISES at start() is isolated (others start; app still builds)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_start_failure_is_isolated_no_cascade(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")], serving_override=RaiseStartServing())
    await app.startup() # must NOT raise — the registry isolates the failure
    try:
        reg = app.registry
        assert "llm-serving" in reg.start_errors # recorded, not propagated
        assert reg.health("llm-serving").status is HealthStatus.down
        assert not reg.is_capability_available("docqa") # dependent gated
        for cap in _INDEP: # independents unaffected
            assert reg.is_capability_available(cap), cap
        # the composed dispatcher still works (no global crash from the start failure)
        assert (await app.agent_runtime.dispatcher.dispatch(
            Action("web_search", {"query": "x"}), "op")).outcome is Outcome.executed
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# a module flipping unhealthy MID-SESSION → availability reflects it; others unaffected
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_midsession_unhealthy_flip(tmp_path):
    flip = FlipServing()
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")], serving_override=flip)
    await app.startup()
    try:
        reg = app.registry
        assert reg.is_capability_available("docqa") # healthy at first
        flip.go_down() # serving degrades mid-session
        assert not reg.is_capability_available("docqa") # re-read health → gated immediately
        for cap in _INDEP:
            assert reg.is_capability_available(cap), cap # unrelated capabilities still up
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# dependent graceful degrade — a chat-dependent tool fails gracefully (no crash),
# and an independent capability still completes in the same session
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dependent_degrades_gracefully_independent_unaffected(tmp_path):
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "answer_question", path="doc.txt", question="what is the capital of France?")]),
        AssistantTurn(tool_calls=[_tc(2, "run_command", command=["echo", "ok"])]), # independent exec hop
        AssistantTurn(text="done"),
    ]
    app, h = _compose(tmp_path, turns, serving_override=RaiseChatServing())
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="answer (chat down) then run")
        tr = ch.events("task_result")[0]
        assert tr["status"] == "completed" # no crash despite the chat fault
        assert any(c[-2:] == ["echo", "ok"] for c in h["runner"].calls) # exec (independent) still ran
        # the loop completed PAST the degraded hop to the final answer (non-vacuous: answer key present)
        assert tr["answer"] == "done"
        assert "EngineNotReady" not in tr["answer"] # the typed fault did not leak into the answer
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# the gate NEVER weakens under a module fault (gated stays gated, blocked stays blocked)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_gate_unweakened_under_fault(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")], serving_override=DownServing())
    await app.startup()
    try:
        disp = app.agent_runtime.dispatcher
        # a gated tool is still gated (pending), never auto-run, even while a module is down
        assert (await disp.dispatch(Action("run_command", {"command": ["ls"]}), "op")).outcome is Outcome.pending
        assert (await disp.dispatch(Action("storage_write", {"connector": "nas", "path": "f", "content": "x"}),
                                    "op")).outcome is Outcome.pending
        # a catastrophic command is still refused
        assert (await disp.dispatch(Action("run_command", {"command": ["rm", "-rf", "/"]}),
                                    "op")).outcome is Outcome.refused
    finally:
        await app.shutdown()
