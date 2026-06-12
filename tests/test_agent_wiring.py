"""Tests for AgentRuntime wiring in build_application.

Verifies the composition root attaches a runtime under `enable_agent`, leaves things
untouched when off, and that a runtime-built orchestrator runs end-to-end with a fake
model + stub tool + programmatic approval (no WS — that's ).

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.runtime import (
    SCOPE_AGENT_RUN,
    SCOPE_APPROVE,
    AgentRuntime,
)
from local_ai_agent.modules.safety.gate import Action

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _settings(**kw):
    return Settings(**_DIRS, **kw)


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


# --- wiring ---------------------------------------------------------------- #
def test_agent_off_no_runtime():
    app = build_application(_settings(enable_agent=False, enable_model_stack=False))
    assert app.agent_runtime is None


def test_agent_on_attaches_runtime():
    app = build_application(_settings(enable_agent=True))
    rt = app.agent_runtime
    assert isinstance(rt, AgentRuntime)
    # the model stack was built so the agent has a serving client to wrap
    assert app.get_module("llm-serving") is not None
    assert app.get_module("model-manager") is not None
    # bounds came from settings
    assert rt.max_steps == 12 and rt.max_tool_calls == 32


def test_agent_bounds_from_settings():
    app = build_application(_settings(enable_agent=True, agent_max_steps=3,
                                     agent_max_tool_calls=5, agent_result_char_cap=128,
                                     approval_decision_timeout_seconds=30))
    rt = app.agent_runtime
    assert (rt.max_steps, rt.max_tool_calls, rt.result_char_cap) == (3, 5, 128)
    assert rt.decision_timeout == 30


def test_scope_names_are_canonical():
    assert SCOPE_AGENT_RUN == "agent:run"
    assert SCOPE_APPROVE == "approve"


# --- runtime.build_orchestrator end-to-end (no WS) ------------------------- #
async def test_runtime_orchestrator_runs_safe_tool():
    # build a runtime by hand (so we can inject a fake model + stub tools deterministically)
    from local_ai_agent.modules.safety.approval import PendingApprovals
    from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
    from local_ai_agent.modules.safety.gate import SafetyGate

    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    tool = RecordingTool("read_file", "file-data")
    dispatcher = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[tool])
    model = ScriptedModel([
        AssistantTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "workspace/a"})]),
        AssistantTurn(text="done"),
    ])
    rt = AgentRuntime(gate=SafetyGate(), approvals=approvals, dispatcher=dispatcher, model=model)
    orch = rt.build_orchestrator() # no approval seam needed for a safe tool
    res = await orch.run("read it", requested_by="alice")
    assert res.status.value == "completed"
    assert tool.calls == [{"path": "workspace/a"}]


async def test_runtime_orchestrator_needs_confirmation_with_programmatic_approval():
    from local_ai_agent.modules.safety.approval import PendingApprovals
    from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
    from local_ai_agent.modules.safety.gate import Decision, SafetyGate

    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    tool = RecordingTool("delete_file", "deleted")
    dispatcher = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[tool])
    approver = FakePrincipal()

    async def approve_seam(action: Action, decision: Decision, approval_id: str) -> None:
        approvals.approve(approval_id, action, approver) # programmatic stand-in for WS

    model = ScriptedModel([
        AssistantTurn(tool_calls=[ToolCall("c1", "delete_file", {"path": "workspace/old"})]),
        AssistantTurn(text="deleted it"),
    ])
    rt = AgentRuntime(gate=SafetyGate(), approvals=approvals, dispatcher=dispatcher, model=model)
    orch = rt.build_orchestrator(approve_seam)
    res = await orch.run("delete old", requested_by="alice")
    assert res.status.value == "completed"
    assert tool.calls == [{"path": "workspace/old"}] # ran only after approval
