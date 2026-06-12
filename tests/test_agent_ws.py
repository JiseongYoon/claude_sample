"""End-to-end WS tests for the agent route over the real gateway.

Uses FastAPI TestClient against `create_gateway`, with a hand-built AgentRuntime (fake
model + stub tools) attached to the Application — exercising the real `/ws` run_task
routing + AgentSession over an actual WebSocket, plus echo back-compat.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate

_DIRS = dict(_env_file=None, model_safetensors_dir="./m", model_gguf_dir="./m")


def _settings(**kw):
    # auth disabled → WS principal is anonymous full-scope (has agent:run + approve)
    return Settings(**_DIRS, auth_enabled=False, **kw)


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


def _app_with_runtime(model, tools, *, decision_timeout=5.0):
    app = Application(modules=[])
    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    dispatcher = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=tools)
    app.agent_runtime = AgentRuntime(gate=SafetyGate(), approvals=approvals,
                                     dispatcher=dispatcher, model=model,
                                     decision_timeout=decision_timeout)
    return app


SAFE = ToolCall("c1", "read_file", {"path": "workspace/a.txt"})
CONFIRM = ToolCall("c2", "delete_file", {"path": "workspace/old.txt"})


def _recv_until(ws, event, *, collect=None):
    """Read frames until one with `event` arrives. Appends every seen event name to
    `collect` if given."""
    while True:
        msg = ws.receive_json()
        if collect is not None:
            collect.append(msg.get("event"))
        if msg.get("event") == event:
            return msg


def test_ws_echo_backcompat_unaffected():
    # an app with no agent_runtime still echoes plain text (existing behavior)
    api = create_gateway(Application(modules=[]), _settings())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["event"] == "capabilities"
        ws.send_text("hi")
        assert ws.receive_json() == {"event": "echo", "data": "hi"}


def test_ws_run_task_without_agent_errors():
    api = create_gateway(Application(modules=[]), _settings())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        ws.send_text(json.dumps({"action": "run_task", "task": "x"}))
        msg = ws.receive_json()
        assert msg == {"event": "error", "reason": "agent not enabled"}


def test_ws_run_task_safe_tool_completes():
    tool = RecordingTool("read_file", "data")
    app = _app_with_runtime(ScriptedModel([AssistantTurn(tool_calls=[SAFE]),
                                           AssistantTurn(text="done")]), [tool])
    api = create_gateway(app, _settings())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        ws.send_text(json.dumps({"action": "run_task", "task": "read it"}))
        seen: list[str] = []
        result = _recv_until(ws, "task_result", collect=seen)
        assert result["status"] == "completed"
        # live tool-activity streamed ahead of the terminal result
        assert "tool_call" in seen and "tool_result" in seen
        assert seen.index("tool_call") < seen.index("tool_result")
    assert tool.calls == [SAFE.args]


def test_ws_full_confirm_approve_roundtrip():
    tool = RecordingTool("delete_file", "deleted")
    app = _app_with_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                           AssistantTurn(text="deleted it")]), [tool])
    api = create_gateway(app, _settings())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        ws.send_text(json.dumps({"action": "run_task", "task": "delete old"}))
        before: list[str] = []
        prompt = _recv_until(ws, "approval_request", collect=before)
        assert prompt["tool"] == "delete_file"
        assert "tool_call" in before # tool_call precedes the approval
        ws.send_text(json.dumps({"action": "approve", "approval_id": prompt["approval_id"]}))
        after: list[str] = []
        result = _recv_until(ws, "task_result", collect=after)
        assert result["status"] == "completed"
        assert "tool_result" in after # tool_result after approval
    assert tool.calls == [CONFIRM.args]


def test_ws_confirm_deny_roundtrip():
    tool = RecordingTool("delete_file", "deleted")
    app = _app_with_runtime(ScriptedModel([AssistantTurn(tool_calls=[CONFIRM]),
                                           AssistantTurn(text="ok")]), [tool])
    api = create_gateway(app, _settings())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        ws.send_text(json.dumps({"action": "run_task", "task": "delete"}))
        prompt = _recv_until(ws, "approval_request")
        ws.send_text(json.dumps({"action": "deny", "approval_id": prompt["approval_id"]}))
        after: list[str] = []
        result = _recv_until(ws, "task_result", collect=after)
        assert result["status"] == "completed"
        assert "tool_result" in after # denied → tool_result outcome reflects abort
    assert tool.calls == [] # denied → never ran
