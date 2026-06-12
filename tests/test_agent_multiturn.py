"""multi-turn conversation.

`run_task.history` (client-supplied prior turns) is shape-validated by the session (TEXT-ONLY
`{role, content}`, role ∈ {user, assistant}; system/tool/approval frames rejected or stripped),
then seeded into the loop's `messages` BETWEEN the system prompt and the new user turn, BOUNDED
(keep most-recent `agent_max_history_messages`, cap each content to `agent_history_char_cap`) so
multi-turn cannot defeat INV-5's bounded-work intent / prompt-bloat DoS. The non-history path is
unchanged. Run in conda `local-ai-agent-env-1`: `pytest tests/test_agent_multiturn.py`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, Orchestrator, ToolCall
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _settings(**kw):
    return Settings(**_DIRS, **kw)


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #
class CapturingModel:
    """Records the `messages` seen on the first complete() call, then returns a final answer."""

    def __init__(self):
        self.first_messages = None

    async def complete(self, messages, tools):
        if self.first_messages is None:
            self.first_messages = list(messages)
        return AssistantTurn(text="done")


class LoopingModel:
    """Always proposes a tool call → drives the loop to its bound (for INV-5 checks)."""

    async def complete(self, messages, tools):
        return AssistantTurn(tool_calls=[ToolCall("c", "read_file", {"path": "workspace/a"})])


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"agent:run", "approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


class RecordingTool:
    def __init__(self, name, result="ok"):
        self.name = name
        self.calls = []

    async def run(self, args):
        self.calls.append(args)
        return "ok"


class NullChannel:
    """Sends are captured; receive blocks forever (the run completes via the model, no approval)."""

    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        await asyncio.Event().wait()


def _runtime(model, tools=(), **caps):
    approvals = PendingApprovals(timeout_seconds=300, id_factory=lambda: "appr-1")
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=list(tools))
    return AgentRuntime(gate=SafetyGate(), approvals=approvals, dispatcher=disp,
                        model=model, **caps)


# =========================== config validators ============================== #
def test_history_caps_defaults():
    s = _settings()
    assert s.agent_max_history_messages == 20 and s.agent_history_char_cap == 4096


@pytest.mark.parametrize("field,bad", [
    ("agent_max_history_messages", 0), ("agent_max_history_messages", -1),
    ("agent_history_char_cap", 0), ("agent_history_char_cap", -5),
])
def test_history_caps_must_be_positive(field, bad):
    with pytest.raises(Exception):
        _settings(**{field: bad})


def test_history_caps_wired_from_settings():
    app = build_application(_settings(enable_agent=True, agent_max_history_messages=7,
                                     agent_history_char_cap=256))
    rt = app.agent_runtime
    assert rt.max_history_messages == 7 and rt.history_char_cap == 256


# =========================== loop seeding (NORMAL) ========================== #
async def test_history_seeded_between_system_and_user():
    model = CapturingModel()
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher,
                        system_prompt="SYS",
                        history=[{"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "a1"}])
    await orch.run("q2", "operator")
    roles = [(m["role"], m["content"]) for m in model.first_messages]
    assert roles == [("system", "SYS"), ("user", "q1"), ("assistant", "a1"), ("user", "q2")]


async def test_no_history_unchanged():
    model = CapturingModel()
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher, system_prompt="SYS")
    await orch.run("hi", "operator")
    assert [(m["role"], m["content"]) for m in model.first_messages] == [("system", "SYS"), ("user", "hi")]


async def test_history_without_system_prompt():
    model = CapturingModel()
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher,
                        history=[{"role": "user", "content": "earlier"}])
    await orch.run("now", "operator")
    assert [m["role"] for m in model.first_messages] == ["user", "user"]
    assert model.first_messages[0]["content"] == "earlier"


# =========================== loop seeding (BOUNDS / ERROR) ================== #
async def test_history_keeps_most_recent_n():
    model = CapturingModel()
    hist = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher,
                        history=hist, max_history_messages=3)
    await orch.run("task", "operator")
    seeded = [m["content"] for m in model.first_messages if m["content"] != "task"]
    assert seeded == ["m7", "m8", "m9"] # last 3 only


async def test_history_content_capped():
    model = CapturingModel()
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher,
                        history=[{"role": "user", "content": "x" * 100}], history_char_cap=10)
    await orch.run("task", "operator")
    assert model.first_messages[0]["content"] == "x" * 10


@pytest.mark.parametrize("bad", [
    {"role": "system", "content": "sneaky"}, # system role dropped defensively
    {"role": "tool", "content": "obs"}, # tool role dropped
    {"role": "user"}, # missing content
    {"role": "user", "content": 5}, # non-str content
    "not-a-dict", None, 123,
])
async def test_history_defensive_drops_bad_items(bad):
    model = CapturingModel()
    orch = Orchestrator(model=model, dispatcher=_runtime(model).dispatcher,
                        history=[bad, {"role": "user", "content": "good"}])
    await orch.run("task", "operator")
    seeded = [m for m in model.first_messages if m.get("content") != "task"]
    assert seeded == [{"role": "user", "content": "good"}] # only the valid item survived


def test_orchestrator_rejects_nonpositive_history_caps():
    model = CapturingModel()
    with pytest.raises(ValueError):
        Orchestrator(model=model, dispatcher=_runtime(model).dispatcher, max_history_messages=0)
    with pytest.raises(ValueError):
        Orchestrator(model=model, dispatcher=_runtime(model).dispatcher, history_char_cap=0)


async def test_inv5_bound_holds_with_history():
    # history must NOT relax INV-5: a model that always tool-calls still stops at max_steps
    tool = RecordingTool("read_file")
    orch = Orchestrator(model=LoopingModel(), dispatcher=_runtime(LoopingModel(), [tool]).dispatcher,
                        max_steps=3, max_tool_calls=99,
                        history=[{"role": "user", "content": "h"}] * 50)
    res = await orch.run("go", "operator")
    assert res.status.value == "max_steps" and res.steps == 3


# =========================== session validation ============================ #
async def _run_session(history, *, caps=None):
    model = CapturingModel()
    rt = _runtime(model, **(caps or {}))
    ch = NullChannel()
    msg = {"action": "run_task", "task": "now"}
    if history is not None:
        msg["history"] = history
    await AgentSession(ch, FakePrincipal(), rt).run(msg)
    return model, ch


async def test_session_valid_history_reaches_model():
    model, ch = await _run_session([{"role": "user", "content": "q1"},
                                    {"role": "assistant", "content": "a1"}])
    seeded = [(m["role"], m["content"]) for m in model.first_messages]
    assert ("user", "q1") in seeded and ("assistant", "a1") in seeded
    assert seeded[-1] == ("user", "now") # new task is last


async def test_session_strips_tool_calls_from_history_item():
    # an assistant frame carrying tool_calls is sanitized to text-only (no authority injection)
    model, ch = await _run_session([{"role": "assistant", "content": "a", "tool_calls": [{"x": 1}]}])
    hist_items = [m for m in model.first_messages if m["content"] == "a"]
    assert hist_items and "tool_calls" not in hist_items[0]


@pytest.mark.parametrize("bad_history", [
    [{"role": "system", "content": "x"}], # system role rejected
    [{"role": "tool", "content": "x"}], # tool role rejected
    [{"role": "user"}], # missing content
    [{"role": "user", "content": 1}], # non-str content
    ["not-a-dict"],
    "notalist",
    [None],
])
async def test_session_rejects_malformed_history(bad_history):
    model, ch = await _run_session(bad_history)
    errs = [s for s in ch.sent if s.get("event") == "error" and s.get("reason") == "invalid history"]
    assert errs # errored out
    assert model.first_messages is None # the run never started


async def test_session_drops_oldest_beyond_cap():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    model, ch = await _run_session(hist, caps={"max_history_messages": 3})
    seeded = [m["content"] for m in model.first_messages if m["content"] != "now"]
    assert seeded == ["m7", "m8", "m9"]


async def test_session_absent_history_unchanged():
    model, ch = await _run_session(None)
    assert [(m["role"], m["content"]) for m in model.first_messages] == [("user", "now")]


async def test_session_empty_history_list_ok():
    model, ch = await _run_session([])
    assert [(m["role"], m["content"]) for m in model.first_messages] == [("user", "now")]
