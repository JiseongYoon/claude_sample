"""agent wire-format correctness (F1 + F2).

Adversarial verification (2026-06-04) surfaced two pre-existing latent defects on the
agent↔real-model path, masked by hermetic-only testing (scripted models never hit `chat()`)
and direct-dispatch smokes (the model never proposed a tool call):

- **F1** — `llm_serving.chat` forwarded only `_PARAM_KEYS`, so the agent's `tools` schemas were
  silently dropped and never reached the engine → the model could never propose a tool call.
- **F2** — the orchestrator loop records an assistant tool-call turn in its own compact shape
  `{"role":"assistant","content":…,"tool_calls":[{"id","tool","args"}]}` and the adapter sent the
  message list verbatim → the engine received a NON-OpenAI assistant shape.

This module proves both fixes deterministically (no real engine), including the COMBINED path
adapter → LLMServingModule(FakeTransport) so the captured wire payload is asserted end-to-end.

Run in conda `local-ai-agent-env-1`: `pytest tests/test_agent_wire_format.py`.
"""
from __future__ import annotations

import json

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.llm_serving import LLMServingModule
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.model_adapter import (
    LLMToolModel,
    to_openai_messages,
)
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate


# ---- fakes (mirror test_llm_serving / test_model_adapter conventions) ------- #
class FakeTarget:
    def __init__(self, serving: bool = True, base_url: str = "http://127.0.0.1:8000") -> None:
        self._serving = serving
        self._base_url = base_url

    @property
    def is_serving(self) -> bool:
        return self._serving

    @property
    def base_url(self) -> str:
        return self._base_url


class CaptureTransport:
    """Captures the exact payload that would go on the wire to /v1/chat/completions."""

    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, base_url: str, payload: dict) -> dict:
        self.calls.append((base_url, payload))
        return self.response


class FakeChat:
    def __init__(self, response: dict | None = None) -> None:
        self.response = response or {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        self.seen: list[tuple[list[dict], dict]] = []

    async def chat(self, messages, **params):
        self.seen.append((messages, params))
        return self.response


def _settings() -> Settings:
    return Settings(_env_file=None, model_safetensors_dir="./m", model_gguf_dir="./m")


def _serving(transport: CaptureTransport) -> LLMServingModule:
    return LLMServingModule(_settings(), FakeTarget(serving=True), transport=transport)


# a representative internal-shape transcript the loop produces (loop.py:163-213)
def _loop_messages() -> list[dict]:
    return [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "summarize the doc"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "tool": "summarize_document", "args": {"path": "a/b.txt"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "[executed] summary…"},
    ]


# =========================== F1 — tools reach the engine ===================== #
async def test_chat_forwards_tools_to_payload():
    tx = CaptureTransport()
    mod = _serving(tx)
    await mod.start()
    schemas = [{"type": "function", "function": {"name": "summarize_document"}}]
    await mod.chat([{"role": "user", "content": "q"}], tools=schemas, temperature=0.0)
    _, payload = tx.calls[0]
    assert payload["tools"] == schemas # F1: the schemas now reach the wire
    assert payload["temperature"] == 0.0


@pytest.mark.parametrize("tools", [None, [], "notalist", 5, {"a": 1}])
async def test_chat_omits_tools_when_absent_or_malformed(tools):
    tx = CaptureTransport()
    mod = _serving(tx)
    await mod.start()
    kwargs = {} if tools is None else {"tools": tools}
    await mod.chat([{"role": "user", "content": "q"}], **kwargs)
    _, payload = tx.calls[0]
    assert "tools" not in payload # only a non-empty list is forwarded


async def test_direct_chat_unaffected_no_tools_key():
    # the non-agent callers (gateway /chat, qa, summarizer, multimodal) pass no tools
    tx = CaptureTransport()
    mod = _serving(tx)
    await mod.start()
    await mod.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    _, payload = tx.calls[0]
    assert "tools" not in payload and payload["max_tokens"] == 8


# =========================== F2 — OpenAI assistant shape ===================== #
def test_to_openai_messages_translates_assistant_tool_calls():
    out = to_openai_messages(_loop_messages())
    asst = out[2]
    assert asst["role"] == "assistant"
    tc = asst["tool_calls"][0]
    assert tc["id"] == "c1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "summarize_document"
    # arguments MUST be a JSON string (OpenAI contract), not a dict
    assert tc["function"]["arguments"] == json.dumps({"path": "a/b.txt"})
    assert json.loads(tc["function"]["arguments"]) == {"path": "a/b.txt"}


def test_to_openai_messages_passthrough_non_tool_messages():
    msgs = _loop_messages()
    out = to_openai_messages(msgs)
    assert out[0] == msgs[0] # system unchanged
    assert out[1] == msgs[1] # user unchanged
    assert out[3] == msgs[3] # tool result unchanged


def test_to_openai_messages_args_already_string_kept():
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "tool": "t", "args": '{"k": 1}'}]}]
    out = to_openai_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"k": 1}'


def test_to_openai_messages_text_only_assistant_unchanged_shape():
    msgs = [{"role": "assistant", "content": "final answer"}]
    out = to_openai_messages(msgs)
    assert out[0] == {"role": "assistant", "content": "final answer"}
    assert "tool_calls" not in out[0]


@pytest.mark.parametrize("junk", [None, 5, "x", [1, 2], {"role": "assistant", "tool_calls": "nope"}])
def test_to_openai_messages_never_raises_on_junk(junk):
    # robustness: arbitrary junk items are skipped or passed through, never crash
    out = to_openai_messages([junk, {"role": "user", "content": "ok"}])
    assert {"role": "user", "content": "ok"} in out


def test_to_openai_messages_skips_non_dict_tool_calls():
    msgs = [{"role": "assistant", "content": "",
             "tool_calls": ["bad", 5, {"id": "c1", "tool": "t", "args": {}}]}]
    out = to_openai_messages(msgs)
    calls = out[0]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "t"


async def test_complete_sends_openai_shaped_messages():
    chat = FakeChat()
    model = LLMToolModel(chat, temperature=0.0)
    await model.complete(_loop_messages(), tools=[{"function": {"name": "summarize_document"}}])
    sent_msgs, params = chat.seen[0]
    # the assistant tool-call message reached chat() in OpenAI shape
    assert sent_msgs[2]["tool_calls"][0]["type"] == "function"
    assert sent_msgs[2]["tool_calls"][0]["function"]["name"] == "summarize_document"
    assert params["tools"] == [{"function": {"name": "summarize_document"}}]


# =================== F1 + F2 COMBINED — end-to-end wire proof ================ #
async def test_combined_adapter_through_serving_to_wire():
    """The real proof: LLMToolModel → real LLMServingModule(CaptureTransport). Both the tool
    schemas (F1) AND the OpenAI-shaped assistant message (F2) must appear on the captured wire
    payload — the path that the agent loop drives with a real engine."""
    tx = CaptureTransport(response={"choices": [{"message": {"content": "done"}}]})
    serving = _serving(tx)
    await serving.start()
    model = LLMToolModel(serving) # adapter over the REAL serving module
    schemas = [{"type": "function", "function": {"name": "summarize_document"}}]
    turn = await model.complete(_loop_messages(), tools=schemas)

    assert turn.text == "done"
    _, payload = tx.calls[0]
    # F1: tools on the wire
    assert payload["tools"] == schemas
    # F2: the assistant tool-call message is OpenAI-shaped on the wire
    wire_asst = payload["messages"][2]
    assert wire_asst["tool_calls"][0]["type"] == "function"
    assert wire_asst["tool_calls"][0]["function"]["name"] == "summarize_document"
    assert wire_asst["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "a/b.txt"})
    # untouched messages stay verbatim
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][3]["role"] == "tool"


# =========================== F3a — tools reach the model ===================== #
class _SchemaTool:
    def __init__(self, name, description=None, parameters=None, input_schema=None):
        self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if input_schema is not None:
            self.input_schema = input_schema

    async def run(self, args):
        return {"ok": True}


def _dispatcher(tools):
    return ToolDispatcher(gate=SafetyGate(), approvals=PendingApprovals(timeout_seconds=300), tools=tools)


def test_tool_schemas_builds_openai_function_shape():
    params = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    disp = _dispatcher([_SchemaTool("answer_question", "Answer a question.", params)])
    schemas = disp.tool_schemas()
    assert schemas == [{"type": "function",
                        "function": {"name": "answer_question",
                                     "description": "Answer a question.",
                                     "parameters": params}}]


def test_tool_schemas_bare_tool_gets_empty_params_fallback():
    disp = _dispatcher([_SchemaTool("bare")])
    fn = disp.tool_schemas()[0]["function"]
    assert fn["name"] == "bare"
    assert "description" not in fn # no metadata → omitted
    assert fn["parameters"] == {"type": "object", "properties": {}} # safe fallback


def test_tool_schemas_mcp_input_schema_fallback():
    # MCP tools expose `input_schema` (not `parameters`) — it is used as the parameter schema
    isch = {"type": "object", "properties": {"x": {"type": "number"}}}
    disp = _dispatcher([_SchemaTool("mcp__srv__tool", "external", input_schema=isch)])
    fn = disp.tool_schemas()[0]["function"]
    assert fn["parameters"] == isch and fn["description"] == "external"


def test_tool_schemas_empty_registry():
    assert _dispatcher([]).tool_schemas() == []


# --- the session must HAND those schemas to the model (was empty before ) -- #
class _CapturingModel:
    def __init__(self):
        self.seen_tools = None

    async def complete(self, messages, tools):
        self.seen_tools = tools
        return AssistantTurn(text="done")


class _Principal:
    subject = "operator"
    scopes = frozenset({"agent:run", "approve"})

    def has_scope(self, scope):
        return scope in self.scopes


class _NullChannel:
    def __init__(self):
        self.sent = []

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        import asyncio
        await asyncio.Event().wait() # never delivers a decision (test completes via the model)


async def test_session_passes_registered_tool_schemas_to_model():
    model = _CapturingModel()
    disp = _dispatcher([_SchemaTool("answer_question", "Answer.",
                                    {"type": "object", "properties": {}})])
    rt = AgentRuntime(gate=SafetyGate(), approvals=disp.approvals, dispatcher=disp, model=model)
    await AgentSession(_NullChannel(), _Principal(), rt).run({"action": "run_task", "task": "hi"})
    # F3a: the model was handed the registered tool's schema (NOT an empty list)
    assert model.seen_tools is not None and len(model.seen_tools) == 1
    assert model.seen_tools[0]["function"]["name"] == "answer_question"


async def test_session_empty_registry_passes_empty_tools():
    model = _CapturingModel()
    disp = _dispatcher([])
    rt = AgentRuntime(gate=SafetyGate(), approvals=disp.approvals, dispatcher=disp, model=model)
    await AgentSession(_NullChannel(), _Principal(), rt).run({"action": "run_task", "task": "hi"})
    assert model.seen_tools == []
