"""token streaming.

A SEPARATE `chat_stream` + SSE transport (sets `stream:true` explicitly) consumes the engine's
streamed deltas, surfaces TEXT deltas to `on_token` for live display, and ACCUMULATES the full turn
(text + tool_call fragments reassembled by index) so the existing `parse_turn` runs at end-of-stream
— preserving tool-calling. The buffered `chat`/`task_result` path is UNTOUCHED. Run in conda
`local-ai-agent-env-1`: `pytest tests/test_token_streaming.py`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.llm_serving import (
    LLMServingModule,
    _parse_sse_line,
    _StreamAccum,
)
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, Orchestrator, ToolCall
from local_ai_agent.modules.orchestrator.model_adapter import LLMToolModel
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate


def _settings(**kw):
    return Settings(_env_file=None, model_safetensors_dir="./m", model_gguf_dir="./m", **kw)


def _text_chunk(s):
    return {"choices": [{"delta": {"content": s}}]}


def _tool_chunk(index, *, id=None, name=None, args=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    tc = {"index": index}
    if id is not None:
        tc["id"] = id
    if fn:
        tc["function"] = fn
    return {"choices": [{"delta": {"tool_calls": [tc]}}]}


# =========================== _StreamAccum (pure) ============================ #
def test_accum_text_deltas_and_response():
    a = _StreamAccum()
    assert a.add(_text_chunk("Hel")) == "Hel"
    assert a.add(_text_chunk("lo")) == "lo"
    assert a.response()["choices"][0]["message"]["content"] == "Hello"


def test_accum_reassembles_tool_call_across_fragments():
    a = _StreamAccum()
    assert a.add(_tool_chunk(0, id="call_1", name="summarize_document")) is None # no text delta
    a.add(_tool_chunk(0, args='{"pa'))
    a.add(_tool_chunk(0, args='th": "a/b.txt"}'))
    msg = a.response()["choices"][0]["message"]
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1" and tc["type"] == "function"
    assert tc["function"]["name"] == "summarize_document"
    assert tc["function"]["arguments"] == '{"path": "a/b.txt"}' # fragments concatenated, not parsed


def test_accum_multiple_tool_calls_by_index():
    a = _StreamAccum()
    a.add(_tool_chunk(0, id="c0", name="t0", args="{}"))
    a.add(_tool_chunk(1, id="c1", name="t1", args="{}"))
    calls = a.response()["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["t0", "t1"]


def test_accum_missing_id_synthesised():
    a = _StreamAccum()
    a.add(_tool_chunk(0, name="t", args="{}"))
    assert a.response()["choices"][0]["message"]["tool_calls"][0]["id"] == "call-0"


@pytest.mark.parametrize("junk", [None, 5, "x", {}, {"choices": []}, {"choices": [{}]},
                                  {"choices": [{"delta": "no"}]}, {"choices": [{"delta": {}}]}])
def test_accum_robust_on_junk(junk):
    a = _StreamAccum()
    assert a.add(junk) is None # never raises, no text delta
    # a junk-only stream yields an empty assistant turn
    assert a.response()["choices"][0]["message"]["content"] is None


def test_accum_text_only_has_no_tool_calls_key():
    a = _StreamAccum()
    a.add(_text_chunk("answer"))
    assert "tool_calls" not in a.response()["choices"][0]["message"]


# =========================== _parse_sse_line (pure) ======================== #
@pytest.mark.parametrize("line,expected", [
    ('data: {"choices": [{"delta": {"content": "x"}}]}', {"choices": [{"delta": {"content": "x"}}]}),
    ("data: [DONE]", None),
    ("", None),
    (": comment", None),
    ("event: ping", None),
    ("data: not-json{{{", None),
    ("data: 5", None), # non-dict json → None
])
def test_parse_sse_line(line, expected):
    assert _parse_sse_line(line) == expected


# =========================== chat_stream (module) ========================== #
class FakeTarget:
    def __init__(self, serving=True):
        self._s = serving

    @property
    def is_serving(self):
        return self._s

    @property
    def base_url(self):
        return "http://127.0.0.1:8000"


async def test_chat_stream_sets_stream_and_forwards_tools_and_tokens():
    captured = {}
    tokens = []

    async def fake_stream(base_url, payload, on_token, read_timeout):
        captured["payload"] = payload
        captured["read_timeout"] = read_timeout
        await on_token("hi")
        return {"choices": [{"message": {"content": "hi"}}]}

    mod = LLMServingModule(_settings(llm_stream_idle_timeout_s=12.0), FakeTarget(),
                           stream_transport=fake_stream)
    await mod.start()

    async def on_token(d):
        tokens.append(d)

    out = await mod.chat_stream([{"role": "user", "content": "q"}], on_token,
                                tools=[{"function": {"name": "t"}}], temperature=0.1)
    assert out["choices"][0]["message"]["content"] == "hi"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["tools"] == [{"function": {"name": "t"}}]
    assert captured["payload"]["temperature"] == 0.1
    assert captured["read_timeout"] == 12.0
    assert tokens == ["hi"]


async def test_chat_stream_raises_when_not_serving():
    from local_ai_agent.modules.llm_serving import EngineNotReady
    mod = LLMServingModule(_settings(), FakeTarget(serving=False),
                           stream_transport=lambda *a: None)
    await mod.start()
    with pytest.raises(EngineNotReady):
        await mod.chat_stream([{"role": "user", "content": "q"}], lambda d: None)


async def test_buffered_chat_never_sets_stream():
    captured = {}

    async def fake_chat(base_url, payload):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "x"}}]}

    mod = LLMServingModule(_settings(), FakeTarget(), transport=fake_chat)
    await mod.start()
    await mod.chat([{"role": "user", "content": "q"}])
    assert "stream" not in captured["payload"] # non-streaming path untouched


# =========================== adapter complete_stream ======================= #
class FakeStreamingChat:
    def __init__(self, response, deltas=()):
        self.response = response
        self.deltas = deltas
        self.seen = None

    async def chat(self, messages, **params):
        return self.response

    async def chat_stream(self, messages, on_token, **params):
        self.seen = (messages, params)
        for d in self.deltas:
            await on_token(d)
        return self.response


async def test_adapter_complete_stream_forwards_tokens_and_parses():
    tokens = []
    chat = FakeStreamingChat({"choices": [{"message": {"content": "final"}}]}, deltas=["fi", "nal"])
    model = LLMToolModel(chat, temperature=0.0)

    async def on_token(d):
        tokens.append(d)

    turn = await model.complete_stream([{"role": "user", "content": "hi"}],
                                       [{"function": {"name": "t"}}], on_token)
    assert turn.text == "final" and tokens == ["fi", "nal"]
    assert chat.seen[1]["tools"] == [{"function": {"name": "t"}}]


async def test_adapter_complete_stream_openai_shapes_messages():
    chat = FakeStreamingChat({"choices": [{"message": {"content": "ok"}}]})
    model = LLMToolModel(chat)
    internal = [{"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "tool": "summarize_document", "args": {"path": "a"}}]}]

    async def on_token(d):
        pass

    await model.complete_stream(internal, [], on_token)
    sent = chat.seen[0]
    assert sent[0]["tool_calls"][0]["type"] == "function" # to_openai_messages applied


# =========================== orchestrator streaming gate =================== #
class StreamingModel:
    """Implements complete_stream → emits tokens; returns a final answer."""

    def __init__(self):
        self.stream_called = False

    async def complete(self, messages, tools):
        return AssistantTurn(text="buffered")

    async def complete_stream(self, messages, tools, on_token):
        self.stream_called = True
        await on_token("to")
        await on_token("ken")
        return AssistantTurn(text="token")


class NonStreamingModel:
    async def complete(self, messages, tools):
        return AssistantTurn(text="done")


class RecordingEmit:
    def __init__(self):
        self.events = []

    async def __call__(self, ev):
        self.events.append(ev)


def _orch(model, *, emit=None):
    approvals = PendingApprovals(timeout_seconds=300)
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[])
    return Orchestrator(model=model, dispatcher=disp, emit=emit)


async def test_orchestrator_streams_tokens_when_emit_and_model_support():
    emit = RecordingEmit()
    model = StreamingModel()
    res = await _orch(model, emit=emit).run("hi", "op")
    assert model.stream_called and res.answer == "token"
    toks = [e["delta"] for e in emit.events if e["event"] == "token"]
    assert toks == ["to", "ken"]


async def test_orchestrator_no_emit_uses_buffered_complete():
    model = StreamingModel()
    res = await _orch(model).run("hi", "op") # no emit → no streaming
    assert model.stream_called is False and res.answer == "buffered"


async def test_orchestrator_emit_but_nonstreaming_model_falls_back():
    emit = RecordingEmit()
    res = await _orch(NonStreamingModel(), emit=emit).run("hi", "op")
    assert res.answer == "done"
    assert not [e for e in emit.events if e["event"] == "token"] # no token events


async def test_streamed_tool_call_turn_is_dispatched():
    # a streaming model that returns a tool-call turn → the loop dispatches it (tool-calling intact)
    class StreamingToolModel:
        def __init__(self):
            self.i = 0

        async def complete(self, messages, tools):
            return AssistantTurn(text="x")

        async def complete_stream(self, messages, tools, on_token):
            self.i += 1
            if self.i == 1:
                return AssistantTurn(tool_calls=[ToolCall("c1", "read_file", {"path": "workspace/a"})])
            return AssistantTurn(text="done")

    class RecTool:
        name = "read_file"

        def __init__(self):
            self.calls = []

        async def run(self, args):
            self.calls.append(args)
            return "data"

    tool = RecTool()
    approvals = PendingApprovals(timeout_seconds=300)
    disp = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[tool])
    emit = RecordingEmit()
    orch = Orchestrator(model=StreamingToolModel(), dispatcher=disp, emit=emit)
    res = await orch.run("read it", "op")
    assert res.status.value == "completed"
    assert tool.calls == [{"path": "workspace/a"}] # the streamed tool call ran (gated)
    assert any(e["event"] == "tool_call" for e in emit.events) # events still fire
