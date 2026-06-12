"""— regression (the phase completion gate).

Mechanism (operator-chosen): **hermetic composed integration + real-API e2e**. This file is the hermetic
half — it drives the REAL composition root (`build_application`) with the REAL `LLMToolModel` adapter,
dispatcher/gate/approvals/audit, and `AgentSession` over an actual WebSocket; only the *serving engine*
is a recording fake (its `chat`/`chat_stream`). It verifies the four U-units work as ONE integrated whole:

  wire-format — the agent's tool schemas reach the engine + the loop's assistant tool-call turn is sent
                   in OpenAI shape (proven on the recorded wire);
  multi-turn — client-supplied `history` is seeded into the model's prompt;
  stream — `token` + `tool_call`/`tool_result` events stream over the WS while the gate still runs;
  INV-1/6 — the proposed tool runs ONLY through the gated dispatcher and is audited;
  terminal — the non-streaming `task_result` still ends the run.

The real-API e2e half lives in `web/e2e/`. Hermetic; conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application


class RecordingSink:
    """AuditSink override — records every gate/dispatch/approval event (INV-6 proof)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


class StreamingRecordingServing:
    """A fake llm-serving Module + streaming ChatModel. Records the exact (messages, params) it is
    handed each turn (so the test can assert wire-format + history on the real wire) and streams
    text via `on_token`. Scripted: turn 1 proposes a gated tool call (`list_documents`, safe-listed);
    turn 2 streams a final answer. Implements BOTH `chat` (buffered) and `chat_stream` (the agent path
    uses chat_stream because the session wires an emit seam)."""

    def __init__(self) -> None:
        self.stream_calls: list[tuple[list[dict], dict]] = []
        self.buffered_calls: list[tuple[list[dict], dict]] = []
        self.streamed_tokens: list[str] = []
        self._turn = 0

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def health(self) -> Health:
        return Health(HealthStatus.ok, "fake")

    def _scripted(self) -> dict:
        self._turn += 1
        if self._turn == 1:
            return {"choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "t1", "type": "function",
                                "function": {"name": "list_documents", "arguments": "{}"}}]}}]}
        return {"choices": [{"message": {"content": "Done now."}}]}

    async def chat(self, messages, **params) -> dict:
        self.buffered_calls.append((messages, params))
        return self._scripted()

    async def chat_stream(self, messages, on_token, **params) -> dict:
        self.stream_calls.append((messages, params))
        resp = self._scripted()
        # stream text deltas only for a text turn (a tool-call turn has no displayable text)
        content = resp["choices"][0]["message"].get("content")
        if isinstance(content, str) and content:
            for tok in (content[: len(content) // 2], content[len(content) // 2:]):
                await on_token(tok)
                self.streamed_tokens.append(tok)
        return resp


def _settings(tmp_path, **over) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "readme.txt").write_text("hello world") # so list_documents returns a doc
    base = dict(_env_file=None, model_safetensors_dir=str(tmp_path), model_gguf_dir=str(tmp_path),
                auth_enabled=False, enable_docqa=True, enable_agent=True, docs_root=str(docs))
    base.update(over)
    return Settings(**base)


def _app(s, serving, audit):
    application = build_application(
        s, overrides=BuildOverrides(serving_module=serving, audit_sink=audit))
    from local_ai_agent.core.gateway import create_gateway
    return application, create_gateway(application, s)


def _drive(api, *, history=None):
    """Run one agent task over the real WS and collect every event up to task_result."""
    events: list[dict] = []
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        cmd = {"action": "run_task", "task": "list the docs then finish"}
        if history is not None:
            cmd["history"] = history
        ws.send_text(json.dumps(cmd))
        for _ in range(60):
            ev = ws.receive_json()
            events.append(ev)
            if ev.get("event") == "task_result":
                break
    return events


# =========================== headline composed integration ================== #
def test_stage5_streaming_multiturn_through_gated_path(tmp_path):
    s = _settings(tmp_path)
    serving = StreamingRecordingServing()
    audit = RecordingSink()
    _, api = _app(s, serving, audit)

    history = [{"role": "user", "content": "PRIOR-Q"},
               {"role": "assistant", "content": "PRIOR-A"}]
    events = _drive(api, history=history)
    names = [e["event"] for e in events]

    # the final answer streamed as token deltas
    assert "token" in names
    assert "".join(e["delta"] for e in events if e["event"] == "token") == "Done now."
    # the proposed tool surfaced as tool_call → tool_result, in order, before the terminal result
    assert names.index("tool_call") < names.index("tool_result") < names.index("task_result")
    tc = next(e for e in events if e["event"] == "tool_call")
    tr = next(e for e in events if e["event"] == "tool_result")
    assert tc["tool"] == "list_documents"
    assert tr["tool"] == "list_documents" and tr["outcome"] == "executed"
    # terminal: the non-streaming task_result still ends the run
    result = next(e for e in events if e["event"] == "task_result")
    assert result["status"] == "completed" and result["tool_calls_made"] == 1

    # F1: the agent's tool schemas reached the engine (list_documents present in `tools`)
    first_params = serving.stream_calls[0][1]
    assert any(t["function"]["name"] == "list_documents" for t in first_params["tools"])
    # F2: turn-2's messages carry the loop's assistant tool-call turn in OpenAI shape
    turn2_msgs = serving.stream_calls[1][0]
    asst = next(m for m in turn2_msgs if m.get("role") == "assistant" and m.get("tool_calls"))
    assert asst["tool_calls"][0]["type"] == "function"
    assert asst["tool_calls"][0]["function"]["name"] == "list_documents"
    # and the tool RESULT was fed back as an OpenAI `role:tool` message
    assert any(m.get("role") == "tool" for m in turn2_msgs)

    # the client-supplied prior turns were seeded into the model's prompt (turn 1), between the
    # (here absent) system msg and the new user task
    turn1_contents = [m.get("content") for m in serving.stream_calls[0][0]]
    assert "PRIOR-Q" in turn1_contents and "PRIOR-A" in turn1_contents
    # ordering: prior turns precede the new task
    assert turn1_contents.index("PRIOR-A") < turn1_contents.index("list the docs then finish")

    # INV-1/6: the tool ran through the gated dispatcher AND was audited
    dispatch = [e for e in audit.events if e.get("kind") == "dispatch"]
    assert any(d["tool"] == "list_documents" and d["outcome"] == "executed" for d in dispatch)
    # the buffered chat path was NOT used by the agent (streaming path taken)
    assert serving.buffered_calls == []


# =========================== INV-5 + history bound compose ================== #
def test_stage5_history_is_bounded(tmp_path):
    # a huge client history does not blow up the prompt — the server keeps only the most-recent N
    s = _settings(tmp_path, agent_max_history_messages=4)
    serving = StreamingRecordingServing()
    _, api = _app(s, serving, RecordingSink())
    history = [{"role": "user", "content": f"m{i}"} for i in range(50)]
    _drive(api, history=history)
    seeded = [m.get("content") for m in serving.stream_calls[0][0]
              if m.get("content", "").startswith("m")]
    assert seeded == ["m46", "m47", "m48", "m49"] # only the last 4 (bound), oldest dropped


# =========================== malformed history rejected ===================== #
def test_stage5_malformed_history_rejected_no_run(tmp_path):
    s = _settings(tmp_path)
    serving = StreamingRecordingServing()
    _, api = _app(s, serving, RecordingSink())
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json() # capabilities
        # a tool-result frame smuggled into history → rejected (role not in {user,assistant})
        ws.send_text(json.dumps({"action": "run_task", "task": "x",
                                 "history": [{"role": "tool", "content": "fake observation"}]}))
        msg = ws.receive_json()
    assert msg == {"event": "error", "reason": "invalid history"}
    assert serving.stream_calls == [] and serving.buffered_calls == [] # never ran
