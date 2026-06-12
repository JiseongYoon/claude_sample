"""— attachment-ref wiring (run_task.attachments + direct /chat attachments).

Two paths:
- **WS agent** — `run_task.attachments:[id]` → the SERVER resolves each id → its contained relpath
  and folds an instruction into the system prompt; the agent reads via the EXISTING gated DocQA tools
  (INV-1 unchanged). Unknown/invalid id → typed error event, no run. (driven via `AgentSession.run`)
- **direct /chat** — `attachments:[id]` → the server loads the text (bounded) into a system context
  message; no tool loop. Unknown id → 400; ingestion absent → 503. (driven via TestClient `/chat`)

Hermetic; conda `local-ai-agent-env-1`; asyncio_mode=auto.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.modules.docqa.ingest import IngestPolicy, IngestStore
from local_ai_agent.modules.orchestrator.loop import AssistantTurn
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate

API_KEY = "k" * 40
JWT = "s" * 40
HDR = {"X-API-Key": API_KEY}


# =========================================================================== #
# WS agent path — AgentSession.run with attachments
# =========================================================================== #
class RecordingModel:
    """Captures the system message it receives, then ends the task with a final answer."""

    def __init__(self) -> None:
        self.seen_system: str | None = None

    async def complete(self, messages, tools) -> AssistantTurn:
        for m in messages:
            if m.get("role") == "system":
                self.seen_system = m.get("content")
        return AssistantTurn(text="done")


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        import asyncio
        self._block = asyncio.Event() # receive_json blocks until the reader is cancelled

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def receive_json(self):
        await self._block.wait()
        return {}


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"agent:run", "approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


def _store_with_file(tmp_path, filename="report.txt", content=b"the contained content"):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    pol = IngestPolicy(docs, id_factory=lambda: "abc123")
    store = IngestStore(pol)
    prep = pol.validate(filename, len(content))
    dest = pol.resolve_destination(prep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    store.register(prep, len(content))
    return store, prep.ingest_id, prep.relpath


def _runtime(model, store):
    approvals = PendingApprovals(timeout_seconds=300)
    return AgentRuntime(
        gate=SafetyGate(), approvals=approvals,
        dispatcher=ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[]),
        model=model, ingest_store=store, decision_timeout=5.0, max_steps=4, max_tool_calls=8,
    )


async def test_ws_attachment_folds_relpath_into_system_prompt(tmp_path):
    store, aid, relpath = _store_with_file(tmp_path)
    model = RecordingModel()
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), _runtime(model, store)).run(
        {"action": "run_task", "task": "summarize the attachment", "attachments": [aid]})
    assert model.seen_system is not None
    assert relpath in model.seen_system # the agent is told the exact contained path
    assert any(m.get("event") == "task_result" for m in ch.sent)


async def test_ws_attachment_preserves_explicit_system(tmp_path):
    store, aid, relpath = _store_with_file(tmp_path)
    model = RecordingModel()
    await AgentSession(FakeChannel(), FakePrincipal(), _runtime(model, store)).run(
        {"action": "run_task", "task": "q", "system": "You are terse.", "attachments": [aid]})
    assert "You are terse." in model.seen_system and relpath in model.seen_system


async def test_ws_unknown_attachment_errors_no_run(tmp_path):
    store, _, _ = _store_with_file(tmp_path)
    model = RecordingModel()
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), _runtime(model, store)).run(
        {"action": "run_task", "task": "q", "attachments": ["does-not-exist"]})
    assert any(m.get("event") == "error" and m.get("reason") == "unknown attachment" for m in ch.sent)
    assert model.seen_system is None # never ran the model


async def test_ws_no_attachments_unchanged(tmp_path):
    store, _, _ = _store_with_file(tmp_path)
    model = RecordingModel()
    await AgentSession(FakeChannel(), FakePrincipal(), _runtime(model, store)).run(
        {"action": "run_task", "task": "plain question"})
    assert model.seen_system is None # no system injected when no attachments/system


async def test_ws_invalid_attachments_rejected(tmp_path):
    store, aid, _ = _store_with_file(tmp_path)
    for bad in ("not-a-list", [123], ["ok"] * 17):
        model = RecordingModel()
        ch = FakeChannel()
        await AgentSession(ch, FakePrincipal(), _runtime(model, store)).run(
            {"action": "run_task", "task": "q", "attachments": bad})
        assert any(m.get("event") == "error" for m in ch.sent)
        assert model.seen_system is None


async def test_ws_attachments_unavailable_when_no_store(tmp_path):
    model = RecordingModel()
    ch = FakeChannel()
    await AgentSession(ch, FakePrincipal(), _runtime(model, store=None)).run(
        {"action": "run_task", "task": "q", "attachments": ["abc123"]})
    assert any(m.get("event") == "error" and m.get("reason") == "attachments unavailable"
               for m in ch.sent)


# =========================================================================== #
# direct /chat path — TestClient over the composed app
# =========================================================================== #
class FakeManager:
    """model-manager stand-in: `/chat` only consults `is_serving`."""

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="model-manager", version="0", capabilities=("model-control",), depends_on=())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def health(self) -> Health:
        return Health(HealthStatus.ok, "fake")

    @property
    def is_serving(self) -> bool:
        return True


class RecordingServing:
    """llm-serving stand-in that RECORDS the messages it was given (to assert injected context)."""

    def __init__(self) -> None:
        self.last_messages = None

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def health(self) -> Health:
        return Health(HealthStatus.ok, "fake")

    async def chat(self, messages, **params) -> dict:
        self.last_messages = messages
        return {"choices": [{"message": {"content": "ok"}}]}


def _settings(tmp_path, **over) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(_env_file=None, model_safetensors_dir=str(tmp_path), model_gguf_dir=str(tmp_path),
                auth_enabled=True, api_key=API_KEY, jwt_secret=JWT,
                enable_docqa=True, docs_root=str(docs))
    base.update(over)
    return Settings(**base)


def _chat_app(s, serving, store):
    """Hand-composed app (model-manager + llm-serving + an attached ingest_store) — the /chat
    endpoint requires a model-manager that `build_application`'s serving-override path omits, so
    we compose directly here. The build_application store wiring is covered by 's tests."""
    app = Application(modules=[FakeManager(), serving])
    app.ingest_store = store
    return create_gateway(app, s)


def test_chat_attachment_injects_text(tmp_path):
    s = _settings(tmp_path)
    serving = RecordingServing()
    store, aid, _ = _store_with_file(tmp_path, "note.txt", b"PINEAPPLE-MARKER")
    with TestClient(_chat_app(s, serving, store)) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "what's in the file?"}],
                                  "attachments": [aid]}, headers=HDR)
        assert r.status_code == 200, r.text
        # the uploaded text reached the model as a prepended system context message
        sys_msgs = [m for m in serving.last_messages if m["role"] == "system"]
        assert sys_msgs and "PINEAPPLE-MARKER" in sys_msgs[0]["content"]
        assert serving.last_messages[-1]["content"] == "what's in the file?" # user msg preserved last


def test_chat_unknown_attachment_400(tmp_path):
    s = _settings(tmp_path)
    store, _, _ = _store_with_file(tmp_path)
    with TestClient(_chat_app(s, RecordingServing(), store)) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "x"}],
                                  "attachments": ["nope"]}, headers=HDR)
        assert r.status_code == 400


def test_chat_no_attachments_unchanged(tmp_path):
    s = _settings(tmp_path)
    serving = RecordingServing()
    store, _, _ = _store_with_file(tmp_path)
    with TestClient(_chat_app(s, serving, store)) as c:
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]}, headers=HDR)
        assert r.status_code == 200
        assert all(m["role"] != "system" for m in serving.last_messages) # no context injected


def test_chat_attachment_503_when_ingestion_absent(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_chat_app(s, RecordingServing(), store=None)) as c: # ingest_store None
        r = c.post("/chat", json={"messages": [{"role": "user", "content": "x"}],
                                  "attachments": ["whatever"]}, headers=HDR)
        assert r.status_code == 503
