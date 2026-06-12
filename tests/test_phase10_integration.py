"""— composed multi-capability integration (hermetic).

Drives the REAL composition root (`build_application`) with **all `enable_*` on at once**, faking only
the leaf dependencies via the `BuildOverrides` seam (fake serving/ChatModel, fake SSH transport,
fake fetcher/search, fake exec runner, fake MCP client, scripted agent model). No GPU/Docker/SSH/network.

It exercises canonical CROSS-capability flows through the single gated dispatcher + WS `AgentSession`
(confirm→approve→execute on gated hops), asserting INV-1 routing, HITL, and that every capability's real
module/tool wiring is reachable in the composed app — plus the composed-path gate overhead, and the
error paths (denied approval, a mid-flow module fault).

Run in conda `local-ai-agent-env-1`: `pytest tests/test_phase10_integration.py`.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.browser.fetcher import FetchResult, SearchHit
from local_ai_agent.modules.exec.sandbox import ProcResult
from local_ai_agent.modules.mcp.policy import RawCallResult, RawToolSpec
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.session import AgentSession
from local_ai_agent.modules.safety.gate import Action
from local_ai_agent.modules.storage.connector import GuardedConnector, RemoteEntry, StorageNotFound

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")
_SROOT = "/srv/share"


# --------------------------------------------------------------------------- #
# fakes — one per leaf dependency the composition root would otherwise build real
# --------------------------------------------------------------------------- #
class FakeServing:
    """A fake model-stack module: satisfies the `Module` protocol (Health.ok) AND `ChatModel`
    (`chat`). `spec.name == "llm-serving"` so DocQA's `depends_on=("llm-serving",)` resolves without a
    live llama-server. Replaces the real ModelManager+LLMServing pair via `overrides.serving_module`."""

    def __init__(self, content: str = "A grounded answer from the docs.") -> None:
        self.content = content
        self.calls: list = []
        self._started = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0.0.0", capabilities=("chat",), depends_on=())

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def health(self) -> Health:
        return Health(HealthStatus.ok if self._started else HealthStatus.absent, "fake serving")

    async def chat(self, messages, **params) -> dict:
        self.calls.append((messages, params))
        return {"choices": [{"message": {"content": self.content}}]}


class FakeTransport:
    """In-memory RemoteTransport (no SSH). abspath → bytes."""

    def __init__(self, files=None, dirs=None) -> None:
        self.files = dict(files or {})
        self.dirs = set(dirs or [])
        self.calls: list = []
        self.closed = False

    async def list_dir(self, path):
        self.calls.append(("list_dir", path))
        return [RemoteEntry(p, False, len(b)) for p, b in self.files.items()
                if p.rsplit("/", 1)[0] == path]

    async def stat(self, path):
        self.calls.append(("stat", path))
        if path in self.files:
            return RemoteEntry(path, False, len(self.files[path]))
        if path in self.dirs:
            return RemoteEntry(path, True, 0)
        raise StorageNotFound(path)

    async def read_bytes(self, path):
        self.calls.append(("read_bytes", path))
        if path not in self.files:
            raise StorageNotFound(path)
        return self.files[path]

    async def write_bytes(self, path, data):
        self.calls.append(("write_bytes", path))
        self.files[path] = data

    async def delete(self, path):
        self.calls.append(("delete", path))
        self.files.pop(path, None)

    async def move(self, src, dst):
        self.calls.append(("move", src, dst))
        self.files[dst] = self.files.pop(src, b"")

    async def realpath(self, path):
        return path

    async def close(self):
        self.closed = True


class FakeSearch:
    def __init__(self, hits=None) -> None:
        self.hits = hits or []
        self.calls: list = []

    async def search(self, query, count):
        self.calls.append((query, count))
        return self.hits[:count]

    async def close(self): ...


class MultiFetcher:
    """fetch(url) → FetchResult from a url→result map; missing → raises (degrade)."""

    def __init__(self, by_url) -> None:
        self.by_url = dict(by_url)
        self.fetched: list = []

    async def fetch(self, url):
        self.fetched.append(url)
        r = self.by_url.get(url)
        if r is None:
            raise RuntimeError("not in fake map")
        if isinstance(r, Exception):
            raise r
        return r

    async def close(self): ...


class FakeRunner:
    """Fake docker ProcRunner: records argv, returns canned results (default exit 0)."""

    def __init__(self, results=None, default=None, raise_exc=None) -> None:
        self.calls: list = []
        self._results = list(results or [])
        self._default = default or ProcResult(returncode=0, stdout=b"ok\n")
        self._raise = raise_exc

    async def __call__(self, argv, *, stdin=None, timeout=None, max_output_bytes=None):
        self.calls.append(list(argv))
        if self._raise is not None:
            raise self._raise
        return self._results.pop(0) if self._results else self._default


class FakeMcpClient:
    def __init__(self, tools) -> None:
        self._tools = list(tools)
        self.calls: list = []
        self.closed = False

    async def connect(self): ...

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return RawCallResult(f"ran {name}", False)

    async def close(self):
        self.closed = True


class ScriptedModel:
    """Emits a fixed sequence of AssistantTurns (one .complete() call → one turn)."""

    def __init__(self, turns) -> None:
        self.turns = list(turns)
        self.i = 0

    async def complete(self, messages, tools):
        t = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return t


class FakeChannel:
    """In-process WS channel: auto-responds to each approval_request via `auto` (a callable
    request→'approve'|'deny'). Records every server→client event."""

    def __init__(self, *, auto=None) -> None:
        self.sent: list = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._auto = auto

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)
        if self._auto and data.get("event") == "approval_request":
            verdict = self._auto(data)
            if verdict in ("approve", "deny"):
                await self._inbox.put({"action": verdict, "approval_id": data["approval_id"]})

    async def receive_json(self):
        return await self._inbox.get()

    def events(self, name):
        return [s for s in self.sent if s.get("event") == name]


@dataclass
class FakePrincipal:
    subject: str = "operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"agent:run", "approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


# --------------------------------------------------------------------------- #
# composed-app builder — the REAL build_application with every leaf faked
# --------------------------------------------------------------------------- #
def _tc(i, tool, **args):
    return ToolCall(f"c{i}", tool, args)


def _compose(tmp_path, turns, *, exec_runner=None, fetch_map=None, search_hits=None,
             serving_content="A grounded answer.", read_only=False, audit=None,
             serving_override=None):
    """Build the full composed app (all enable_* on) with fakes injected via BuildOverrides.
    Returns (app, handles) where handles exposes each fake for assertions."""
    docs_root = tmp_path / "docs"
    docs_root.mkdir(exist_ok=True)
    (docs_root / "doc.txt").write_text("The capital of France is Paris.", encoding="utf-8")

    servers_file = tmp_path / "servers.json"
    servers_file.write_text(json.dumps({"servers": [{"name": "fs", "command": "srv-x"}]}), encoding="utf-8")

    settings = Settings(
        **_DIRS,
        enable_agent=True, enable_docqa=True, enable_storage=True,
        enable_browser=True, enable_exec=True, enable_mcp=True,
        docs_root=docs_root,
        mcp_servers_file=servers_file,
        approval_decision_timeout_seconds=5.0,
    )

    serving = serving_override if serving_override is not None else FakeServing(serving_content)
    transport = FakeTransport(files={f"{_SROOT}/remote.txt": b"Berlin is the capital of Germany."},
                              dirs={_SROOT})
    connector = GuardedConnector(transport, allowed_root=_SROOT, read_only=read_only,
                                 max_bytes=1_000_000, realpath=transport.realpath, name="nas")
    fetcher = MultiFetcher(fetch_map or {})
    search = FakeSearch(search_hits or [])
    runner = exec_runner or FakeRunner()
    mcp_client = FakeMcpClient([RawToolSpec("read", "reads a file", {})])

    overrides = BuildOverrides(
        serving_module=serving,
        agent_model=ScriptedModel(turns),
        storage_connectors={"nas": connector},
        browser_fetcher=fetcher,
        browser_search=search,
        exec_runner=runner,
        mcp_client_factory=lambda sc: mcp_client,
        audit_sink=audit,
    )
    app = build_application(settings, overrides=overrides)
    handles = dict(serving=serving, transport=transport, fetcher=fetcher, search=search,
                   runner=runner, mcp=mcp_client, app=app)
    return app, handles


async def _run_task(app, *, auto="approve", task="do it"):
    """Drive one agent task over a FakeChannel; return the channel for event assertions."""
    ch = FakeChannel(auto=(lambda req: auto))
    await AgentSession(ch, FakePrincipal(), app.agent_runtime).run({"action": "run_task", "task": task})
    return ch


def _dispatcher_tools(app):
    return set(app.agent_runtime.dispatcher._tools.keys()) # registered tool names (INV-1: the only path)


# --------------------------------------------------------------------------- #
# composition sanity — all capabilities present behind ONE gated dispatcher (INV-1)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_composed_app_wires_all_capabilities_on_one_dispatcher(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")])
    await app.startup()
    try:
        tools = _dispatcher_tools(app)
        # every capability's tools are registered on the single dispatcher (the only execution path)
        for expected in ("summarize_document", "answer_question", # docqa
                         "storage_read", "storage_write", "summarize_remote", # storage (+remote→docqa)
                         "web_search", "open_url", # browser
                         "run_command", "read_workspace_file", "write_workspace_file", # exec
                         "mcp__fs__read"): # mcp (discovered at start)
            assert expected in tools, f"{expected} not registered"
        # capability health surfaced for the whole composed graph
        names = set(app.registry.health_snapshot())
        assert {"docqa", "storage", "browser", "exec", "mcp", "llm-serving"} <= names
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# F1 — storage → DocQA (the cross-cap regression anchor; both read → no approval)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flow_F1_storage_to_docqa(tmp_path):
    turns = [AssistantTurn(tool_calls=[_tc(1, "summarize_remote", connector="nas", path="remote.txt")]),
             AssistantTurn(text="done")]
    app, h = _compose(tmp_path, turns, serving_content="Germany summary.")
    await app.startup()
    try:
        ch = await _run_task(app, task="summarize the remote file")
        assert ch.events("task_result")[0]["status"] == "completed"
        assert ("read_bytes", f"{_SROOT}/remote.txt") in h["transport"].calls # storage read happened
        assert h["serving"].calls # docqa summarize via chat
        assert not ch.events("approval_request") # both read → safe, no HITL
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# F2 — browser → exec (web_search safe; open_url + run_command gated → 2 approvals)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flow_F2_browser_to_exec(tmp_path):
    page = b"<html><body><article><p>Paris is the capital of France.</p></article></body></html>"
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "web_search", query="capital of France")]),
        AssistantTurn(tool_calls=[_tc(2, "open_url", url="http://site.local/page")]),
        AssistantTurn(tool_calls=[_tc(3, "run_command", command=["echo", "saved"])]),
        AssistantTurn(text="done"),
    ]
    app, h = _compose(
        tmp_path, turns,
        search_hits=[SearchHit("France", "http://site.local/page", "...")],
        fetch_map={"http://site.local/page": FetchResult("http://site.local/page", 200, "text/html", page)},
    )
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="search then run")
        assert ch.events("task_result")[0]["status"] == "completed"
        assert h["search"].calls # browser search (safe) ran
        assert "http://site.local/page" in h["fetcher"].fetched # open_url fetched (gated→approved)
        # the runner records the broker-built docker argv; the command is appended after `docker exec`
        assert any(c[:2] == ["docker", "exec"] and c[-2:] == ["echo", "saved"]
                   for c in h["runner"].calls) # exec run_command (gated→approved) ran
        # exactly the two gated hops prompted for approval (web_search is safe-listed)
        gated = [e["tool"] for e in ch.events("approval_request")]
        assert sorted(gated) == ["open_url", "run_command"]
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# F3 — DocQA → MCP (answer_question safe; external mcp tool untrusted → gated)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flow_F3_docqa_to_mcp(tmp_path):
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "answer_question", path="doc.txt", question="capital of France?")]),
        AssistantTurn(tool_calls=[_tc(2, "mcp__fs__read", path="x")]),
        AssistantTurn(text="done"),
    ]
    app, h = _compose(tmp_path, turns)
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="answer then call mcp")
        assert ch.events("task_result")[0]["status"] == "completed"
        assert h["serving"].calls # docqa answer via chat (safe)
        assert h["mcp"].calls and h["mcp"].calls[0][0] == "read" # external tool ran (gated→approved)
        gated = [e["tool"] for e in ch.events("approval_request")]
        assert gated == ["mcp__fs__read"] # only the untrusted MCP tool prompted
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# F4 — exec → DocQA (run_command gated; answer_question safe)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flow_F4_exec_to_docqa(tmp_path):
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "run_command", command=["ls", "-la"])]),
        AssistantTurn(tool_calls=[_tc(2, "answer_question", path="doc.txt", question="what is the capital of France?")]),
        AssistantTurn(text="done"),
    ]
    app, h = _compose(tmp_path, turns)
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="run then answer")
        assert ch.events("task_result")[0]["status"] == "completed"
        assert any(c[:2] == ["docker", "exec"] and c[-2:] == ["ls", "-la"]
                   for c in h["runner"].calls) # exec ran (gated→approved)
        assert h["serving"].calls # docqa answered (safe)
        assert [e["tool"] for e in ch.events("approval_request")] == ["run_command"]
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# error class — denied approval halts only that action; a module fault is isolated
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_denied_approval_blocks_only_that_action(tmp_path):
    turns = [AssistantTurn(tool_calls=[_tc(1, "run_command", command=["rm", "file"])]),
             AssistantTurn(text="aborted")]
    app, h = _compose(tmp_path, turns)
    await app.startup()
    try:
        ch = await _run_task(app, auto="deny", task="try to run")
        assert ch.events("approval_request") # it was gated
        # DENIED → the command was never handed to the runner (startup docker calls don't contain "file")
        assert not any("file" in c for c in h["runner"].calls)
        assert ch.events("task_result")[0]["status"] == "completed" # loop ends gracefully, no crash
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_midflow_module_fault_is_isolated(tmp_path):
    # the exec runner crashes; the run_command tool returns a graceful error and the loop continues
    # to the next capability (DocQA) — no crash, no cascade.
    turns = [
        AssistantTurn(tool_calls=[_tc(1, "run_command", command=["boom"])]),
        AssistantTurn(tool_calls=[_tc(2, "answer_question", path="doc.txt", question="what is the capital of France?")]),
        AssistantTurn(text="done"),
    ]
    app, h = _compose(tmp_path, turns, exec_runner=FakeRunner(raise_exc=RuntimeError("docker exploded")))
    await app.startup()
    try:
        ch = await _run_task(app, auto="approve", task="faulty exec then answer")
        assert ch.events("task_result")[0]["status"] == "completed" # bounded, no crash
        assert h["serving"].calls # DocQA still worked (no cascade)
        # no raw error/host detail leaked into the task_result answer
        answer = ch.events("task_result")[0].get("answer", "")
        assert "docker exploded" not in answer
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# perf — composed-path gate overhead (recorded for system/deployment.md)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_composed_gate_overhead_is_bounded(tmp_path):
    app, h = _compose(tmp_path, [AssistantTurn(text="idle")])
    await app.startup()
    try:
        gate = app.agent_runtime.gate
        safe = Action("read_workspace_file", {"path": "a.txt"})
        confirm = Action("run_command", {"command": ["echo", "hi"]})
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            gate.classify(safe)
            gate.classify(confirm)
        per_call_us = (time.perf_counter() - t0) / (2 * N) * 1e6
        # the full composed gate (all capability safe-tools + every rule) classifies in well under
        # 1ms/call — recorded as the perf baseline for deployment.md (generous ceiling, not a tight SLA).
        assert per_call_us < 1000, f"gate classify {per_call_us:.1f}µs/call exceeds 1ms"
    finally:
        await app.shutdown()
