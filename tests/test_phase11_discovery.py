"""— read-only discovery APIs.

Endpoints: `GET /model/files` · `GET /storage/connectors` · `GET /mcp/servers` · `GET /tools`.
All read-only (no authority), authenticated, and — the security crux — they leak **no secret**
(storage `auth`/`password_env`/`key_path`; MCP `env`/`secret_env` values) and **no absolute path**
(`/model/files` returns bare names). A missing module degrades to an empty list, not an error.

Hermetic: TestClient over `create_gateway`; the model-manager is load-on-demand (no launch), storage
uses a config file with a SECRET set (SSHTransport is lazy → no connect), MCP + the tool roster use a
fake client. No GPU / SSH / subprocess / network. Run in conda `local-ai-agent-env-1`:
`pytest tests/test_phase11_discovery.py`.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.mcp.policy import RawCallResult, RawToolSpec
from local_ai_agent.modules.model_manager.module import ModelManagerModule
from local_ai_agent.modules.storage.module import StorageModule

API_KEY = "k" * 40
JWT = "s" * 40
HDR = {"X-API-Key": API_KEY}


def _settings(d, **over) -> Settings:
    base = dict(_env_file=None, model_safetensors_dir=str(d), model_gguf_dir=str(d),
                auth_enabled=True, api_key=API_KEY, jwt_secret=JWT)
    base.update(over)
    return Settings(**base)


class FakeServing:
    """Minimal fake model-stack module: satisfies Module + ChatModel so DocQA's
    `depends_on=("llm-serving",)` resolves with no live llama-server."""

    def __init__(self) -> None:
        self._s = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self) -> None:
        self._s = True

    async def stop(self) -> None:
        self._s = False

    def health(self) -> Health:
        return Health(HealthStatus.ok if self._s else HealthStatus.absent, "fake")

    async def chat(self, messages, **params) -> dict:
        return {"choices": [{"message": {"content": "x"}}]}


class FakeMcpClient:
    """Fake McpClient — connects without a subprocess, lists a canned tool."""

    def __init__(self, tools) -> None:
        self._tools = list(tools)
        self.closed = False

    async def connect(self):
        ...

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, args):
        return RawCallResult(f"ran {name}", False)

    async def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# GET /model/files
# --------------------------------------------------------------------------- #
def test_model_files_lists_sorted_bare_names(tmp_path):
    (tmp_path / "b.gguf").write_bytes(b"\x00")
    (tmp_path / "a.gguf").write_bytes(b"\x00")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[ModelManagerModule(s)]), s)) as c:
        r = c.get("/model/files", headers=HDR)
        assert r.status_code == 200
        files = r.json()["files"]
        assert files == ["a.gguf", "b.gguf"] # sorted, .txt excluded
        assert all("/" not in f and "\\" not in f for f in files) # bare names, no path leak


def test_model_files_503_when_model_stack_absent(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[]), s)) as c:
        assert c.get("/model/files", headers=HDR).status_code == 503


def test_model_files_empty_dir_no_crash(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[ModelManagerModule(s)]), s)) as c:
        assert c.get("/model/files", headers=HDR).json()["files"] == []


def test_model_files_requires_auth(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[ModelManagerModule(s)]), s)) as c:
        assert c.get("/model/files").status_code in (401, 403)


# --------------------------------------------------------------------------- #
# GET /storage/connectors — NON-SECRET projection (the crux)
# --------------------------------------------------------------------------- #
def _storage_settings(tmp_path) -> Settings:
    conns = {"connectors": [{
        "name": "nas", "kind": "ssh", "host": "10.0.0.5", "port": 2222, "username": "svc",
        "auth": {"password_env": "NAS_SECRET_PW"}, # by-reference secret NAME — must NOT serialize
        "allowed_root": "/srv/share", "read_only": True,
    }]}
    f = tmp_path / "connectors.json"
    f.write_text(json.dumps(conns), encoding="utf-8")
    return _settings(tmp_path, storage_connectors_file=f)


def test_storage_connectors_nonsecret_projection(tmp_path):
    s = _storage_settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[StorageModule(s)]), s)) as c:
        r = c.get("/storage/connectors", headers=HDR)
        assert r.status_code == 200
        body = r.json()
        assert "health" in body
        conns = body["connectors"]
        assert len(conns) == 1
        v = conns[0]
        assert v["name"] == "nas" and v["kind"] == "ssh" and v["host"] == "10.0.0.5"
        assert v["read_only"] is True and v["allowed_root"] == "/srv/share"
        # SECURITY: no secret key/name/value anywhere in the wire response
        raw = r.text
        for bad in ("password_env", "key_path", "NAS_SECRET_PW"):
            assert bad not in raw, f"secret marker {bad!r} leaked"


def test_storage_connectors_empty_when_module_absent(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[]), s)) as c:
        assert c.get("/storage/connectors", headers=HDR).json() == {"connectors": [], "health": None}


def test_storage_connectors_requires_auth(tmp_path):
    s = _storage_settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[StorageModule(s)]), s)) as c:
        assert c.get("/storage/connectors").status_code in (401, 403)


# --------------------------------------------------------------------------- #
# GET /mcp/servers — NON-SECRET projection (the crux)
# --------------------------------------------------------------------------- #
def _mcp_settings(tmp_path) -> Settings:
    servers = {"servers": [{
        "name": "fs", "command": "srv-x", "args": ["--flag"],
        "env": {"LANG": "C"}, "secret_env": ["MCP_SECRET_TOKEN"], # must NOT serialize
    }]}
    f = tmp_path / "servers.json"
    f.write_text(json.dumps(servers), encoding="utf-8")
    return _settings(tmp_path, enable_mcp=True, mcp_servers_file=f)


def _fake_factory(_sc):
    return FakeMcpClient([RawToolSpec("read", "reads a file", {})])


def test_mcp_servers_nonsecret_projection(tmp_path):
    s = _mcp_settings(tmp_path)
    app = build_application(s, overrides=BuildOverrides(mcp_client_factory=_fake_factory))
    with TestClient(create_gateway(app, s)) as c:
        r = c.get("/mcp/servers", headers=HDR)
        assert r.status_code == 200
        servers = r.json()["servers"]
        assert len(servers) == 1
        v = servers[0]
        assert v["name"] == "fs" and v["command"] == "srv-x" and v["connected"] is True
        raw = r.text
        for bad in ("secret_env", "MCP_SECRET_TOKEN", "--flag", "LANG"):
            assert bad not in raw, f"secret/non-projected marker {bad!r} leaked"


def test_mcp_servers_empty_when_module_absent(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[]), s)) as c:
        assert c.get("/mcp/servers", headers=HDR).json() == {"servers": [], "health": None}


# --------------------------------------------------------------------------- #
# GET /tools — roster (name · tier · capability)
# --------------------------------------------------------------------------- #
def test_tool_roster_tiers_and_capability(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "d.txt").write_text("hi", encoding="utf-8")
    servers = {"servers": [{"name": "fs", "command": "srv-x"}]}
    f = tmp_path / "servers.json"
    f.write_text(json.dumps(servers), encoding="utf-8")
    s = _settings(tmp_path, enable_agent=True, enable_docqa=True, enable_mcp=True,
                  docs_root=docs, mcp_servers_file=f)
    app = build_application(s, overrides=BuildOverrides(
        serving_module=FakeServing(), mcp_client_factory=_fake_factory))
    with TestClient(create_gateway(app, s)) as c:
        r = c.get("/tools", headers=HDR)
        assert r.status_code == 200
        tools = {t["name"]: t for t in r.json()["tools"]}
        # DocQA tools are safe-listed
        assert tools["answer_question"]["tier"] == "safe"
        assert tools["answer_question"]["capability"] == "docqa"
        # MCP tools are namespaced + needs_confirmation + capability "mcp"
        mcp_names = [n for n in tools if n.startswith("mcp__")]
        assert mcp_names, "expected at least one mcp__ tool"
        assert all(tools[n]["tier"] == "needs_confirmation" for n in mcp_names)
        assert tools[mcp_names[0]]["capability"] == "mcp"


def test_tools_empty_when_agent_absent(tmp_path):
    s = _settings(tmp_path)
    with TestClient(create_gateway(Application(modules=[]), s)) as c:
        assert c.get("/tools", headers=HDR).json() == {"tools": []}
