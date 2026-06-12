"""— McpModule + namespaced gated tools + main wiring (daemon-free).

A fake `McpClient` (injected via `client_factory`) drives the module with no real server/subprocess.
Covers: best-effort multi-server connect (one-down degrades only itself), namespaced gated tool
registration, **gate routing via the REAL `SafetyGate`** (every `mcp__*` tool → `needs_confirmation`,
no native-tool shadowing, NO safe-list leak), graceful tool errors, and `enable_mcp` flag isolation in
the composition root.
"""
from __future__ import annotations

import json

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.mcp.module import McpModule
from local_ai_agent.modules.mcp.policy import McpError, RawCallResult, RawToolSpec
from local_ai_agent.modules.mcp.tools import MCP_SAFE_TOOL_NAMES, McpTool
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


class FakeClient:
    """A fake raw McpClient — connects (unless `fail`), lists canned tools, echoes calls."""

    def __init__(self, tools, *, fail_connect=False, fail_call=False):
        self._tools = tools
        self._fail_connect = fail_connect
        self._fail_call = fail_call
        self.closed = False

    async def connect(self):
        if self._fail_connect:
            raise RuntimeError("server unreachable cmd=/secret token=abc")

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, args):
        if self._fail_call:
            raise RuntimeError("call boom token=SEKRET")
        return RawCallResult(content=f"ran {name} with {sorted(args)}", is_error=False)

    async def close(self):
        self.closed = True


def _servers_file(tmp_path, servers):
    f = tmp_path / "servers.json"
    f.write_text(json.dumps({"servers": servers}))
    return f


def _settings(tmp_path, servers, **extra):
    return Settings(enable_mcp=True, mcp_servers_file=str(_servers_file(tmp_path, servers)),
                    **extra, **_DIRS)


def _two_server_factory():
    tools = {
        "fs": [RawToolSpec("read", "read a file", {}), RawToolSpec("list_files", "list", {})],
        "web": [RawToolSpec("get", "http get", {})],
        "down": [RawToolSpec("x", "", {})],
    }
    def factory(sc):
        return FakeClient(tools.get(sc.name, []), fail_connect=(sc.name == "down"))
    return factory


# --------------------------------------------------------------------------- #
# NORMAL class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multi_server_connect_and_namespace(tmp_path):
    s = _settings(tmp_path, [{"name": "fs", "command": "srv-fs"}, {"name": "web", "command": "srv-web"}])
    registered = []
    m = McpModule(s, client_factory=_two_server_factory())
    m.bind_register(registered.append)
    await m.start()
    names = {t.name for t in m.tools}
    assert names == {"mcp__fs__read", "mcp__fs__list_files", "mcp__web__get"}
    assert {t.name for t in registered} == names # all registered on the (fake) dispatcher
    assert m.health().status.name == "ok"
    await m.stop()


@pytest.mark.asyncio
async def test_tool_run_round_trip(tmp_path):
    s = _settings(tmp_path, [{"name": "fs", "command": "srv-fs"}])
    m = McpModule(s, client_factory=_two_server_factory())
    await m.start()
    read = next(t for t in m.tools if t.name == "mcp__fs__read")
    out = await read.run({"path": "a.txt"})
    assert out["ok"] is True and out["is_error"] is False
    assert out["content"].startswith("ran read")
    await m.stop()


# --------------------------------------------------------------------------- #
# ERROR / SECURITY class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_one_server_down_degrades_only_itself(tmp_path):
    s = _settings(tmp_path, [{"name": "fs", "command": "srv-fs"}, {"name": "down", "command": "srv-down"}])
    registered = []
    m = McpModule(s, client_factory=_two_server_factory())
    m.bind_register(registered.append)
    await m.start() # must NOT raise despite 'down' failing
    names = {t.name for t in m.tools}
    assert names == {"mcp__fs__read", "mcp__fs__list_files"} # only the healthy server's tools
    assert m.health().status.name == "degraded" # 1/2 connected
    await m.stop()


def test_gate_routes_all_mcp_tools_to_confirm_no_safelist_leak():
    gate = SafetyGate() # defaults include the confirm.mcp_external rule
    for name in ("mcp__fs__read", "mcp__fs__list_files", "mcp__evil__run_command",
                 "mcp__evil__read_file", "mcp__x__web_search"):
        d = gate.classify(Action(name, {}))
        assert d.verdict is Verdict.needs_confirmation, name
        assert name not in DEFAULT_SAFE_TOOLS
    # the namespacing means an external tool named like a native safe tool does NOT match the allowlist
    assert "list_files" in DEFAULT_SAFE_TOOLS and "mcp__fs__list_files" not in DEFAULT_SAFE_TOOLS
    # and the explicit rule is what fired
    assert gate.classify(Action("mcp__fs__read", {})).rule == "confirm.mcp_external"


def test_mcp_safe_tool_names_is_empty():
    assert MCP_SAFE_TOOL_NAMES == frozenset()


@pytest.mark.asyncio
async def test_tool_call_error_is_graceful_no_leak(tmp_path):
    s = _settings(tmp_path, [{"name": "fs", "command": "srv-fs"}])
    def factory(sc):
        return FakeClient([RawToolSpec("read", "", {})], fail_call=True)
    m = McpModule(s, client_factory=factory)
    await m.start()
    read = next(t for t in m.tools if t.name == "mcp__fs__read")
    out = await read.run({"path": "a"})
    assert out["ok"] is False
    for leak in ("SEKRET", "token", "boom"):
        assert leak not in out["error"] # type-only, no server detail leaked
    await m.stop()


@pytest.mark.asyncio
async def test_no_servers_configured_is_degraded(tmp_path):
    s = Settings(enable_mcp=True, **_DIRS) # no servers_file
    m = McpModule(s)
    await m.start()
    assert m.tools == [] and m.health().status.name == "degraded"
    await m.stop()


# --------------------------------------------------------------------------- #
# wiring / flag isolation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path):
    s = _settings(tmp_path, [{"name": "fs", "command": "srv-fs"}])
    registered = []
    m = McpModule(s, client_factory=_two_server_factory())
    m.bind_register(registered.append)
    await m.start()
    await m.start() # second start() must not re-register / double-count
    assert len(m.tools) == 2 and len(registered) == 2
    assert m.health().status.name == "ok"
    await m.stop()


def test_wiring_flag_isolation_off():
    app = build_application(Settings(**_DIRS)) # enable_mcp default False
    assert not any(getattr(m, "spec", None) and m.spec.name == "mcp" for m in app.modules)


def test_wiring_enabled_builds_module(tmp_path):
    f = _servers_file(tmp_path, [{"name": "fs", "command": "srv-fs"}])
    app = build_application(Settings(enable_mcp=True, mcp_servers_file=str(f), **_DIRS))
    mcp_mods = [m for m in app.modules if getattr(m, "spec", None) and m.spec.name == "mcp"]
    assert len(mcp_mods) == 1 # present in the registry, no crash at build
    # tools are discovered at start() (not yet), so .tools is empty pre-start — that's expected
    assert mcp_mods[0].tools == []
