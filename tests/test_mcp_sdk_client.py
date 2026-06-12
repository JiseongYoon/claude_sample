"""— SdkMcpClient over the official mcp SDK, stdio (hermetic).

Drives the concrete `SdkMcpClient` against the SDK's **in-memory** client/server transport (a real
protocol round-trip via FastMCP, no subprocess) and a stub session for timing. Confirms: connect +
pin tools (D6), call mapping, a server-side tool error returned as DATA (not raised), close
idempotency, connect timeout, and unreachable→`McpUnavailable` (the only test that touches a real
stdio spawn — a bogus command, no network). End-to-end through `GuardedMcpClient` proves the
namespacing/bounding layer composes with the real client.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from local_ai_agent.modules.mcp import (
    GuardedMcpClient,
    McpConfig,
    McpPolicy,
    McpServerConfig,
    McpTimeout,
    McpUnavailable,
    RawCallResult,
    RawToolSpec,
    SdkMcpClient,
)

mcp_sdk = pytest.importorskip("mcp", reason="official mcp SDK ([mcp] extra) not installed")
from mcp.server.fastmcp import FastMCP # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session as connected # noqa: E402


def _make_server() -> FastMCP:
    s = FastMCP("test")

    @s.tool(description="echo a value back")
    def echo(value: str) -> str:
        return value

    @s.tool()
    def boom() -> str:
        raise ValueError("server-side tool failure SECRET=xyz host=10.0.0.1")

    return s


def _inmem_factory(server: FastMCP):
    @asynccontextmanager
    async def factory():
        async with connected(server._mcp_server) as session:
            yield session
    return factory


def _policy(max_desc=4000, max_result=1_000_000) -> McpPolicy:
    return McpPolicy(McpConfig(servers_file=None, call_timeout=30.0, connect_timeout=20.0,
                               max_result_bytes=max_result, max_description_chars=max_desc))


def _cfg(command="mcp-server-x", args=None) -> McpServerConfig:
    return McpServerConfig(name="fs", command=command, args=args or [])


# --------------------------------------------------------------------------- #
# NORMAL class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_connect_list_pin_inmemory():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    await client.connect()
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert {"echo", "boom"} <= names
    assert all(isinstance(t, RawToolSpec) for t in tools)
    # pinned: repeated list_tools returns the same captured set (D6 — no re-query semantics)
    assert {t.name for t in await client.list_tools()} == names
    await client.close()


@pytest.mark.asyncio
async def test_call_tool_round_trip_inmemory():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    await client.connect()
    res = await client.call_tool("echo", {"value": "hi"})
    assert isinstance(res, RawCallResult) and res.content == "hi" and not res.is_error
    await client.close()


@pytest.mark.asyncio
async def test_end_to_end_through_guarded():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    g = GuardedMcpClient(client, policy=_policy(), server_name="fs", call_timeout=10.0)
    await g.connect()
    specs = await g.list_tools()
    assert "mcp__fs__echo" in {s.namespaced_name for s in specs}
    out = await g.call_tool("mcp__fs__echo", {"value": "world"})
    assert out.content == "world" and not out.is_error and not out.truncated
    await g.close()


@pytest.mark.asyncio
async def test_close_idempotent():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    await client.connect()
    await client.close()
    await client.close() # safe twice


# --------------------------------------------------------------------------- #
# ERROR / SECURITY class
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_server_tool_error_is_data_not_exception():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    await client.connect()
    res = await client.call_tool("boom", {})
    assert res.is_error is True # a server-reported failure → data, NOT a raised exception
    assert isinstance(res.content, str) # the loop survives; content is untrusted (bounded by policy)
    await client.close()


@pytest.mark.asyncio
async def test_call_or_list_before_connect_unavailable():
    client = SdkMcpClient(_cfg(), session_factory=_inmem_factory(_make_server()))
    with pytest.raises(McpUnavailable):
        await client.list_tools()
    with pytest.raises(McpUnavailable):
        await client.call_tool("echo", {})


@pytest.mark.asyncio
async def test_connect_timeout_with_slow_stub():
    import asyncio

    class _SlowSession:
        async def list_tools(self):
            await asyncio.sleep(1.0)
            return type("R", (), {"tools": []})()
        async def call_tool(self, name, args):
            return type("R", (), {"content": [], "isError": False})()

    @asynccontextmanager
    async def slow_factory():
        yield _SlowSession()

    client = SdkMcpClient(_cfg(), connect_timeout=0.05, session_factory=slow_factory)
    with pytest.raises(McpTimeout):
        await client.connect()


@pytest.mark.asyncio
async def test_unreachable_real_stdio_maps_to_unavailable():
    # the ONLY real-stdio test: a bogus command fails to spawn → McpUnavailable (no crash, no leak).
    client = SdkMcpClient(_cfg(command="this-command-does-not-exist-xyz-123"), connect_timeout=5.0)
    with pytest.raises(McpUnavailable) as ei:
        await client.connect()
    assert "this-command-does-not-exist" not in str(ei.value) # type-only, no command leak


@pytest.mark.asyncio
async def test_call_failure_maps_to_unavailable_no_leak():
    class _BadSession:
        async def list_tools(self):
            return type("R", (), {"tools": [type("T", (), {"name": "x", "description": "", "inputSchema": {}})()]})()
        async def call_tool(self, name, args):
            raise RuntimeError("call boom token=SEKRET host=10.0.0.1")

    @asynccontextmanager
    async def bad_factory():
        yield _BadSession()

    client = SdkMcpClient(_cfg(), session_factory=bad_factory)
    await client.connect()
    with pytest.raises(McpUnavailable) as ei:
        await client.call_tool("x", {})
    msg = str(ei.value)
    for leak in ("SEKRET", "token", "10.0.0.1", "boom"):
        assert leak not in msg
    await client.close()
