#!/usr/bin/env python
"""Operator smoke for MCP (real stdio reference server). NOT part of the hermetic suite.

Connects to a REAL MCP server over stdio through the full guarded path (`SdkMcpClient` →
`GuardedMcpClient`), lists its tools (namespaced `mcp__<server>__<tool>`), optionally calls one, and
tears down cleanly. Mirrors `scripts/smoke_exec.py` (operator-run, real dependency).

Defaults to the reference "everything" server via npx (Node 22 is in the conda env):
    MCP_SMOKE_COMMAND=npx MCP_SMOKE_ARGS='-y @modelcontextprotocol/server-everything'

Run (from project root, in the conda env):
    conda run -n local-ai-agent-env-1 python scripts/smoke_mcp.py
Override the server:
    MCP_SMOKE_COMMAND=/path/to/server MCP_SMOKE_ARGS='--flag x' \
        conda run -n local-ai-agent-env-1 python scripts/smoke_mcp.py
Optionally call a tool: MCP_SMOKE_CALL_TOOL=echo MCP_SMOKE_CALL_ARGS='{"message":"hi"}'
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from local_ai_agent.modules.mcp import (
    GuardedMcpClient,
    McpConfig,
    McpError,
    McpPolicy,
    McpServerConfig,
    SdkMcpClient,
)


def _server_config() -> McpServerConfig:
    command = os.environ.get("MCP_SMOKE_COMMAND", "npx")
    args = os.environ.get("MCP_SMOKE_ARGS", "-y @modelcontextprotocol/server-everything").split()
    return McpServerConfig(name="smoke", command=command, args=args)


async def main() -> int:
    cfg = _server_config()
    policy = McpPolicy(McpConfig(servers_file=None, call_timeout=30.0, connect_timeout=30.0,
                                 max_result_bytes=1_000_000, max_description_chars=4000))
    client = SdkMcpClient(cfg, connect_timeout=30.0)
    guarded = GuardedMcpClient(client, policy=policy, server_name="smoke", call_timeout=30.0)

    checks: list[tuple[str, bool]] = []
    print(f"[smoke] connecting to: {cfg.command} {' '.join(cfg.args)}")
    try:
        await guarded.connect()
        checks.append(("connect", True))

        tools = await guarded.list_tools()
        checks.append(("list_tools non-empty", len(tools) > 0))
        checks.append(("all namespaced mcp__smoke__*",
                       all(t.namespaced_name.startswith("mcp__smoke__") for t in tools)))
        print(f"[smoke] {len(tools)} tools: " + ", ".join(t.namespaced_name for t in tools[:12])
              + (" …" if len(tools) > 12 else ""))

        call_tool = os.environ.get("MCP_SMOKE_CALL_TOOL")
        if call_tool:
            ns = f"mcp__smoke__{call_tool}"
            call_args = json.loads(os.environ.get("MCP_SMOKE_CALL_ARGS", "{}"))
            res = await guarded.call_tool(ns, call_args)
            print(f"[smoke] {ns} → is_error={res.is_error} truncated={res.truncated} "
                  f"content[:200]={res.content[:200]!r}")
            checks.append((f"call {ns}", isinstance(res.content, str)))
    except McpError as exc:
        print(f"[smoke] FAILED with typed error: {type(exc).__name__}: {exc}")
        checks.append(("no unexpected error", False))
    finally:
        await guarded.close()
        checks.append(("clean teardown", True))

    print("\n[smoke] results:")
    ok = True
    for name, passed in checks:
        print(f" {'PASS' if passed else 'FAIL'} {name}")
        ok = ok and passed
    print(f"\n[smoke] {'ALL PASS' if ok else 'FAILURES PRESENT'} ({sum(p for _, p in checks)}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
