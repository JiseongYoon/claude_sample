"""MCP gated tools.

Each external MCP tool is wrapped as one `Tool` the agent reaches through the dispatcher.
Unlike the other capabilities, MCP tool NAMES are **discovered at connect** (the server lists them),
so `McpModule.start()` builds these and registers them on the dispatcher — they are not known at
construction time.

**The reframe, enforced here:** every external tool is `needs_confirmation` and **NONE is safe-listed**
(`MCP_SAFE_TOOL_NAMES = frozenset()`). The name is the namespaced `mcp__<server>__<tool>` (so it can
never shadow a native tool, and the gate's `confirm.mcp_external` rule / fail-safe default both force
confirm). The agent never holds a raw client — only these tools, behind the gate. Every known failure
→ a graceful `{"ok": False, "error": <type-only>}`; the `McpError` message is already
internal-detail-scrubbed by the policy/guard layer.
"""
from __future__ import annotations

from typing import Any

from .policy import GuardedMcpClient, McpError, McpToolSpec

# Intentionally EMPTY: external MCP tools are an UNTRUSTED server's tools — never auto-run. They all
# fall through to the gate's `needs_confirmation` (the `confirm.mcp_external` rule + the fail-safe
# default). This is the load-bearing control of .
MCP_SAFE_TOOL_NAMES: frozenset[str] = frozenset()


class McpTool:
    """`mcp__<server>__<tool>` — one external tool, called through `GuardedMcpClient` (namespaced +
    result-bounded + timeout). Gated `needs_confirmation` (never safe-listed). `description` /
    `input_schema` are the server-provided, policy-bounded, UNTRUSTED metadata (kept for a future
    tool-schema-surfacing step; the gate, not this metadata, is the security control)."""

    def __init__(self, guarded: GuardedMcpClient, spec: McpToolSpec) -> None:
        self.name = spec.namespaced_name
        self.server = spec.server
        self.description = spec.description # bounded + untrusted (policy already capped it)
        self.input_schema = spec.input_schema
        self._guarded = guarded

    async def run(self, args: dict) -> Any:
        try:
            res = await self._guarded.call_tool(self.name, args if isinstance(args, dict) else {})
        except McpError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "content": res.content, # untrusted server output, bounded by the policy
            "is_error": res.is_error, # a server-reported tool failure is DATA, not a crash
            "truncated": res.truncated,
        }


def build_tools(guarded: GuardedMcpClient, specs: list[McpToolSpec]) -> list[McpTool]:
    """Build one `McpTool` per listed spec for a single connected server."""
    return [McpTool(guarded, spec) for spec in specs]
