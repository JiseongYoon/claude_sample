"""MCP capability — an MCP *client* consuming external servers' tools as GATED agent tools.

**The reframe:** an MCP server is an UNTRUSTED external party. The canonical MCP client trusts the
server (auto-exposes its tools, feeds its descriptions to the model); we must NOT. Every external tool
routes through OUR gate (default `needs_confirmation`, never safe-listed); tool descriptions +
results are bounded + treated as untrusted input; the client advertises NO capabilities (consume-only,
no server→our-model channel). v1 transport = stdio (HTTP/SSE → sub-phase 8.1); servers are
operator-declared (the model never spawns one).

 ships the pure, daemon-free / SDK-free / network-free security crux: the `McpClient`/
`GuardedMcpClient` seam, `McpPolicy` (namespacing + untrusted-output bounding + empty client caps,
fail-closed), and the config (`McpServerConfig` + `load_server_configs` + `McpConfig`) — all testable
with an injected `FakeMcpClient`, no real MCP server required.
"""
from __future__ import annotations

from .config import McpConfig, McpServerConfig, load_server_configs, resolve_server_env
from .sdk_client import SdkMcpClient
from .policy import (
    ClientCapabilities,
    GuardedMcpClient,
    McpBlocked,
    McpCallResult,
    McpClient,
    McpError,
    McpPolicy,
    McpTimeout,
    McpTooLarge,
    McpToolSpec,
    McpUnavailable,
    RawCallResult,
    RawToolSpec,
)

__all__ = [
    "McpConfig",
    "McpServerConfig",
    "load_server_configs",
    "resolve_server_env",
    "ClientCapabilities",
    "GuardedMcpClient",
    "McpBlocked",
    "McpCallResult",
    "McpClient",
    "McpError",
    "McpPolicy",
    "McpTimeout",
    "McpTooLarge",
    "McpToolSpec",
    "McpUnavailable",
    "RawCallResult",
    "RawToolSpec",
    "SdkMcpClient",
]
