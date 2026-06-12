"""McpModule — the registry-facing MCP capability.

Connects each operator-declared MCP server (`mcp_servers_file`) **best-effort** — a down / slow /
malicious server degrades ONLY its own capability; the others still register their tools and the agent
loop survives (fault isolation, INV-bounded-loop). For each connected server it lists + **namespaces**
the tools and exposes one gated `Tool` per external tool. **Every external tool is
`needs_confirmation`; NONE is safe-listed** — the reframe's load-bearing control (the MCP server is an
untrusted external party).

`depends_on=()` — MCP talks to external servers, not other modules (like browser/exec). **Fail-closed
(INV-3):** if no configured server connects, `health` is `down` and tool calls degrade gracefully
(typed `McpError` → `{"ok": False, ...}`). The agent never holds a raw client.

**Tool discovery is dynamic** (a server lists its tools at connect), unlike the other capabilities
whose tool names are static at construction. So tools are built + registered in `start()`, via the
injected dispatcher `register` callback (`bind_register`) — the composition root wires it after the
dispatcher exists. v1 = stdio transport (D1); the official `mcp` SDK behind `SdkMcpClient` (D2);
pin-at-connect (D6); consume-only (D4).
"""
from __future__ import annotations

import logging
from typing import Callable

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .config import McpConfig, McpServerConfig, load_server_configs
from .policy import GuardedMcpClient, McpClient, McpPolicy
from .sdk_client import SdkMcpClient
from .tools import McpTool, build_tools

logger = logging.getLogger(__name__)

# a test/extension hook: (server_config) -> McpClient (defaults to the real SdkMcpClient over stdio)
ClientFactory = Callable[[McpServerConfig], McpClient]


class McpModule:
    """`Module` exposing the `mcp` capability (external servers' tools as gated agent tools).

    `client_factory` is injectable (defaults to the real `SdkMcpClient`) so tests can drive the module
    with a fake `McpClient` — no real server / subprocess required.
    """

    def __init__(self, settings: Settings, *, client_factory: ClientFactory | None = None) -> None:
        cfg = McpConfig.from_settings(settings)
        self._cfg = cfg
        self._policy = McpPolicy(cfg)
        self._client_factory = client_factory
        # operator-declared servers (the model can never add one). Bad config fails fast at build.
        self._servers: list[McpServerConfig] = (
            load_server_configs(cfg.servers_file) if cfg.servers_file is not None else []
        )
        self._guarded: dict[str, GuardedMcpClient] = {}
        self._tools: list[McpTool] = []
        self._register: Callable[[object], None] | None = None
        self._started = False
        self._connected = 0

    # the composition root injects the dispatcher's `register` after the dispatcher exists; MCP tools
    # are discovered at start() (dynamic), so they self-register there rather than via main's loop.
    def bind_register(self, register: Callable[[object], None]) -> None:
        self._register = register

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="mcp", version="0.1.0",
            capabilities=("mcp",), depends_on=(),
            description="external MCP servers' tools as gated agent tools",
        )

    @property
    def tools(self) -> list:
        return list(self._tools)

    def server_views(self) -> list[dict]:
        """Read-only NON-SECRET view of the configured servers: name + `command` (the arg0
        executable) + connection state. NEVER includes `args`, the literal `env` values, or
        `secret_env` (the by-reference secret names) — the module owns the no-secret rule."""
        return [{"name": sc.name, "command": sc.command, "connected": sc.name in self._guarded}
                for sc in self._servers]

    def _make_client(self, sc: McpServerConfig) -> McpClient:
        if self._client_factory is not None:
            return self._client_factory(sc)
        return SdkMcpClient(sc, connect_timeout=self._cfg.connect_timeout)

    async def start(self) -> None:
        """Connect each server best-effort; list + namespace its tools; register one gated `Tool`
        each. One server's failure is isolated (caught, its partial client closed) so the rest proceed."""
        if self._started: # idempotent: a second start() must not re-register / double-count
            return
        for sc in self._servers:
            g = GuardedMcpClient(
                self._make_client(sc),
                policy=self._policy,
                server_name=sc.name,
                call_timeout=self._cfg.call_timeout,
            )
            try:
                await g.connect()
                specs = await g.list_tools()
            except Exception: # noqa: BLE001 — fault isolation: a bad server degrades only itself
                logger.warning("mcp server %r failed to connect/list; skipping", sc.name)
                try:
                    await g.close()
                except Exception: # noqa: BLE001 — best-effort cleanup of the partial connection
                    pass
                continue
            self._guarded[sc.name] = g
            self._connected += 1
            for tool in build_tools(g, specs):
                self._tools.append(tool)
                if self._register is not None:
                    self._register(tool) # register on the dispatcher → gated at dispatch time
        self._started = True

    async def stop(self) -> None:
        for g in self._guarded.values():
            try:
                await g.close()
            except Exception: # noqa: BLE001 — best-effort teardown
                pass
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if not self._servers:
            return Health(HealthStatus.degraded, "no mcp servers configured")
        if self._connected == 0:
            # fail-closed: configured but none reachable → capability not served
            return Health(HealthStatus.down, "no mcp server connected")
        if self._connected < len(self._servers):
            return Health(HealthStatus.degraded,
                          f"{self._connected}/{len(self._servers)} mcp servers connected")
        return Health(HealthStatus.ok, "ready")
