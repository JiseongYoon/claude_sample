"""SdkMcpClient — the concrete MCP transport over the official `mcp` SDK, stdio.

The MCP analogue of 's `DockerSandbox`, 's `HttpxFetcher`, 's `SSHTransport`:
the one concrete `McpClient` implementation, wrapping the **official `mcp` Python SDK** (D2) on the
**stdio** transport (D1). It spawns the operator-configured server subprocess, runs the `initialize`
handshake advertising **no client capabilities** (D4 — we pass no sampling/roots/elicitation callback,
so the SDK advertises none, structurally), and **pins** the tool list at connect (D6 — a later
`tools/list_changed` does not mutate the pinned set). The SDK is **lazy-imported** and only ever
touched here; tools talk to `GuardedMcpClient`, never to this class or the SDK directly.

Every SDK / transport / subprocess failure is mapped to a typed `McpError` whose message carries only
the exception type — never a server command / token / host (reframe i). A server-reported tool failure
is returned as **data** (`is_error=True` content), not raised (the loop survives — reframe j).

Unit-tested hermetically against the SDK's **in-memory** client/server transport (a real protocol
round-trip, no subprocess) via the injectable `session_factory`; a real stdio reference server is
exercised by `scripts/smoke_mcp.py`.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Awaitable, Callable

from .config import McpServerConfig, resolve_server_env
from .policy import (
    McpBlocked,
    McpError,
    McpTimeout,
    McpUnavailable,
    RawCallResult,
    RawToolSpec,
)

# A test hook: an async callable returning an async-context-manager that yields a connected +
# initialized SDK `ClientSession`. Production leaves it None → the real stdio path is used.
SessionFactory = Callable[[], "AsyncSessionCM"]


class AsyncSessionCM: # pragma: no cover - structural typing alias for an async context manager
    async def __aenter__(self): ...
    async def __aexit__(self, *exc): ...


def _to_raw_tool(t) -> RawToolSpec:
    return RawToolSpec(
        name=t.name,
        description=getattr(t, "description", "") or "",
        input_schema=getattr(t, "inputSchema", None) or {},
    )


def _to_raw_result(res) -> RawCallResult:
    """Flatten a `CallToolResult`'s content blocks to text (non-text blocks summarized by type). The
    text is server-controlled → it is bounded + labelled untrusted by the policy layer."""
    parts: list[str] = []
    for block in (getattr(res, "content", None) or []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(f"[{getattr(block, 'type', 'content')}]")
    return RawCallResult(content="\n".join(parts), is_error=bool(getattr(res, "isError", False)))


class SdkMcpClient:
    """Concrete `McpClient` over the official SDK + stdio. Lifecycle: `connect()` (spawn + initialize +
    pin tools) → `list_tools()` (pinned) / `call_tool()` → `close()` (terminate). Not reusable after
    close; build a new instance to reconnect."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        connect_timeout: float = 20.0,
        session_factory: SessionFactory | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be > 0")
        self._cfg = config
        self._connect_timeout = float(connect_timeout)
        self._session_factory = session_factory # test hook; None → real stdio
        self._environ = environ # injectable host env (for secret_env resolution)
        self._stack: AsyncExitStack | None = None
        self._session = None
        self._pinned: list[RawToolSpec] | None = None # tools captured at connect (D6 pin)

    # -- lifecycle ------------------------------------------------------------ #
    async def _open_real_session(self, stack: AsyncExitStack):
        """The real stdio path: spawn the operator server subprocess and open an SDK session. We pass
        NO sampling/roots/elicitation callback → the SDK advertises no client capabilities (D4)."""
        from mcp import ClientSession, StdioServerParameters # lazy — only when actually connecting
        from mcp.client.stdio import stdio_client

        # auth/secrets by REFERENCE: literal env + `secret_env` names resolved from the host
        # env, fail-closed if a declared secret is unset. The SDK merges this over a CURATED default env
        # (PATH/HOME/…), NOT the full os.environ — so the agent's ambient secrets are never forwarded.
        try:
            env = resolve_server_env(self._cfg, environ=self._environ)
        except ValueError as exc:
            # a declared secret var is unset → fail-closed, type-only (the message names no value)
            raise McpBlocked(f"server env resolution failed: {exc}") from exc
        params = StdioServerParameters(
            command=self._cfg.command,
            args=list(self._cfg.args),
            env=env or None,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=self._connect_timeout)
        return session

    async def connect(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        try:
            if self._session_factory is not None:
                # test path: the factory yields an already-connected + initialized session
                session = await stack.enter_async_context(self._session_factory())
            else:
                session = await self._open_real_session(stack)
            result = await asyncio.wait_for(session.list_tools(), timeout=self._connect_timeout)
            self._pinned = [_to_raw_tool(t) for t in (getattr(result, "tools", None) or [])]
        except McpError:
            await self._safe_aclose(stack)
            raise
        except asyncio.TimeoutError as exc:
            await self._safe_aclose(stack)
            raise McpTimeout(f"mcp connect/list exceeded {self._connect_timeout}s") from exc
        except BaseException as exc: # noqa: BLE001 — anyio ExceptionGroup / spawn failure / etc.
            await self._safe_aclose(stack)
            raise McpUnavailable(f"mcp connect failed: {type(exc).__name__}") from exc
        self._stack = stack
        self._session = session

    async def list_tools(self) -> list[RawToolSpec]:
        if self._pinned is None:
            raise McpUnavailable("not connected")
        return list(self._pinned) # pinned at connect — a mid-session list_changed is ignored (D6)

    async def call_tool(self, name: str, args: dict) -> RawCallResult:
        if self._session is None:
            raise McpUnavailable("not connected")
        try:
            res = await self._session.call_tool(name, args or {})
        except McpError:
            raise
        except BaseException as exc: # noqa: BLE001 — never echo a server cmd/token/host
            raise McpUnavailable(f"mcp call failed: {type(exc).__name__}") from exc
        return _to_raw_result(res)

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        self._pinned = None
        if stack is not None:
            await self._safe_aclose(stack)

    @staticmethod
    async def _safe_aclose(stack: AsyncExitStack) -> None:
        try:
            await stack.aclose()
        except BaseException: # noqa: BLE001 — best-effort teardown (anyio cancel-scope quirks)
            pass
