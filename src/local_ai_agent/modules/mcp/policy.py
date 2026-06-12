"""MCP client seam + the pure McpPolicy crux + GuardedMcpClient.

The security foundation of — the MCP analogue of 's `ExecPolicy`, 's
`contain_url`, 's `GuardedConnector`, 's `docs_root`. **The reframe:** an MCP server is
an UNTRUSTED external party. The canonical MCP client *trusts* the server (auto-exposes its tools,
feeds its descriptions to the model, trusts its permissioning); we must NOT. This layer encodes that
distrust, **fail-closed, before any connection**, and is **pure / SDK-free / network-free / subprocess-
free** so the real control is fully hermetic (tested with an injected `FakeMcpClient`).

Two layers (so policy can never be forgotten per client):

  * **`McpClient`** — the narrow *raw* transport seam a concrete client implements (the official-SDK
    `SdkMcpClient` in ). It raises the typed `McpError`s — never a raw SDK/transport exception
    that could embed a server command / URL / token / host.
  * **`GuardedMcpClient`** — wraps an `McpClient` and enforces policy on every call: **namespace** each
    tool (`mcp__<server>__<tool>`), **bound + label-untrusted** descriptions and results, enforce the
    per-call **timeout**, and map any non-typed exception to `McpUnavailable` (type name only — no
    leak). Tools hold only this; they never touch a raw client or the SDK, so no call escapes
    containment.

**Honest invariant (mirrors "gate ≠ containment"):** the gate (, every external tool →
`needs_confirmation`) is the load-bearing control; description/result bounding here is defense-in-depth,
not a guarantee. **Consume-only (D4):** `advertised_client_capabilities()` is empty — no
sampling/roots/elicitation, so a server has no channel to drive our model.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .config import McpServerConfig


# --------------------------------------------------------------------------- #
# value types + typed errors (never leak a server command / URL / token / host)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawToolSpec:
    """A tool as reported by a server (pre-policy). `description`/`input_schema` are ATTACKER-
    INFLUENCED — bounded + treated as untrusted by the policy before reaching the model."""

    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RawCallResult:
    """A raw tool-call result from a server (pre-policy). `content` is untrusted input."""

    content: str = ""
    is_error: bool = False


@dataclass(frozen=True)
class McpToolSpec:
    """A namespaced, description-bounded tool ready to register as a gated `Tool`."""

    server: str
    tool: str
    namespaced_name: str # mcp__<server>__<tool>
    description: str # bounded + untrusted
    input_schema: dict


@dataclass(frozen=True)
class McpCallResult:
    """A bounded, untrusted-labelled tool-call result returned by `GuardedMcpClient.call_tool`."""

    content: str
    is_error: bool
    truncated: bool


@dataclass(frozen=True)
class ClientCapabilities:
    """What the client advertises to a server at `initialize`. v1 = all False (consume-only)."""

    sampling: bool = False
    roots: bool = False
    elicitation: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.sampling or self.roots or self.elicitation)


class McpError(Exception):
    """Base for all MCP client failures."""


class McpBlocked(McpError):
    """Refused by policy/containment (bad server config, un-namespaceable name, bad args)."""


class McpTimeout(McpError):
    """A connect/call exceeded its time budget."""


class McpTooLarge(McpError):
    """A result exceeds the configured byte cap (when not silently truncated)."""


class McpUnavailable(McpError):
    """The server is unreachable or errored (message carries only the exception type)."""


# --------------------------------------------------------------------------- #
# the raw transport seam (implemented by concrete clients — SdkMcpClient in )
# --------------------------------------------------------------------------- #
@runtime_checkable
class McpClient(Protocol):
    """A 1:1 session with one MCP server. Concrete impls raise the typed `McpError`s only (never a raw
    SDK/transport exception that could embed a server command / URL / token / host)."""

    async def connect(self) -> None: ...
    async def list_tools(self) -> list[RawToolSpec]: ...
    async def call_tool(self, name: str, args: dict) -> RawCallResult: ...
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# the pure policy crux — namespacing + untrusted-output bounding + client caps
# --------------------------------------------------------------------------- #
_NS_PREFIX = "mcp"
_NS_SEP = "__"
# a name component is strict so it can never break/collide with the `__` namespace separator
_NAME_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class McpPolicy:
    """Pure, fail-closed policy. No I/O, no SDK, no network. Owns: server-config validation,
    tool namespacing, description/result bounding (untrusted-output), and the advertised client
    capabilities (empty — consume-only)."""

    def __init__(self, config) -> None:
        # accept McpConfig or any object carrying the two bounds (kept duck-typed for terse tests)
        self._max_desc = int(config.max_description_chars)
        self._max_result = int(config.max_result_bytes)
        if self._max_desc <= 0 or self._max_result <= 0:
            raise ValueError("max_description_chars / max_result_bytes must be > 0")

    # -- consume-only (D4) ---------------------------------------------------- #
    def advertised_client_capabilities(self) -> ClientCapabilities:
        """Empty — no sampling/roots/elicitation. A server cannot drive our model."""
        return ClientCapabilities()

    # -- server config (D5; defense-in-depth over the pydantic model) --------- #
    def validate_server_config(self, cfg: McpServerConfig) -> McpServerConfig:
        """Fail-closed re-check of an operator server config: stdio only, name namespaceable,
        command not option-shaped, nothing with a NUL. The pydantic model already enforces most of
        this at load; this is the policy's own gate so the crux is self-contained."""
        if not isinstance(cfg, McpServerConfig):
            raise McpBlocked("server config must be an McpServerConfig")
        if cfg.transport != "stdio":
            raise McpBlocked(f"unsupported transport (v1 = stdio): {cfg.transport!r}")
        self._require_name_component(cfg.name)
        if not isinstance(cfg.command, str) or not cfg.command.strip() or cfg.command.startswith("-"):
            raise McpBlocked("command must be a non-empty, non-option-shaped executable")
        for s in (cfg.command, *cfg.args, *cfg.env.keys(), *cfg.env.values()):
            if "\x00" in s:
                raise McpBlocked("server config must not contain a NUL byte")
        return cfg

    # -- namespacing (reframe f) ---------------------------------------------- #
    def _require_name_component(self, value: str) -> None:
        # NOTE: fullmatch (not match) — `$` matches before a trailing '\n', so `match` would admit a
        # name like "read\n"; fullmatch anchors the end so any trailing/embedded whitespace is rejected.
        if not isinstance(value, str) or not _NAME_COMPONENT_RE.fullmatch(value) or _NS_SEP in value:
            raise McpBlocked("name component is empty / malformed / contains the '__' separator")

    def namespace(self, server: str, tool: str) -> str:
        """`mcp__<server>__<tool>`. Rejects any component that is empty, malformed, or contains the
        `__` separator — so (server, tool) → name is unambiguous and an external tool can never shadow
        a native tool (the `mcp__` prefix guarantees it)."""
        self._require_name_component(server)
        self._require_name_component(tool)
        return f"{_NS_PREFIX}{_NS_SEP}{server}{_NS_SEP}{tool}"

    # -- untrusted-output bounding (reframe b / c) ---------------------------- #
    def bound_description(self, desc) -> str:
        """Cap length + strip control chars. The description is server-controlled (tool poisoning):
        it is kept as DATA, never executed. Defense-in-depth — the gate is the real control."""
        if not isinstance(desc, str):
            return ""
        cleaned = "".join(
            ch for ch in desc if ch in ("\n", "\t") or (0x20 <= ord(ch) < 0x7F) or ord(ch) > 0x9F
        )
        return cleaned[: self._max_desc]

    def bound_result(self, raw: RawCallResult) -> McpCallResult:
        """Cap content to `max_result_bytes` (utf-8) and label untrusted. Server output is untrusted
        input — bounded so a server cannot flood the loop context."""
        content = raw.content if isinstance(raw.content, str) else str(raw.content)
        encoded = content.encode("utf-8", "replace")
        truncated = len(encoded) > self._max_result
        if truncated:
            content = encoded[: self._max_result].decode("utf-8", "ignore")
        return McpCallResult(content=content, is_error=bool(raw.is_error), truncated=truncated)


# --------------------------------------------------------------------------- #
# the policy layer — the single contained path to an MCP client
# --------------------------------------------------------------------------- #
class GuardedMcpClient:
    """Enforces namespacing + untrusted-output bounding + per-call timeout around an `McpClient`.
    Tools call only `list_tools()` / `call_tool(namespaced_name, args)`; they never hold a raw client,
    so no call escapes the policy. A tool whose name cannot be namespaced is SKIPPED at list time
    (fail-closed — it is never exposed)."""

    def __init__(
        self,
        client: McpClient,
        *,
        policy: McpPolicy,
        server_name: str,
        call_timeout: float = 30.0,
    ) -> None:
        if call_timeout <= 0:
            raise ValueError("call_timeout must be > 0")
        # the server name must itself be namespaceable (fail-closed before any connect)
        policy._require_name_component(server_name)
        self._client = client
        self._policy = policy
        self._server = server_name
        self._timeout = float(call_timeout)
        self._name_map: dict[str, str] = {} # namespaced_name -> raw tool name (pinned at list time)

    @property
    def server_name(self) -> str:
        return self._server

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._name_map)

    async def _guarded(self, fn, *args):
        try:
            return await fn(*args)
        except McpError:
            raise
        except Exception as exc: # noqa: BLE001 — never echo a server cmd/URL/token/host from a lib exc
            raise McpUnavailable(f"mcp error: {type(exc).__name__}") from exc

    async def connect(self) -> None:
        await self._guarded(self._client.connect)

    async def list_tools(self) -> list[McpToolSpec]:
        """List the server's tools, namespaced + description-bounded. The mapping namespaced→raw is
        pinned here (D6 pin-at-connect). An un-namespaceable tool is skipped (never exposed)."""
        raw_tools = await self._guarded(self._client.list_tools)
        out: list[McpToolSpec] = []
        self._name_map = {}
        for r in raw_tools or []:
            try:
                ns = self._policy.namespace(self._server, getattr(r, "name", ""))
            except McpBlocked:
                continue # fail-closed: a tool we cannot safely namespace is not exposed
            self._name_map[ns] = r.name
            schema = r.input_schema if isinstance(getattr(r, "input_schema", None), dict) else {}
            out.append(McpToolSpec(
                server=self._server,
                tool=r.name,
                namespaced_name=ns,
                description=self._policy.bound_description(getattr(r, "description", "")),
                input_schema=schema,
            ))
        return out

    async def call_tool(self, namespaced_name: str, args: dict) -> McpCallResult:
        """Call a *previously listed* tool by its namespaced name. Unknown/unlisted name → `McpBlocked`
        (a server cannot smuggle in a tool that was never listed). Bounded by `call_timeout`; the
        result is size-bounded + untrusted-labelled. Errors carry only a type name (no args/content)."""
        raw_name = self._name_map.get(namespaced_name)
        if raw_name is None:
            raise McpBlocked("unknown or unregistered tool")
        if not isinstance(args, dict):
            raise McpBlocked("tool args must be a dict")
        try:
            raw = await asyncio.wait_for(
                self._guarded(self._client.call_tool, raw_name, args), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise McpTimeout(f"tool call exceeded {self._timeout}s") from exc
        if not isinstance(raw, RawCallResult):
            raise McpUnavailable("client returned a non-RawCallResult")
        return self._policy.bound_result(raw)

    async def close(self) -> None:
        await self._guarded(self._client.close)
