"""MCP client config + operator server declaration.

MCP servers are declared in a JSON file (`MCP_SERVERS_FILE`) — a list of typed `McpServerConfig`.
**The model can never add, configure, or spawn a server** (D5): the connectable set is exactly what
the operator declares here, by name. This mirrors 's `storage_connectors_file` (structural
prevention — the agent picks a server by name, never by an arbitrary command/URL from a tool arg).

v1 transport is **stdio** (D1): a server is an operator-trusted local executable spawned as a
subprocess (`command` + `args`), JSON-RPC over stdin/stdout. HTTP/SSE (a server URL + auth) is
sub-phase 8.1. **Auth secrets are never inline**: `env` carries operator-declared per-server
environment (e.g. an API-key var name/value the operator controls); the agent's own process secrets
are never forwarded ( plumbs this). `McpConfig` is the small frozen view consumed by the policy
and module (mirrors `ExecConfig`/`BrowserConfig`).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, field_validator

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# A literal `env` is for NON-secret vars only. These key shapes look secret-bearing and MUST go through
# `secret_env` (by reference: a host-env NAME resolved at connect, never an inline value). Mirrors the
# exec module's `_env_is_secret` deny-list (kept local — modules do not import each other; S2).
_SECRET_KEY_SUBSTRINGS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "SESSION")
_SECRET_KEY_PREFIXES = ("AWS_", "SSH_", "GH_", "GITHUB_", "DOCKER_")


def _key_looks_secret(key: str) -> bool:
    up = key.upper()
    return any(s in up for s in _SECRET_KEY_SUBSTRINGS) or any(up.startswith(p) for p in _SECRET_KEY_PREFIXES)

if TYPE_CHECKING: # avoid an import cycle / coupling at runtime
    from ...config import Settings


def _no_nul(value: str, *, what: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{what} must not contain a NUL byte")
    return value


class McpServerConfig(BaseModel):
    """One operator-declared MCP server. v1 = stdio only. `command` is the executable (NOT
    option-shaped); `args` are its arguments; `env` is operator-declared per-server environment
    (auth lives here by reference, never the agent's ambient secrets — )."""

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio"] = "stdio"
    command: str # the server executable, e.g. "npx" or "/usr/bin/mcp-server-x"
    args: list[str] = [] # arguments, e.g. ["-y", "@modelcontextprotocol/server-everything"]
    env: dict[str, str] = {} # NON-secret literal env for the server (e.g. LANG, NODE_ENV)
    secret_env: list[str] = [] # NAMES of host env vars (auth/secrets) to forward — never inline
                                       # (mirrors storage `password_env`; resolved at connect)

    @field_validator("name")
    @classmethod
    def _name_shape(cls, v: str) -> str:
        # the server name becomes part of the tool namespace `mcp__<name>__<tool>`; keep it strict so a
        # name can never break or collide with the namespace separator. (The policy re-checks this.)
        if not isinstance(v, str) or not v.strip():
            raise ValueError("server name must be a non-empty string")
        _no_nul(v, what="server name")
        if "__" in v or "/" in v or any(c.isspace() for c in v):
            raise ValueError("server name must not contain '__', '/', or whitespace")
        return v

    @field_validator("command")
    @classmethod
    def _command_shape(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("command must be a non-empty string (the server executable)")
        _no_nul(v, what="command")
        if v.startswith("-"):
            raise ValueError("command must not be option-shaped (must be an executable, not a flag)")
        return v

    @field_validator("args")
    @classmethod
    def _args_shape(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or any(not isinstance(a, str) for a in v):
            raise ValueError("args must be a list of strings")
        for a in v:
            _no_nul(a, what="arg")
        return v

    @field_validator("env")
    @classmethod
    def _env_shape(cls, v: dict[str, str]) -> dict[str, str]:
        if not isinstance(v, dict) or any(
            not isinstance(k, str) or not isinstance(val, str) for k, val in v.items()
        ):
            raise ValueError("env must be a mapping of string→string")
        for k, val in v.items():
            _no_nul(k, what="env key")
            _no_nul(val, what="env value")
            # belt-and-braces: a secret-looking key in the LITERAL env is a config error — secrets must be
            # declared by reference in `secret_env`, never as an inline value. Fail-closed (no value echoed).
            if _key_looks_secret(k):
                raise ValueError(
                    f"env key {k!r} looks secret-bearing; declare it in 'secret_env' "
                    "(a host env-var NAME resolved at connect), not as a literal 'env' value"
                )
        return v

    @field_validator("secret_env")
    @classmethod
    def _secret_env_shape(cls, v: list[str]) -> list[str]:
        # NAMES only (never values). Strict env-var-name shape so a value can't be smuggled in here.
        if not isinstance(v, list) or any(not isinstance(n, str) for n in v):
            raise ValueError("secret_env must be a list of env-var NAMES (strings)")
        for n in v:
            if not _ENV_NAME_RE.match(n):
                raise ValueError(f"secret_env entry is not a valid env var name: {n!r}")
        return v


def load_server_configs(path: str | Path) -> list[McpServerConfig]:
    """Parse + validate the servers JSON file: `{"servers": [ {...}, ... ]}`. Raises `ValueError`
    (clear, no secret content) on a missing file, malformed JSON, schema violation, or duplicate
    server name. Never echoes an `env` value."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read mcp servers file {str(p)!r}: {type(exc).__name__}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"mcp servers file is not valid JSON: {exc.msg}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("servers"), list):
        raise ValueError("mcp servers file must be an object with a 'servers' list")
    out: list[McpServerConfig] = []
    seen: set[str] = set()
    for i, item in enumerate(doc["servers"]):
        try:
            cfg = McpServerConfig.model_validate(item)
        except Exception as exc: # noqa: BLE001 — pydantic ValidationError → clean ValueError (no secret)
            raise ValueError(f"mcp server #{i} is invalid: {type(exc).__name__}") from exc
        if cfg.name in seen:
            raise ValueError(f"duplicate mcp server name: {cfg.name!r}")
        seen.add(cfg.name)
        out.append(cfg)
    return out


def resolve_server_env(cfg: McpServerConfig, *, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Build the env passed to a server subprocess: the non-secret literal `env` plus each `secret_env`
    NAME resolved from the host environment at connect time. **Fail-closed**: a declared secret var that
    is unset raises `ValueError` (clear, NO secret value). The returned dict's secret VALUES are never
    logged — only the operator-declared names live in the config. The agent's ambient (non-declared)
    secrets are NOT forwarded (the SDK merges this over a curated default env, not the full os.environ)."""
    env = dict(os.environ if environ is None else environ)
    out: dict[str, str] = dict(cfg.env) # non-secret literal env first
    for name in cfg.secret_env:
        if name not in env or env[name] == "":
            raise ValueError(f"secret env var {name!r} is declared but not set")
        out[name] = env[name]
    return out


@dataclass(frozen=True)
class McpConfig:
    """Resolved MCP settings. Numeric bounds are > 0 (validated on `Settings`)."""

    servers_file: Path | None
    call_timeout: float # per tool-call wall-clock budget (seconds)
    connect_timeout: float # per-server connect/handshake budget (seconds)
    max_result_bytes: int # tool-result content byte cap (untrusted-output bound)
    max_description_chars: int # tool-description char cap (tool-poisoning bound)

    @classmethod
    def from_settings(cls, settings: "Settings") -> "McpConfig":
        return cls(
            servers_file=settings.mcp_servers_file,
            call_timeout=settings.mcp_call_timeout,
            connect_timeout=settings.mcp_connect_timeout,
            max_result_bytes=settings.mcp_max_result_bytes,
            max_description_chars=settings.mcp_max_description_chars,
        )
