"""Storage connector config + credential model.

Connectors are declared in a JSON file (`STORAGE_CONNECTORS_FILE`) — a list of typed
`StorageConnectorConfig`. A tool always selects a connector BY NAME, never by an arbitrary host
from a tool arg, so the set of connectable hosts is exactly what config declares (structural
SSRF prevention).

**Credentials are never inline.** SSH auth is either a `key_path` (a private-key file path) or a
`password_env` (the NAME of an environment variable holding the password). The secret is resolved
from the environment at connect time (`resolve_password`); it is never stored in the config object's
repr/logs. `read_only` defaults to True — writes require explicit opt-in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class SSHAuth(BaseModel):
    """Exactly one of `key_path` / `password_env`. Neither holds a secret value inline:
    `key_path` is a file path; `password_env` is the NAME of an env var to read at connect time."""

    model_config = ConfigDict(extra="forbid")

    key_path: Path | None = None
    password_env: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "SSHAuth":
        if (self.key_path is None) == (self.password_env is None):
            raise ValueError("SSH auth requires exactly one of key_path / password_env")
        return self


class StorageConnectorConfig(BaseModel):
    """One declared connector. No secret value is stored here (see `SSHAuth`)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["ssh"] = "ssh"
    host: str
    port: int = 22
    username: str
    auth: SSHAuth
    allowed_root: str # absolute POSIX remote path the connector may touch
    read_only: bool = True # secure default; writes are explicit opt-in
    max_bytes: int = 10_000_000 # per-read size cap
    # SSH host-key verification source (a known_hosts file). REQUIRED at connect time — the
    # connector refuses to connect without it (never disables verification → MITM protection).
    known_hosts_path: Path | None = None

    @field_validator("name", "host", "username")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("allowed_root")
    @classmethod
    def _abs_root(cls, v: str) -> str:
        if not isinstance(v, str) or not v.startswith("/"):
            raise ValueError("allowed_root must be an absolute POSIX path (starts with '/')")
        return v

    @field_validator("port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be in 1..65535")
        return v

    @field_validator("max_bytes")
    @classmethod
    def _bytes_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_bytes must be > 0")
        return v


def resolve_password(auth: SSHAuth) -> str | None:
    """Resolve the password from the named env var (or None if key-based). Raises if the env var
    is declared but unset — never returns a partial/empty credential silently."""
    if auth.password_env is None:
        return None
    value = os.environ.get(auth.password_env)
    if not value:
        raise ValueError(f"password env var {auth.password_env!r} is not set")
    return value


def load_connector_configs(path: str | Path) -> list[StorageConnectorConfig]:
    """Parse + validate the connectors JSON file: `{"connectors": [ {...}, ... ]}`. Raises
    `ValueError` (clear, no secret content) on a missing file, malformed JSON, schema violation,
    or duplicate connector name."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read connectors file {str(p)!r}: {type(exc).__name__}") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"connectors file is not valid JSON: {exc.msg}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("connectors"), list):
        raise ValueError("connectors file must be an object with a 'connectors' list")
    out: list[StorageConnectorConfig] = []
    seen: set[str] = set()
    for i, item in enumerate(doc["connectors"]):
        try:
            cfg = StorageConnectorConfig.model_validate(item)
        except Exception as exc: # noqa: BLE001 — pydantic ValidationError → clean ValueError
            raise ValueError(f"connector #{i} is invalid: {exc}") from exc
        if cfg.name in seen:
            raise ValueError(f"duplicate connector name: {cfg.name!r}")
        seen.add(cfg.name)
        out.append(cfg)
    return out
