"""Storage gated tools.

Six read/mutation `Tool` adapters the agent reaches through the dispatcher. Each selects a
connector BY NAME from the module's `connectors` registry (no host from a tool arg) and delegates
to its `GuardedConnector` (which contains + enforces read-only/size). Mutation tool names match
the gate's `_STORAGE_MUTATE_TOOLS` so they are auto-`needs_confirmation`; the read names are
safe-listed at wiring time. Every known failure → a graceful `{"ok": False, "error": ...}` (the
`StorageError` message is already credential-scrubbed by the transport).
"""
from __future__ import annotations

from typing import Any

from .connector import GuardedConnector, StorageError

# read-only tools → safe to allowlist on the gate ( wiring)
STORAGE_SAFE_TOOL_NAMES = frozenset({"storage_list", "storage_stat", "storage_read"})


def _err(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bad(msg: str) -> dict:
    return {"ok": False, "error": f"ValueError: {msg}"}


class _BaseStorageTool:
    def __init__(self, connectors: dict[str, GuardedConnector]) -> None:
        self._connectors = connectors

    def _resolve(self, args: dict):
        name = args.get("connector")
        if not isinstance(name, str) or not name:
            return _bad("'connector' (name) is required")
        conn = self._connectors.get(name)
        if conn is None:
            return _bad(f"unknown connector: {name!r}")
        return conn

    @staticmethod
    def _str_arg(args: dict, key: str, *, required: bool = True, default: str | None = None):
        v = args.get(key, default)
        if v is None and not required:
            return default
        if not isinstance(v, str) or (required and not v):
            return _bad(f"{key!r} must be a non-empty string")
        return v


_CONNECTOR_PROP = {"type": "string", "description": "Name of the configured storage connector."}


class StorageListTool(_BaseStorageTool):
    name = "storage_list"
    description = "List entries under a path on a remote storage connector."
    parameters = {
        "type": "object",
        "properties": {
            "connector": _CONNECTOR_PROP,
            "path": {"type": "string", "description": "Connector-relative path ('.' = root).", "default": "."},
        },
        "required": ["connector"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path", required=False, default=".")
        if isinstance(path, dict):
            return path
        try:
            entries = await conn.list(path)
        except StorageError as exc:
            return _err(exc)
        return {"ok": True, "entries": [
            {"path": e.path, "is_dir": e.is_dir, "size": e.size} for e in entries]}


class StorageStatTool(_BaseStorageTool):
    name = "storage_stat"
    description = "Stat a single path on a remote storage connector (size / is_dir)."
    parameters = {
        "type": "object",
        "properties": {"connector": _CONNECTOR_PROP,
                       "path": {"type": "string", "description": "Connector-relative path."}},
        "required": ["connector", "path"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        try:
            e = await conn.stat(path)
        except StorageError as exc:
            return _err(exc)
        return {"ok": True, "entry": {"path": e.path, "is_dir": e.is_dir, "size": e.size}}


class StorageReadTool(_BaseStorageTool):
    name = "storage_read"
    description = "Read a text file from a remote storage connector (size-bounded, contained)."
    parameters = {
        "type": "object",
        "properties": {"connector": _CONNECTOR_PROP,
                       "path": {"type": "string", "description": "Connector-relative file path."}},
        "required": ["connector", "path"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        try:
            data = await conn.read_bytes(path)
        except StorageError as exc:
            return _err(exc)
        return {"ok": True, "text": data.decode("utf-8", "replace"), "bytes": len(data)}


class StorageWriteTool(_BaseStorageTool):
    name = "storage_write"
    description = "Write a text file to a remote storage connector (mutating; requires approval)."
    parameters = {
        "type": "object",
        "properties": {"connector": _CONNECTOR_PROP,
                       "path": {"type": "string", "description": "Connector-relative file path."},
                       "content": {"type": "string", "description": "UTF-8 text content to write."}},
        "required": ["connector", "path", "content"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        content = args.get("content")
        if not isinstance(content, str):
            return _bad("'content' must be a string")
        try:
            await conn.write_bytes(path, content.encode("utf-8"))
        except StorageError as exc:
            return _err(exc)
        return {"ok": True}


class StorageDeleteTool(_BaseStorageTool):
    name = "storage_delete"
    description = "Delete a path on a remote storage connector (mutating; requires approval)."
    parameters = {
        "type": "object",
        "properties": {"connector": _CONNECTOR_PROP,
                       "path": {"type": "string", "description": "Connector-relative path to delete."}},
        "required": ["connector", "path"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        try:
            await conn.delete(path)
        except StorageError as exc:
            return _err(exc)
        return {"ok": True}


class StorageMoveTool(_BaseStorageTool):
    name = "storage_move"
    description = "Move/rename a path on a remote storage connector (mutating; requires approval)."
    parameters = {
        "type": "object",
        "properties": {"connector": _CONNECTOR_PROP,
                       "src": {"type": "string", "description": "Connector-relative source path."},
                       "dst": {"type": "string", "description": "Connector-relative destination path."}},
        "required": ["connector", "src", "dst"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        src = self._str_arg(args, "src")
        if isinstance(src, dict):
            return src
        dst = self._str_arg(args, "dst")
        if isinstance(dst, dict):
            return dst
        try:
            await conn.move(src, dst)
        except StorageError as exc:
            return _err(exc)
        return {"ok": True}


def build_tools(connectors: dict[str, GuardedConnector]) -> list:
    """The 6 storage tools, in registration order (reads then mutations)."""
    return [
        StorageListTool(connectors),
        StorageStatTool(connectors),
        StorageReadTool(connectors),
        StorageWriteTool(connectors),
        StorageDeleteTool(connectors),
        StorageMoveTool(connectors),
    ]
