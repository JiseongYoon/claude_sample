"""Exec gated tools.

Three `Tool` adapters the agent reaches through the dispatcher — all backed by the single
`GuardedExecutor` (so every call is contained + capped):

  * **`run_command`** — run an argv command in the sandbox. The name is in the gate's `_SHELL_TOOLS`,
    so it is auto-`needs_confirmation` (HITL on every command) WITHOUT touching the security core.
  * **`read_workspace_file`** — read a workspace-contained file. Read-only + side-effect-free →
    **safe-listed** at wiring time (`EXEC_SAFE_TOOL_NAMES`); an absolute/escaping path still gets
    gated (nonlocal-path rule) and refused by containment.
  * **`write_workspace_file`** — write a workspace-contained file. A mutating op → gated
    `needs_confirmation` (the name is added to the gate's `_MUTATING_FILE_TOOLS`).

The agent never holds a raw `Executor`/`DockerSandbox` — only these tools, behind the gate. Every known
failure → a graceful `{"ok": False, "error": ...}`; the `ExecError` message is already type-only /
no-leak (the sandbox/policy layer scrubbed it). Output is byte-bounded by `GuardedExecutor`.
"""
from __future__ import annotations

from typing import Any

from .config import ExecConfig
from .policy import ExecError, GuardedExecutor

# read-only / side-effect-free → safe to allowlist on the gate ( wiring).
# `run_command` (→ `_SHELL_TOOLS`) and `write_workspace_file` (→ `_MUTATING_FILE_TOOLS`) stay gated.
EXEC_SAFE_TOOL_NAMES = frozenset({"read_workspace_file"})

_MAX_TEXT = 1_000_000 # defensive decode cap (GuardedExecutor already byte-bounds the raw output)


def _err(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bad(msg: str) -> dict:
    return {"ok": False, "error": f"ValueError: {msg}"}


def _text(data: bytes) -> str:
    return data[:_MAX_TEXT].decode("utf-8", "replace")


class RunCommandTool:
    """`run_command` — run an argv command in the sandbox (contained, capped). Gated via `_SHELL_TOOLS`."""

    name = "run_command"
    description = ("Run a command in the sandboxed container (argv list of strings; contained, "
                   "resource-capped). Mutating; requires approval.")
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "array", "items": {"type": "string"},
                        "description": "The command as an argv list, e.g. [\"ls\", \"-la\"]."},
        },
        "required": ["command"],
    }

    def __init__(self, guarded: GuardedExecutor) -> None:
        self._g = guarded

    async def run(self, args: dict) -> Any:
        command = args.get("command", args.get("argv"))
        if not isinstance(command, (list, tuple)):
            return _bad("'command' must be an argv list of strings")
        try:
            res = await self._g.run_command(list(command))
        except ExecError as exc:
            return _err(exc)
        return {
            "ok": True,
            "exit_code": res.exit_code,
            "stdout": _text(res.stdout),
            "stderr": _text(res.stderr),
            "truncated": res.truncated,
            "timed_out": res.timed_out,
            "duration": res.duration,
        }


class ReadWorkspaceFileTool:
    """`read_workspace_file` — read a workspace-contained file. Safe-listed (read-only, contained)."""

    name = "read_workspace_file"
    description = "Read a file from the sandbox workspace (read-only, contained)."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}},
        "required": ["path"],
    }

    def __init__(self, guarded: GuardedExecutor) -> None:
        self._g = guarded

    async def run(self, args: dict) -> Any:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            return _bad("'path' must be a non-empty string")
        try:
            data = await self._g.read_workspace_file(path.strip())
        except ExecError as exc:
            return _err(exc)
        return {"ok": True, "path": path.strip(), "content": _text(data)}


class WriteWorkspaceFileTool:
    """`write_workspace_file` — write a workspace-contained file. Mutating → gated (`_MUTATING_FILE_TOOLS`)."""

    name = "write_workspace_file"
    description = "Write a file into the sandbox workspace (mutating; requires approval)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path."},
            "content": {"type": "string", "description": "UTF-8 text content to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, guarded: GuardedExecutor) -> None:
        self._g = guarded

    async def run(self, args: dict) -> Any:
        path = args.get("path")
        content = args.get("content", args.get("data"))
        if not isinstance(path, str) or not path.strip():
            return _bad("'path' must be a non-empty string")
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, (bytes, bytearray)):
            return _bad("'content' must be a string or bytes")
        try:
            await self._g.write_workspace_file(path.strip(), bytes(content))
        except ExecError as exc:
            return _err(exc)
        return {"ok": True, "path": path.strip(), "bytes_written": len(content)}


def build_tools(guarded: GuardedExecutor, *, config: ExecConfig) -> list:
    """The exec tools, in registration order (run → read → write)."""
    return [
        RunCommandTool(guarded),
        ReadWorkspaceFileTool(guarded),
        WriteWorkspaceFileTool(guarded),
    ]
