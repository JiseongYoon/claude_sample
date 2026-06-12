"""Exec capability — sandboxed task/command/code execution.

v1 = **direct sandboxed-subprocess**: the agent runs commands / reads / writes files inside a hardened
Docker session-container, with mutating ops gated through the dispatcher AND wrapped in a real
OS sandbox (gate ≠ containment; the sandbox is the sole post-approval defense). OpenCode-as-engine was
rejected (own loop + own gate = INV-1-incompatible, zero sandbox) → speculative sub-phase 7.2; network
egress / package installs → sub-phase 7.1 (v1 is `--network none`).

 ships the pure, daemon-free security crux: the `Executor`/`GuardedExecutor` seam, `ExecPolicy`
(computes the full hardened `SandboxSpec`, fail-closed), workspace-path containment, and the config —
all testable with an injected fake `Executor`, no Docker required.
"""
from __future__ import annotations

from .config import ExecConfig
from .policy import (
    Executor,
    ExecBlocked,
    ExecError,
    ExecPolicy,
    ExecResourceExceeded,
    ExecTimeout,
    ExecTooLarge,
    ExecUnavailable,
    GuardedExecutor,
    Mount,
    RawExecResult,
    SandboxSpec,
)
from .sandbox import (
    CONTAINER_PREFIX,
    VOLUME_PREFIX,
    DockerSandbox,
    ProcResult,
    ProcRunner,
    build_exec_argv,
    build_read_argv,
    build_run_argv,
    build_write_argv,
    gc_orphans,
    probe,
    remove_volume,
)

__all__ = [
    "ExecConfig",
    "Executor",
    "ExecBlocked",
    "ExecError",
    "ExecPolicy",
    "ExecResourceExceeded",
    "ExecTimeout",
    "ExecTooLarge",
    "ExecUnavailable",
    "GuardedExecutor",
    "Mount",
    "RawExecResult",
    "SandboxSpec",
    # — broker + DockerSandbox
    "CONTAINER_PREFIX",
    "VOLUME_PREFIX",
    "DockerSandbox",
    "ProcResult",
    "ProcRunner",
    "build_exec_argv",
    "build_read_argv",
    "build_run_argv",
    "build_write_argv",
    "gc_orphans",
    "probe",
    "remove_volume",
]
