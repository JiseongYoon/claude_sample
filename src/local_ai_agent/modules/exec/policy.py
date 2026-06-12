"""Executor seam + the GuardedExecutor policy layer + ExecPolicy.

The security foundation of — the *execution* analogue of 's `contain_url`, 's
`GuardedConnector`, and 's `docs_root` sandbox. Two layers, so containment can never be
forgotten per executor:

  * **`Executor`** — the narrow *raw* interface a concrete executor implements (the broker-mediated
    `DockerSandbox` in ). It runs a command / reads / writes a file inside an *already-hardened*
    `SandboxSpec` and does **not** decide policy. It raises the typed `ExecError`s — never a raw library
    exception that could embed a host path / credential / daemon detail.
  * **`GuardedExecutor`** — wraps an `Executor` and, on every call, computes the hardened spec via
    `ExecPolicy`, **contains the command / workspace path FIRST**, enforces the output-byte cap +
    wall-clock timeout uniformly, and maps any non-typed exception to `ExecUnavailable`. Tools
    talk only to this; they never hold a raw executor, so there is no un-contained execution path.

The crux is **`ExecPolicy`** — a *pure, daemon-free* function that computes the FULL hardened container
spec (or refuses, fail-closed, before any side effect). The 2026-06-01 verification (F3) established
that the real containment must be hermetically testable; putting the hardening in a pure spec object —
not buried inside Docker calls — is exactly what makes that possible. `SandboxSpec` is total: every
field that bears on containment is set here, and there is deliberately **no field** for a host bind, a
docker.sock mount, `--privileged`, or `--cap-add` — those options are *unrepresentable*, so the agent
can never request them (F1: docker-daemon access = host root; the broker in only ever translates
a `SandboxSpec`).
"""
from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .config import ExecConfig


# --------------------------------------------------------------------------- #
# value types + typed errors (never leak a host path / cred / daemon detail)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mount:
    """A mount in the sandbox. ExecPolicy only ever produces a named *volume* mount (the workspace);
    there is NO field for a host bind path, so the agent can never request `-v /host:/x`."""

    volume: str # named docker volume (e.g. local_ai_agent_ws_<id>) — NOT a host path
    target: str # in-container mount point (e.g. /workspace)
    read_only: bool = False


@dataclass(frozen=True)
class SandboxSpec:
    """A fully-hardened, runnable container spec computed by `ExecPolicy`. Frozen + total: a concrete
    `Executor` only translates it to `docker run` flags — it cannot soften the policy. There
    is deliberately NO field for a host bind, docker.sock, `--privileged`, or `--cap-add`."""

    image: str
    workdir: str
    network: str # always "none" in v1 (no egress)
    read_only: bool # rootfs read-only — always True
    user: str # non-root, e.g. "1000:1000"
    cap_drop: tuple[str, ...] # always ("ALL",)
    no_new_privileges: bool # always True
    init: bool # always True (PID 1 reaps detached children)
    seccomp: str # "default" (do not use unconfined)
    mounts: tuple[Mount, ...] # the workspace volume (only writable path besides tmpfs)
    tmpfs: tuple[str, ...] # e.g. ("/tmp",) — mounted noexec,nosuid by the executor
    tmpfs_size_bytes: int
    env: tuple[tuple[str, str], ...] # scrubbed/whitelisted env as ordered (k, v) pairs
    mem_bytes: int
    memory_swap: int # == mem_bytes (no swap escape)
    cpus: float
    pids_limit: int
    ulimit_nofile: int
    ulimit_fsize: int


@dataclass(frozen=True)
class RawExecResult:
    """One raw execution result from an `Executor`. `truncated`/`timed_out` flags are surfaced rather
    than hidden so the caller (and 's contract) never mistakes a capped run for a clean one."""

    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    truncated: bool = False
    timed_out: bool = False
    duration: float = 0.0


class ExecError(Exception):
    """Base for all execution failures."""


class ExecBlocked(ExecError):
    """A request refused by policy/containment (bad command, workspace path escape, bad spec)."""


class ExecTimeout(ExecError):
    """A command exceeded its wall-clock budget."""


class ExecTooLarge(ExecError):
    """Output (or a read) exceeds the configured byte cap."""


class ExecUnavailable(ExecError):
    """The executor/daemon is unreachable or errored (message carries only the exception type)."""


class ExecResourceExceeded(ExecError):
    """The sandbox hit a resource cap (OOM / pids) — surfaced typed by the concrete executor."""


# --------------------------------------------------------------------------- #
# the raw seam (implemented by the broker-mediated DockerSandbox in )
# --------------------------------------------------------------------------- #
@runtime_checkable
class Executor(Protocol):
    """Raw execution inside an already-hardened `SandboxSpec`. `argv`/`path` are already contained by
    the policy layer. Concrete impls raise the typed `ExecError`s (never a raw lib exception that could
    embed a host path / credential)."""

    async def run(
        self, spec: SandboxSpec, argv: Sequence[str], *, timeout: float, max_output_bytes: int
    ) -> RawExecResult: ...
    async def read_file(self, spec: SandboxSpec, path: str, *, max_bytes: int) -> bytes: ...
    async def write_file(self, spec: SandboxSpec, path: str, data: bytes) -> None: ...
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# the security crux — pure, daemon-free spec computation + containment
# --------------------------------------------------------------------------- #
# Env names that must NEVER be forwarded into the sandbox even by accident. The whitelist is the
# primary control (only listed vars pass); this is a belt-and-braces deny that also refuses to forward
# a secret-looking var that an operator mistakenly whitelisted.
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "SESSION")
_SECRET_ENV_PREFIXES = ("AWS_", "SSH_", "GH_", "GITHUB_", "DOCKER_")


def _env_is_secret(name: str) -> bool:
    up = name.upper()
    return any(s in up for s in _SECRET_ENV_SUBSTRINGS) or any(up.startswith(p) for p in _SECRET_ENV_PREFIXES)


class ExecPolicy:
    """Computes the FULL hardened `SandboxSpec` and contains commands / workspace paths — purely, with
    no I/O and no daemon. Fail-closed: bad input raises `ExecBlocked` and **never** returns a partial or
    runnable spec. `environ` is injectable so env-scrub is deterministically testable."""

    def __init__(self, config: ExecConfig, *, environ: Mapping[str, str] | None = None) -> None:
        self._c = config
        self._environ = dict(environ) if environ is not None else dict(os.environ)

    # -- env scrub -------------------------------------------------------- #
    def _scrub_env(self) -> tuple[tuple[str, str], ...]:
        """Start from an EMPTY env; forward only the configured whitelist, and never a secret-looking
        var. Deterministic order = whitelist order."""
        out: list[tuple[str, str]] = []
        for name in self._c.env_whitelist:
            if name in self._environ and not _env_is_secret(name):
                out.append((name, self._environ[name]))
        return tuple(out)

    # -- spec ------------------------------------------------------------- #
    def build_spec(self, *, workspace_volume: str) -> SandboxSpec:
        """Compute the hardened spec for a task whose workspace is the named `workspace_volume`. Every
        containment-bearing field is set here and cannot be omitted (fail-closed on a bad volume name).
        v1 always sets network=none / read-only / non-root / cap-drop ALL / no-new-privileges / init."""
        if not isinstance(workspace_volume, str) or not workspace_volume.strip():
            raise ExecBlocked("workspace_volume must be a non-empty string")
        c = self._c
        return SandboxSpec(
            image=c.image,
            workdir=c.workspace_root,
            network=c.network, # validated == "none" in v1
            read_only=True,
            user=c.user,
            cap_drop=("ALL",),
            no_new_privileges=True,
            init=True,
            seccomp="default",
            mounts=(Mount(volume=workspace_volume.strip(), target=c.workspace_root, read_only=False),),
            tmpfs=("/tmp",),
            tmpfs_size_bytes=c.tmpfs_size_bytes,
            env=self._scrub_env(),
            mem_bytes=c.mem_bytes,
            memory_swap=c.mem_bytes, # no swap escape
            cpus=c.cpus,
            pids_limit=c.pids_limit,
            ulimit_nofile=c.ulimit_nofile,
            ulimit_fsize=c.ulimit_fsize,
        )

    # -- command containment --------------------------------------------- #
    def contain_command(self, command: Sequence[str]) -> tuple[str, ...]:
        """Validate a command into an argv vector. Must be a non-empty list/tuple of non-empty `str`
        with no NUL byte. A bare string is rejected — callers pass an explicit argv (a shell is an
        explicit `["sh", "-c", ...]`, never implicit). Fail-closed; the raw executor is never reached
        for a bad command."""
        if isinstance(command, (str, bytes)):
            raise ExecBlocked("command must be an argv list, not a string")
        if not isinstance(command, (list, tuple)) or len(command) == 0:
            raise ExecBlocked("command must be a non-empty argv list")
        argv: list[str] = []
        for arg in command:
            if not isinstance(arg, str) or arg == "":
                raise ExecBlocked("command args must be non-empty strings")
            if "\x00" in arg:
                raise ExecBlocked("command arg contains a NUL byte")
            argv.append(arg)
        return tuple(argv)

    # -- workspace path containment (the docs_root analogue) -------------- #
    def contain_workspace_path(self, rel: str) -> str:
        """Resolve `rel` under the (in-container) workspace root and refuse any escape. Lexical
        containment on a POSIX path: an absolute path or a `..` that climbs above the root → blocked.
        (Symlink containment is a *runtime* property enforced by the read-only rootfs + the workspace
        being a dedicated volume in 's `DockerSandbox`.)"""
        if not isinstance(rel, str) or not rel.strip():
            raise ExecBlocked("workspace path must be a non-empty string")
        if "\x00" in rel:
            raise ExecBlocked("workspace path contains a NUL byte")
        root = posixpath.normpath(self._c.workspace_root)
        # posixpath.join resets to `rel` if rel is absolute → normpath → caught by the prefix check
        norm = posixpath.normpath(posixpath.join(root, rel))
        if norm != root and not norm.startswith(root + "/"):
            raise ExecBlocked("workspace path escapes the workspace root")
        if norm == root:
            raise ExecBlocked("workspace path must name a file under the workspace root")
        return norm


# --------------------------------------------------------------------------- #
# the policy layer — the single contained path to an executor
# --------------------------------------------------------------------------- #
class GuardedExecutor:
    """Enforces spec hardening + command/path containment + the output cap & timeout around an
    `Executor`. Tools call only these methods; they never hold a raw executor, so no execution escapes
    containment."""

    def __init__(
        self,
        executor: Executor,
        *,
        policy: ExecPolicy,
        workspace_volume: str,
        timeout: float,
        max_output_bytes: int,
        name: str = "exec",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be > 0")
        if not isinstance(workspace_volume, str) or not workspace_volume.strip():
            raise ValueError("workspace_volume must be a non-empty string")
        self.name = name
        self._e = executor
        self._policy = policy
        self._ws = workspace_volume.strip()
        self._timeout = float(timeout)
        self._max_output = int(max_output_bytes)

    async def _guarded(self, fn, *args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ExecError:
            raise
        except Exception as exc: # noqa: BLE001 — never echo an internal path/cred from a lib exception
            raise ExecUnavailable(f"exec error: {type(exc).__name__}") from exc

    async def run_command(self, command: Sequence[str]) -> RawExecResult:
        """Contain the command, build the hardened spec, run it under the output + time caps."""
        argv = self._policy.contain_command(command) # fail-closed before any spec/executor
        spec = self._policy.build_spec(workspace_volume=self._ws)
        return await self._guarded(
            self._e.run, spec, argv, timeout=self._timeout, max_output_bytes=self._max_output
        )

    async def read_workspace_file(self, path: str) -> bytes:
        contained = self._policy.contain_workspace_path(path)
        spec = self._policy.build_spec(workspace_volume=self._ws)
        return await self._guarded(self._e.read_file, spec, contained, max_bytes=self._max_output)

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise ExecBlocked("workspace write data must be bytes")
        contained = self._policy.contain_workspace_path(path)
        spec = self._policy.build_spec(workspace_volume=self._ws)
        await self._guarded(self._e.write_file, spec, contained, bytes(data))

    async def close(self) -> None:
        await self._guarded(self._e.close)
