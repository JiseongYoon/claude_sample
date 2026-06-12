"""exec-broker + DockerSandbox — the daemon round-trip behind 's seam.

Per F1 (docker-socket access = host root), the **broker** is the *sole* constructor of a `docker`
command line: `build_run_argv` translates a `SandboxSpec` into a literal, hard-coded `docker run`
argv and **defensively re-validates** the spec, refusing to emit `-v`/`--privileged`/`--cap-add`/
host-mount/docker.sock/`--network` anything-but-none. The agent never names a flag, mount, or image —
they are unrepresentable in `SandboxSpec` and re-checked here.

v1 talks to the daemon via the **`docker` CLI as a subprocess** (user-chosen 2026-06-01): literal flags
map 1:1 to the hardening set, no extra dependency, and a production deploy can run the broker as a
separate user / sudo-allowlisted `docker` so the agent process never holds the socket. The process
runner is injected, so unit tests use a fake (no daemon); real Docker is exercised only by
`scripts/smoke_exec.py`. Container/volume names follow the user's `local_ai_agent_<name>` convention.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from .policy import (
    Executor,
    ExecBlocked,
    ExecError,
    ExecTimeout,
    ExecTooLarge,
    ExecUnavailable,
    RawExecResult,
    SandboxSpec,
)

# Naming (user directive 2026-06-01): exec resources use the UNDERSCORE form for easy identification.
CONTAINER_PREFIX = "local_ai_agent_exec_"
VOLUME_PREFIX = "local_ai_agent_ws_"

_ROOT_USERS = frozenset({"", "root", "0", "0:0", "root:root"})
# flags that must NEVER appear in an emitted argv (asserted by tests; we simply never build them)
FORBIDDEN_ARGV_TOKENS = ("--privileged", "--cap-add", "-v", "--volume", "--pid=host", "--pid", "--ipc=host")


# --------------------------------------------------------------------------- #
# process-runner seam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcResult:
    """Result of one subprocess invocation. `truncated` = output hit the byte cap; `timed_out` = the
    wall-clock budget was exceeded (the runner killed the process)."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    truncated: bool = False


# async (argv, *, stdin=None, timeout=None, max_output_bytes=None) -> ProcResult
ProcRunner = Callable[..., Awaitable[ProcResult]]


async def _subprocess_runner(
    argv: Sequence[str],
    *,
    stdin: bytes | None = None,
    timeout: float | None = None,
    max_output_bytes: int | None = None,
) -> ProcResult:
    """Real runner: spawn `argv`, feed `stdin` (or /dev/null), read stdout/stderr **bounded** so a flood
    can't OOM the host, all under a wall-clock `timeout` (kill on expiry). Never raises for a normal
    non-zero exit — that is returned as `returncode`."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    cap = max_output_bytes if max_output_bytes and max_output_bytes > 0 else None
    hard = (cap * 2) if cap else None # read a little past the cap so we can flag truncation

    async def _drain(stream) -> tuple[bytes, bool]:
        if stream is None:
            return b"", False
        buf = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if hard is not None and len(buf) >= hard:
                truncated = True
                break
        return bytes(buf), truncated

    async def _go() -> ProcResult:
        if stdin is not None:
            proc.stdin.write(stdin)
            proc.stdin.close()
        out, out_trunc = await _drain(proc.stdout)
        err, err_trunc = await _drain(proc.stderr)
        await proc.wait()
        return ProcResult(proc.returncode, out, err, timed_out=False, truncated=out_trunc or err_trunc)

    try:
        return await asyncio.wait_for(_go(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return ProcResult(returncode=-1, stdout=b"", stderr=b"", timed_out=True)


# --------------------------------------------------------------------------- #
# broker — the sole docker-argv constructor (pure; the security core)
# --------------------------------------------------------------------------- #
def _assert_hardened(spec: SandboxSpec) -> None:
    """Defense in depth on top of : refuse to translate a spec that is not fully hardened, so even
    a hand-constructed/buggy spec can never reach `docker run`."""
    if not isinstance(spec, SandboxSpec):
        raise ExecBlocked("spec must be a SandboxSpec")
    if spec.network != "none":
        raise ExecBlocked("spec.network must be 'none'")
    if not spec.read_only:
        raise ExecBlocked("spec.read_only must be True")
    if spec.cap_drop != ("ALL",):
        raise ExecBlocked("spec.cap_drop must be ('ALL',)")
    if not spec.no_new_privileges:
        raise ExecBlocked("spec.no_new_privileges must be True")
    if (spec.user or "").strip().lower() in _ROOT_USERS:
        raise ExecBlocked("spec.user must be non-root")
    if not str(spec.image).strip():
        raise ExecBlocked("spec.image must be non-empty")
    # option-shaped image/user/workdir would be parsed by docker as a flag (capability escalation) —
    # refuse, and `build_run_argv` also emits a `--` separator before the image as belt-and-braces.
    if str(spec.image).strip().startswith("-"):
        raise ExecBlocked("spec.image must not be option-shaped")
    if str(spec.workdir).startswith("-") or (spec.user or "").startswith("-"):
        raise ExecBlocked("spec.workdir / spec.user must not be option-shaped")
    for m in spec.mounts:
        if "docker.sock" in m.target or "docker.sock" in m.volume:
            raise ExecBlocked("docker.sock mount is forbidden")
        if m.target.rstrip("/") in ("", "/"):
            raise ExecBlocked("mount target must not be the container root")


def build_run_argv(spec: SandboxSpec, *, name: str) -> list[str]:
    """Translate a hardened `SandboxSpec` into a literal `docker run -d … sleep infinity` argv. Re-asserts
    hardening first; emits only the fixed hardening flag set — never a forbidden flag."""
    _assert_hardened(spec)
    if not isinstance(name, str) or not name.strip():
        raise ExecBlocked("container name must be a non-empty string")
    argv: list[str] = ["docker", "run", "-d", "--name", name]
    argv += ["--network", spec.network] # "none"
    if spec.read_only:
        argv += ["--read-only"]
    argv += ["--tmpfs", f"/tmp:rw,noexec,nosuid,size={spec.tmpfs_size_bytes}"]
    for m in spec.mounts:
        spec_str = f"type=volume,src={m.volume},dst={m.target}"
        if m.read_only:
            spec_str += ",readonly"
        argv += ["--mount", spec_str]
    argv += ["--workdir", spec.workdir]
    argv += ["--user", spec.user]
    argv += ["--cap-drop", "ALL"]
    if spec.no_new_privileges:
        argv += ["--security-opt", "no-new-privileges"]
    # seccomp "default" is applied by docker automatically — pass a profile only if a custom one is set.
    if spec.seccomp and spec.seccomp != "default":
        argv += ["--security-opt", f"seccomp={spec.seccomp}"]
    if spec.init:
        argv += ["--init"]
    argv += ["--memory", str(spec.mem_bytes), "--memory-swap", str(spec.memory_swap)]
    argv += ["--cpus", str(spec.cpus)]
    argv += ["--pids-limit", str(spec.pids_limit)]
    argv += ["--ulimit", f"nofile={spec.ulimit_nofile}", "--ulimit", f"fsize={spec.ulimit_fsize}"]
    for k, v in spec.env:
        argv += ["--env", f"{k}={v}"]
    # `--` terminates docker's option parsing → the image / command can never be read as a flag, even
    # if some field were option-shaped (defense in depth on top of `_assert_hardened`).
    argv += ["--", spec.image, "sleep", "infinity"]
    return argv


def build_exec_argv(name: str, argv: Sequence[str]) -> list[str]:
    """`docker exec <name> <argv>` — non-interactive (stdin /dev/null, no TTY; the runner feeds DEVNULL)."""
    return ["docker", "exec", name, *argv]


def build_read_argv(name: str, path: str) -> list[str]:
    return ["docker", "exec", name, "cat", "--", path]


def build_write_argv(name: str, path: str) -> list[str]:
    # $0 = path; data arrives on stdin. The workspace volume is the only writable mount.
    return ["docker", "exec", "-i", name, "sh", "-c", 'cat > "$0"', path]


# --------------------------------------------------------------------------- #
# the concrete Executor
# --------------------------------------------------------------------------- #
class DockerSandbox(Executor):
    """A long-lived hardened container per task; commands injected via `docker exec`. Created lazily on
    first use, torn down on `close`. The only docker access path is the injected `runner`."""

    def __init__(
        self,
        *,
        name: str,
        runner: ProcRunner | None = None,
        create_timeout: float = 30.0,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        self.name = name.strip()
        self._run_proc: ProcRunner = runner or _subprocess_runner
        self._create_timeout = float(create_timeout)
        self._clock = monotonic or time.monotonic
        self._created = False
        self._volume: str | None = None

    async def _ensure(self, spec: SandboxSpec) -> None:
        if self._created:
            return
        argv = build_run_argv(spec, name=self.name) # re-validates + builds (may raise ExecBlocked)
        self._volume = spec.mounts[0].volume if spec.mounts else None
        res = await self._invoke(argv, timeout=self._create_timeout)
        if res.timed_out:
            raise ExecUnavailable("container create timed out")
        if res.returncode != 0:
            raise ExecUnavailable(f"container create failed (rc={res.returncode})")
        self._created = True

    async def _invoke(self, argv, *, stdin=None, timeout=None, max_output_bytes=None) -> ProcResult:
        """Call the runner, mapping any non-`ExecError` exception to `ExecUnavailable` (type name only —
        never echo stderr/path/cred)."""
        try:
            return await self._run_proc(
                argv, stdin=stdin, timeout=timeout, max_output_bytes=max_output_bytes
            )
        except ExecError:
            raise
        except Exception as exc: # noqa: BLE001 — never leak a docker/daemon detail
            raise ExecUnavailable(f"docker invocation error: {type(exc).__name__}") from exc

    def _bound(self, data: bytes, cap: int) -> tuple[bytes, bool]:
        if cap and len(data) > cap:
            return data[:cap], True
        return data, False

    async def run(
        self, spec: SandboxSpec, argv: Sequence[str], *, timeout: float, max_output_bytes: int
    ) -> RawExecResult:
        await self._ensure(spec)
        start = self._clock()
        res = await self._invoke(
            build_exec_argv(self.name, argv), timeout=timeout, max_output_bytes=max_output_bytes
        )
        if res.timed_out:
            await self._kill()
            raise ExecTimeout(f"command exceeded {timeout}s")
        out, out_trunc = self._bound(res.stdout, max_output_bytes)
        err, err_trunc = self._bound(res.stderr, max_output_bytes)
        return RawExecResult(
            exit_code=res.returncode,
            stdout=out,
            stderr=err,
            truncated=res.truncated or out_trunc or err_trunc,
            timed_out=False,
            duration=max(0.0, self._clock() - start),
        )

    async def read_file(self, spec: SandboxSpec, path: str, *, max_bytes: int) -> bytes:
        await self._ensure(spec)
        res = await self._invoke(build_read_argv(self.name, path), timeout=self._create_timeout,
                                 max_output_bytes=max_bytes)
        if res.timed_out:
            await self._kill()
            raise ExecTimeout("read timed out")
        if res.returncode != 0:
            raise ExecError(f"read failed (rc={res.returncode})")
        if res.truncated or len(res.stdout) > max_bytes:
            raise ExecTooLarge(f"read exceeds {max_bytes} bytes")
        return res.stdout

    async def write_file(self, spec: SandboxSpec, path: str, data: bytes) -> None:
        await self._ensure(spec)
        res = await self._invoke(build_write_argv(self.name, path), stdin=bytes(data),
                                 timeout=self._create_timeout)
        if res.timed_out:
            await self._kill()
            raise ExecTimeout("write timed out")
        if res.returncode != 0:
            raise ExecError(f"write failed (rc={res.returncode})")

    async def _kill(self) -> None:
        await self._best_effort(["docker", "kill", self.name])

    async def _best_effort(self, argv) -> None:
        """Cleanup that must never raise (a teardown failure shouldn't crash the agent loop)."""
        try:
            await self._run_proc(argv, timeout=self._create_timeout)
        except Exception: # noqa: BLE001 — best-effort
            pass

    async def close(self) -> None:
        await self._best_effort(["docker", "rm", "-f", self.name])
        if self._volume:
            await self._best_effort(["docker", "volume", "rm", self._volume])
        self._created = False


async def gc_orphans(runner: ProcRunner | None = None, *, prefix: str = CONTAINER_PREFIX) -> list[str]:
    """Remove any leftover exec containers (best-effort startup cleanup). Returns the names removed."""
    run_proc = runner or _subprocess_runner
    try:
        res = await run_proc(["docker", "ps", "-aq", "--filter", f"name={prefix}"], timeout=15.0)
    except Exception: # noqa: BLE001
        return []
    if res.returncode != 0:
        return []
    ids = [ln for ln in res.stdout.decode("utf-8", "replace").split() if ln]
    for cid in ids:
        try:
            await run_proc(["docker", "rm", "-f", cid], timeout=15.0)
        except Exception: # noqa: BLE001
            pass
    return ids


async def remove_volume(name: str, runner: ProcRunner | None = None) -> None:
    """Best-effort `docker volume rm <name>` (the time-bomb guard: a fresh workspace per session so a
    prior session's planted state can't be inherited). Never raises."""
    run_proc = runner or _subprocess_runner
    try:
        await run_proc(["docker", "volume", "rm", name], timeout=15.0)
    except Exception: # noqa: BLE001 — best-effort (volume may not exist / be in use)
        pass


async def probe(runner: ProcRunner | None = None) -> bool:
    """Daemon reachability for module health. The deeper cap-enforcement self-test runs in the
    operator smoke against real Docker."""
    run_proc = runner or _subprocess_runner
    try:
        res = await run_proc(["docker", "info"], timeout=15.0)
    except Exception: # noqa: BLE001
        return False
    return res.returncode == 0
