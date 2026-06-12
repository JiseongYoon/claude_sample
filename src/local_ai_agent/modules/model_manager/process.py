"""Engine process controller — testable lifecycle for `llama-server`.

Owns the serving subprocess: start / stop / switch, with an explicit state
machine, a single-flight lock, crash detection, clean termination, and quant
allowlist validation. The launcher and readiness probe are injected so unit
tests never spawn a real 27 GB model.

Security: the model to load is validated to a bare `*.gguf` filename that exists
inside `MODEL_GGUF_DIR` (no path separators / `..`) — argv is assembled only from
validated settings, so a caller can never cause an arbitrary process exec.
"""
from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

from ...config import Settings

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    unloaded = "unloaded"
    loading = "loading"
    ready = "ready"
    error = "error"
    stopping = "stopping"


@runtime_checkable
class ProcessHandle(Protocol):
    @property
    def pid(self) -> int: ...
    def running(self) -> bool: ...
    async def terminate(self, timeout: float = 10.0) -> None: ...


@runtime_checkable
class ProcessLauncher(Protocol):
    async def spawn(self, argv: list[str], env: dict[str, str]) -> ProcessHandle: ...


ReadinessProbe = Callable[[str], Awaitable[bool]]


# -- default subprocess-backed launcher --------------------------------------
class _SubprocessHandle:
    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    def running(self) -> bool:
        return self._proc.returncode is None

    async def terminate(self, timeout: float = 10.0) -> None:
        if self._proc.returncode is not None:
            return
        self._proc.terminate() # SIGTERM
        try:
            await asyncio.wait_for(self._proc.wait(), timeout)
        except asyncio.TimeoutError:
            self._proc.kill() # SIGKILL
            await self._proc.wait()


class SubprocessLauncher:
    """Default launcher using asyncio subprocesses (stdout/stderr discarded)."""

    async def spawn(self, argv: list[str], env: dict[str, str]) -> ProcessHandle:
        proc = await asyncio.create_subprocess_exec(
            *argv, env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return _SubprocessHandle(proc)


async def _http_readiness(base_url: str) -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/health")
            return r.status_code == 200
    except Exception: # noqa: BLE001 — not ready / unreachable
        return False


class EngineProcessController:
    """Process control over `llama-server`, single-flighted and fault-isolated."""

    def __init__(self, settings: Settings, launcher: ProcessLauncher | None = None,
                 readiness: ReadinessProbe | None = None, *,
                 readiness_timeout: float = 300.0, poll_interval: float = 2.0) -> None:
        self._s = settings
        self._launcher = launcher or SubprocessLauncher()
        self._readiness = readiness or _http_readiness
        self._timeout = readiness_timeout
        self._poll = poll_interval
        self._lock = asyncio.Lock()
        self._handle: ProcessHandle | None = None
        self._state = EngineState.unloaded
        self._detail = ""
        self._loaded_file: str | None = None

    # -- introspection --------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self._s.serve_host}:{self._s.serve_port}"

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def loaded_file(self) -> str | None:
        return self._loaded_file

    @property
    def state(self) -> EngineState:
        # crash detection: a ready engine whose process died reads as error
        if self._state is EngineState.ready and self._handle is not None and not self._handle.running():
            self._state = EngineState.error
            self._detail = "engine process exited unexpectedly"
            self._loaded_file = None
        return self._state

    # -- validation / argv ----------------------------------------------------
    def _resolve_bare_gguf(self, name: str, field: str) -> Path:
        """Validate `name` to a bare existing `*.gguf` filename inside MODEL_GGUF_DIR, with a
        resolved-path containment check (symlink-escape guard). `field` labels the error."""
        if name != os.path.basename(name) or os.sep in name or ".." in name \
                or (os.altsep and os.altsep in name):
            raise ValueError(f"{field} must be a bare filename (no path components)")
        if not name.endswith(".gguf"):
            raise ValueError(f"{field} must end with .gguf")
        base = Path(self._s.model_gguf_dir)
        path = base / name
        if not path.is_file():
            raise ValueError(f"{field} not found in model dir: {name}")
        # defense-in-depth: the *resolved* path (after following any symlink) must
        # stay inside MODEL_GGUF_DIR — blocks a symlink escaping the model dir.
        try:
            if not path.resolve().is_relative_to(base.resolve()):
                raise ValueError(f"resolved {field} path escapes the model directory")
        except OSError as exc:
            raise ValueError(f"cannot resolve {field} path: {exc}") from None
        return path

    def resolve_model_path(self, gguf_file: str | None = None) -> Path:
        """Validate to a bare existing `*.gguf` filename inside MODEL_GGUF_DIR."""
        name = gguf_file or self._s.gguf_file
        if not name:
            raise ValueError("no GGUF file specified")
        return self._resolve_bare_gguf(name, "gguf_file")

    def resolve_mmproj_path(self) -> Path | None:
        """The optional vision projector. `None` when `model_mmproj_file` is unset (text-only);
        otherwise validated with the SAME containment discipline as the model GGUF."""
        name = self._s.model_mmproj_file
        if not name:
            return None
        return self._resolve_bare_gguf(name, "mmproj_file")

    def build_argv(self, model_path: Path) -> list[str]:
        s = self._s
        argv = [
            "llama-server",
            "-m", str(model_path),
            "--host", s.serve_host,
            "--port", str(s.serve_port),
            "-ngl", str(s.n_gpu_layers),
            "--split-mode", s.split_mode.value,
            "-ts", s.tensor_split,
            "-c", str(s.ctx_size),
            "--alias", s.served_model_name,
        ]
        mmproj = self.resolve_mmproj_path() # vision projector → multimodal launch
        if mmproj is not None:
            argv += ["--mmproj", str(mmproj)]
        if s.use_jinja:
            argv.append("--jinja")
        return argv

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self._s.cuda_visible_devices
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return env

    # -- lifecycle (public methods take the lock; *_locked assume it held) ----
    async def start(self, gguf_file: str | None = None) -> EngineState:
        async with self._lock:
            return await self._start_locked(gguf_file)

    async def stop(self) -> EngineState:
        async with self._lock:
            return await self._stop_locked()

    async def switch(self, gguf_file: str) -> EngineState:
        async with self._lock:
            await self._stop_locked()
            return await self._start_locked(gguf_file)

    async def _start_locked(self, gguf_file: str | None) -> EngineState:
        if self.state in (EngineState.ready, EngineState.loading):
            return self._state # already up / loading — no double spawn
        model_path = self.resolve_model_path(gguf_file) # validation BEFORE any spawn
        self._state = EngineState.loading
        self._detail = ""
        try:
            self._handle = await self._launcher.spawn(self.build_argv(model_path), self._build_env())
        except Exception as exc: # noqa: BLE001 — spawn failure is isolated
            self._state = EngineState.error
            self._detail = f"spawn failed: {exc!r}"
            self._handle = None
            return self._state

        waited = 0.0
        while waited < self._timeout:
            if self._handle is None or not self._handle.running():
                self._state = EngineState.error
                self._detail = "engine process exited during load"
                self._handle = None
                return self._state
            if await self._readiness(self.base_url):
                self._state = EngineState.ready
                self._loaded_file = model_path.name
                return self._state
            await asyncio.sleep(self._poll)
            waited += self._poll

        await self._terminate_handle()
        self._state = EngineState.error
        self._detail = "readiness timeout"
        return self._state

    async def _stop_locked(self) -> EngineState:
        self._state = EngineState.stopping
        await self._terminate_handle()
        self._state = EngineState.unloaded
        self._loaded_file = None
        return self._state

    async def _terminate_handle(self) -> None:
        if self._handle is not None:
            try:
                await self._handle.terminate(timeout=10.0)
            except Exception as exc: # noqa: BLE001
                logger.warning("error terminating engine process: %r", exc)
            self._handle = None
