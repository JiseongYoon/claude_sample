"""Tests for EngineProcessController.

Hermetic: a fake launcher + fake readiness probe (no real llama-server / model),
and tmp GGUF files for path validation. Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.model_manager.process import (
    EngineProcessController,
    EngineState,
)

GGUF = "gemma-4-31B-it-UD-Q6_K_XL.gguf"
GGUF_Q4 = "gemma-4-31B-it-UD-Q4_K_XL.gguf"


# --- fakes ------------------------------------------------------------------
class FakeHandle:
    def __init__(self) -> None:
        self._running = True
        self.terminated = False

    @property
    def pid(self) -> int:
        return 4242

    def running(self) -> bool:
        return self._running

    async def terminate(self, timeout: float = 10.0) -> None:
        self._running = False
        self.terminated = True

    def crash(self) -> None:
        self._running = False


class FakeLauncher:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.spawns = 0
        self.last_argv: list[str] | None = None
        self.last_handle: FakeHandle | None = None

    async def spawn(self, argv, env): # noqa: ANN001
        self.spawns += 1
        self.last_argv = argv
        if self.fail:
            raise RuntimeError("spawn boom")
        self.last_handle = FakeHandle()
        return self.last_handle


def _ready_after(n: int):
    calls = {"n": 0}

    async def probe(base_url: str) -> bool: # noqa: ARG001
        calls["n"] += 1
        return calls["n"] > n
    return probe


async def _never_ready(base_url: str) -> bool: # noqa: ARG001
    return False


def _settings(tmp: Path, **over) -> Settings:
    base = dict(
        _env_file=None,
        model_safetensors_dir=str(tmp),
        model_gguf_dir=str(tmp),
        gguf_file=GGUF,
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def gguf_dir(tmp_path: Path) -> Path:
    for name in (GGUF, GGUF_Q4):
        (tmp_path / name).write_bytes(b"\x00") # fake model file
    return tmp_path


def _controller(tmp: Path, **kw):
    launcher = kw.pop("launcher", FakeLauncher())
    readiness = kw.pop("readiness", _ready_after(0)) # ready immediately
    return EngineProcessController(_settings(tmp), launcher=launcher, readiness=readiness,
                                   readiness_timeout=kw.pop("timeout", 5.0),
                                   poll_interval=kw.pop("poll", 0.01))


# --- validation -------------------------------------------------------------
def test_resolve_accepts_existing_basename(gguf_dir):
    ctrl = _controller(gguf_dir)
    assert ctrl.resolve_model_path(GGUF).name == GGUF


@pytest.mark.parametrize("bad", ["../evil.gguf", "sub/dir.gguf", "/abs/x.gguf",
                                 "..", "model.txt", "missing.gguf"])
def test_resolve_rejects_bad_inputs(gguf_dir, bad):
    ctrl = _controller(gguf_dir)
    with pytest.raises(ValueError):
        ctrl.resolve_model_path(bad)


def test_resolve_rejects_symlink_escape(gguf_dir, tmp_path):
    # a symlink inside the model dir pointing OUTSIDE must be rejected
    outside = tmp_path.parent / "outside-secret.gguf"
    outside.write_bytes(b"\x00")
    link = gguf_dir / "sneaky.gguf"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    ctrl = _controller(gguf_dir)
    with pytest.raises(ValueError):
        ctrl.resolve_model_path("sneaky.gguf")


def test_build_argv_has_adr0001_flags(gguf_dir):
    ctrl = _controller(gguf_dir)
    argv = ctrl.build_argv(ctrl.resolve_model_path(GGUF))
    assert argv[0] == "llama-server"
    for flag in ("-ngl", "--split-mode", "-ts", "-c", "--jinja", "--alias", "--host", "--port"):
        assert flag in argv
    assert "999" in argv and "layer" in argv and "1,1" in argv


# --- lifecycle --------------------------------------------------------------
async def test_start_reaches_ready(gguf_dir):
    ctrl = _controller(gguf_dir)
    assert await ctrl.start() is EngineState.ready
    assert ctrl.loaded_file == GGUF


async def test_stop_unloads_and_terminates(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher)
    await ctrl.start()
    assert await ctrl.stop() is EngineState.unloaded
    assert ctrl.loaded_file is None
    assert launcher.last_handle.terminated is True


async def test_switch_terminates_old_and_starts_new(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher)
    await ctrl.start(GGUF)
    first = launcher.last_handle
    assert await ctrl.switch(GGUF_Q4) is EngineState.ready
    assert first.terminated is True
    assert launcher.spawns == 2
    assert ctrl.loaded_file == GGUF_Q4


async def test_single_flight_no_double_spawn(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher)
    await asyncio.gather(ctrl.start(), ctrl.start(), ctrl.start())
    assert launcher.spawns == 1
    assert ctrl.state is EngineState.ready


async def test_crash_detection(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher)
    await ctrl.start()
    assert ctrl.state is EngineState.ready
    launcher.last_handle.crash() # process dies
    assert ctrl.state is EngineState.error # reconciled on read
    assert ctrl.loaded_file is None


# --- error class ------------------------------------------------------------
async def test_readiness_timeout_terminates_and_errors(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher, readiness=_never_ready,
                       timeout=0.05, poll=0.01)
    assert await ctrl.start() is EngineState.error
    assert launcher.last_handle.terminated is True # no leaked process
    assert "timeout" in ctrl.detail


async def test_spawn_failure_isolated(gguf_dir):
    ctrl = _controller(gguf_dir, launcher=FakeLauncher(fail=True))
    assert await ctrl.start() is EngineState.error
    assert "spawn failed" in ctrl.detail


async def test_invalid_quant_rejected_before_spawn(gguf_dir):
    launcher = FakeLauncher()
    ctrl = _controller(gguf_dir, launcher=launcher)
    with pytest.raises(ValueError):
        await ctrl.start("../escape.gguf")
    assert launcher.spawns == 0 # validation happened before spawn
