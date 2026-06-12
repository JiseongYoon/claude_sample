"""Tests for ModelManagerModule.

Uses a real EngineProcessController wired to a fake launcher + immediate
readiness (hermetic; no real model). Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import HealthStatus, Module
from local_ai_agent.core.registry import ModuleRegistry
from local_ai_agent.modules.model_manager.module import ModelManagerModule
from local_ai_agent.modules.model_manager.process import EngineProcessController, EngineState

GGUF = "gemma-4-31B-it-UD-Q6_K_XL.gguf"


class FakeHandle:
    def __init__(self) -> None:
        self._running = True
        self.terminated = False

    @property
    def pid(self) -> int:
        return 1

    def running(self) -> bool:
        return self._running

    async def terminate(self, timeout: float = 10.0) -> None:
        self._running = False
        self.terminated = True

    def crash(self) -> None:
        self._running = False


class FakeLauncher:
    def __init__(self) -> None:
        self.last_handle: FakeHandle | None = None

    async def spawn(self, argv, env): # noqa: ANN001
        self.last_handle = FakeHandle()
        return self.last_handle


async def _always_ready(base_url: str) -> bool: # noqa: ARG001
    return True


@pytest.fixture
def gguf_dir(tmp_path: Path) -> Path:
    (tmp_path / GGUF).write_bytes(b"\x00")
    return tmp_path


def _module(gguf_dir: Path):
    settings = Settings(_env_file=None, model_safetensors_dir=str(gguf_dir),
                        model_gguf_dir=str(gguf_dir), gguf_file=GGUF)
    launcher = FakeLauncher()
    ctrl = EngineProcessController(settings, launcher=launcher, readiness=_always_ready,
                                   readiness_timeout=5.0, poll_interval=0.01)
    return ModelManagerModule(settings, controller=ctrl), launcher


def test_satisfies_module_protocol_and_spec(gguf_dir):
    mod, _ = _module(gguf_dir)
    assert isinstance(mod, Module)
    assert mod.spec.name == "model-manager"
    assert mod.spec.capabilities == ("model-management",)


async def test_health_absent_before_start(gguf_dir):
    mod, _ = _module(gguf_dir)
    assert mod.health().status is HealthStatus.absent


async def test_started_with_no_model_is_ok_and_available(gguf_dir):
    mod, _ = _module(gguf_dir)
    await mod.start()
    # subsystem operational even though no model is loaded
    assert mod.health().status is HealthStatus.ok
    assert mod.engine_state is EngineState.unloaded
    assert mod.loaded_file is None
    assert mod.is_serving is False
    # in a registry, model-management capability stays available (load must work)
    reg = ModuleRegistry()
    reg.register(mod)
    await reg.start_all()
    assert reg.is_module_available("model-manager") is True
    assert "model-management" in reg.available_capabilities()


async def test_load_unload_switch(gguf_dir):
    mod, launcher = _module(gguf_dir)
    await mod.start()
    assert await mod.load() is EngineState.ready
    assert mod.is_serving is True
    assert mod.loaded_file == GGUF
    handle1 = launcher.last_handle
    assert await mod.switch(GGUF) is EngineState.ready
    assert handle1.terminated is True
    assert await mod.unload() is EngineState.unloaded
    assert mod.is_serving is False


async def test_engine_crash_surfaces_degraded_but_available(gguf_dir):
    mod, launcher = _module(gguf_dir)
    await mod.start()
    await mod.load()
    launcher.last_handle.crash() # engine dies
    h = mod.health()
    assert h.status is HealthStatus.degraded # subsystem up, engine errored
    assert h.available is True # reload still possible
    assert mod.engine_state is EngineState.error


async def test_stop_terminates_engine_and_goes_absent(gguf_dir):
    mod, launcher = _module(gguf_dir)
    await mod.start()
    await mod.load()
    await mod.stop()
    assert launcher.last_handle.terminated is True
    assert mod.health().status is HealthStatus.absent
