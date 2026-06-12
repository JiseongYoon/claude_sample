"""Tests for the Application container + composition root.

Covers: empty-app boot, wired-module lifecycle, and the fault-isolation seed
(a module whose start() raises does not crash startup; others still start).
Run inside conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import build_application

# Explicit settings so the composition-root test is isolated from any real .env.
_SETTINGS = Settings(
    _env_file=None,
    model_safetensors_dir="./models/gemma-4-safetensors",
    model_gguf_dir="./models/gemma-4-gguf",
)


@dataclass
class _Spy:
    """A test module that records lifecycle calls and can be told to fail."""

    name: str
    fail_on_start: bool = False
    started: bool = field(default=False, init=False)
    stopped: bool = field(default=False, init=False)

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name=self.name, capabilities=(f"{self.name}-cap",))

    async def start(self) -> None:
        if self.fail_on_start:
            raise RuntimeError("boom")
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def health(self) -> Health:
        return Health(HealthStatus.ok if self.started else HealthStatus.absent)


async def test_empty_app_boots_cleanly():
    app = Application()
    await app.startup()
    assert app.health_snapshot() == {}
    assert app.start_errors == {}
    await app.shutdown() # no error on empty


async def test_wired_module_lifecycle_driven_by_app():
    spy = _Spy("docqa")
    app = Application(modules=[spy])
    await app.startup()
    assert spy.started is True
    assert app.get_module("docqa") is spy
    assert app.health_snapshot()["docqa"].status is HealthStatus.ok
    await app.shutdown()
    assert spy.stopped is True


async def test_failing_module_is_isolated():
    bad = _Spy("bad", fail_on_start=True)
    good = _Spy("good")
    app = Application(modules=[bad, good])
    await app.startup()
    # bad failed but did not crash startup; good still started
    assert good.started is True
    assert "bad" in app.start_errors
    snap = app.health_snapshot()
    assert snap["bad"].status is HealthStatus.down # failure surfaced
    assert snap["good"].status is HealthStatus.ok # unrelated module fine
    await app.shutdown()
    assert good.stopped is True # only started ones stopped
    assert bad.stopped is False


def test_build_application_is_the_wiring_point_and_empty_in_step1():
    app = build_application(_SETTINGS)
    assert isinstance(app, Application)
    assert app.modules == [] # no capability modules wired yet
