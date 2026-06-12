"""Tests for ModuleRegistry.

Covers registration, dependency-ordered + fault-isolated lifecycle, health
tracking, and capability gating by dependency health (incl. missing-dep and
cycle safety). Run inside conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.core.registry import ModuleRegistry


@dataclass
class M:
    """Configurable test module: declares caps/deps, can fail on start, records
    start order into a shared list, and allows manual status changes."""

    name: str
    caps: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    fail_on_start: bool = False
    recorder: list[str] | None = None
    _status: HealthStatus = field(default=HealthStatus.absent, init=False)

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name=self.name, capabilities=self.caps, depends_on=self.deps)

    async def start(self) -> None:
        if self.fail_on_start:
            raise RuntimeError("boom")
        if self.recorder is not None:
            self.recorder.append(self.name)
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status)

    def set_status(self, s: HealthStatus) -> None:
        self._status = s


def test_register_rejects_duplicate():
    reg = ModuleRegistry()
    reg.register(M("a"))
    with pytest.raises(ValueError):
        reg.register(M("a"))


async def test_dependency_ordered_start():
    rec: list[str] = []
    reg = ModuleRegistry()
    # register dependent BEFORE its dependency to prove reordering happens
    reg.register(M("b", deps=("a",), recorder=rec))
    reg.register(M("a", recorder=rec))
    await reg.start_all()
    assert rec.index("a") < rec.index("b") # dependency started first


async def test_start_fault_isolated_and_health():
    a, bad, c = M("a"), M("bad", fail_on_start=True), M("c")
    reg = ModuleRegistry()
    for m in (a, bad, c):
        reg.register(m)
    await reg.start_all()
    assert "bad" in reg.start_errors
    snap = reg.health_snapshot()
    assert snap["a"].status is HealthStatus.ok
    assert snap["bad"].status is HealthStatus.down # failure surfaced
    assert snap["c"].status is HealthStatus.ok # unrelated module fine


async def test_health_absent_before_start_and_after_stop():
    reg = ModuleRegistry()
    reg.register(M("a"))
    assert reg.health("a").status is HealthStatus.absent
    await reg.start_all()
    assert reg.health("a").status is HealthStatus.ok
    await reg.stop_all()
    assert reg.health("a").status is HealthStatus.absent


def test_health_absent_for_unregistered():
    reg = ModuleRegistry()
    assert reg.health("ghost").status is HealthStatus.absent


async def test_capability_gating_by_dependency_health():
    model = M("model", caps=("load",))
    tuner = M("tuner", caps=("tune",), deps=("model",))
    reg = ModuleRegistry()
    reg.register(model)
    reg.register(tuner)
    await reg.start_all()
    # both healthy → both capabilities offered
    assert reg.is_capability_available("tune") is True
    assert reg.available_capabilities() == {"load", "tune"}
    # model goes down → tuner's capability is gated, model's too; unrelated nothing
    model.set_status(HealthStatus.down)
    assert reg.is_module_available("tuner") is False
    assert reg.is_capability_available("tune") is False
    assert reg.is_capability_available("load") is False
    assert reg.available_capabilities() == set()


async def test_degraded_dependency_still_available():
    model = M("model", caps=("load",))
    tuner = M("tuner", caps=("tune",), deps=("model",))
    reg = ModuleRegistry()
    reg.register(model)
    reg.register(tuner)
    await reg.start_all()
    model.set_status(HealthStatus.degraded) # degraded counts as available
    assert reg.is_capability_available("tune") is True


async def test_missing_dependency_gates_capability():
    x = M("x", caps=("xcap",), deps=("ghost",))
    reg = ModuleRegistry()
    reg.register(x)
    await reg.start_all()
    assert reg.is_module_available("x") is False # dep not registered
    assert reg.is_capability_available("xcap") is False


def test_unknown_capability_is_false():
    reg = ModuleRegistry()
    assert reg.is_capability_available("nope") is False


async def test_dependency_cycle_is_safe():
    a = M("a", caps=("ac",), deps=("b",))
    b = M("b", caps=("bc",), deps=("a",))
    reg = ModuleRegistry()
    reg.register(a)
    reg.register(b)
    await reg.start_all() # must not hang
    # both healthy and the cycle resolves without hanging or error
    assert reg.is_module_available("a") is True
    assert reg.is_module_available("b") is True
    assert reg.available_capabilities() == {"ac", "bc"}
