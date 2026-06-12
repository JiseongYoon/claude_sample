"""Tests for the Module seam.

Verifies the interface contract + the HealthStatus states fault isolation needs.
Run inside conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from local_ai_agent.core.module import (
    BaseModule,
    Health,
    HealthStatus,
    Module,
    ModuleSpec,
)


def test_healthstatus_has_exact_states():
    assert {s.value for s in HealthStatus} == {"ok", "degraded", "down", "absent"}


def test_health_available_only_ok_or_degraded():
    assert Health(HealthStatus.ok).available is True
    assert Health(HealthStatus.degraded).available is True
    assert Health(HealthStatus.down).available is False
    assert Health(HealthStatus.absent).available is False


def test_modulespec_defaults_and_fields():
    spec = ModuleSpec(name="docqa", capabilities=("summarize",), depends_on=("model",))
    assert spec.name == "docqa"
    assert spec.version == "0.0.0"
    assert spec.capabilities == ("summarize",)
    assert spec.depends_on == ("model",)


async def test_basemodule_lifecycle_roundtrip():
    m = BaseModule(_spec=ModuleSpec(name="dummy", capabilities=("c",)))
    assert m.health().status is HealthStatus.absent # before start
    await m.start()
    assert m.health().status is HealthStatus.ok
    await m.stop()
    assert m.health().status is HealthStatus.absent


def test_basemodule_satisfies_module_protocol():
    m = BaseModule(_spec=ModuleSpec(name="dummy"))
    assert isinstance(m, Module) # runtime_checkable Protocol


def test_custom_module_satisfies_protocol():
    class Custom:
        @property
        def spec(self) -> ModuleSpec:
            return ModuleSpec(name="custom")

        async def start(self) -> None: ...

        async def stop(self) -> None: ...

        def health(self) -> Health:
            return Health(HealthStatus.ok)

    assert isinstance(Custom(), Module)
