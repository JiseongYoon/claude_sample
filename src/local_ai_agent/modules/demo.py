"""Demo capability modules — proof of fault isolation + capability gating.

These are intentionally trivial; their job is to demonstrate (and let tests prove)
the platform's core property end-to-end: a broken/absent module disables only its
own + dependent capabilities while unrelated modules keep serving. They are wired
only when `enable_demo_modules=True` so production stays clean.

Each implements the `Module` seam directly (no cross-module imports).
"""
from __future__ import annotations

from ..core.module import Health, HealthStatus, ModuleSpec


class EchoModule:
    """Unrelated, always-healthy module (capability `echo`)."""

    def __init__(self) -> None:
        self._status = HealthStatus.absent

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="echo", version="0.1.0", capabilities=("echo",),
                          description="trivial always-healthy demo capability")

    async def start(self) -> None:
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status)


class FlakyModule:
    """A module that can be broken at startup (`broken=True`) or toggled at
    runtime via `set_status` — the lever the fault-isolation demo pulls."""

    def __init__(self, broken: bool = False) -> None:
        self._broken = broken
        self._status = HealthStatus.absent

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="flaky", version="0.1.0", capabilities=("flaky",),
                          description="demo module that can fail / degrade")

    async def start(self) -> None:
        if self._broken:
            raise RuntimeError("flaky module deliberately failed to start")
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def set_status(self, status: HealthStatus) -> None:
        """Force a runtime health transition (for the live-gating demo)."""
        self._status = status

    def health(self) -> Health:
        return Health(self._status)


class DependentModule:
    """Capability `dependent` that soft-depends on `flaky` — gated by its health."""

    def __init__(self) -> None:
        self._status = HealthStatus.absent

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="dependent", version="0.1.0", capabilities=("dependent",),
                          depends_on=("flaky",),
                          description="demo capability gated on the flaky module")

    async def start(self) -> None:
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status)
