"""The Module seam — the single extensibility interface every capability implements.

This is the heart of the project's modularity rules (requirements.md):
- one module = one capability (single responsibility);
- modules expose lifecycle (`start`/`stop`) + `health()` and declare their
  `capabilities` and `depends_on` *by name* — they do NOT import one another;
- the composition root (`main`) and the registry wire them via this
  protocol, so a broken/absent module is isolated behind its health state.

Keep this module dependency-free (stdlib only) so every capability can import the
seam without dragging in platform internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class HealthStatus(str, Enum):
    """Lifecycle/health state of a module — the states fault isolation needs.

    - ok : started and serving its capabilities.
    - degraded : running but impaired (e.g. a soft dependency is down); some
                 capabilities may be limited but the module has not failed.
    - down : registered but not serving (failed to start, or crashed).
    - absent : not registered / disabled — its capabilities are simply
                 unavailable. NOT an error; distinguished from `down` so the
                 gateway can report "unavailable" without implying a fault.
    """

    ok = "ok"
    degraded = "degraded"
    down = "down"
    absent = "absent"


@dataclass(frozen=True, slots=True)
class Health:
    """A point-in-time health reading: a status plus an optional human detail."""

    status: HealthStatus
    detail: str = ""

    @property
    def available(self) -> bool:
        """Whether this module's capabilities should be offered (ok or degraded)."""
        return self.status in (HealthStatus.ok, HealthStatus.degraded)


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Static identity/metadata a module advertises to the composition root.

    `capabilities` are the capability names this module provides; `depends_on`
    are the *names* of other modules it needs (resolved via the registry by
    health, never via direct import — that is the loose-coupling guarantee).
    """

    name: str
    version: str = "0.0.0"
    capabilities: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    description: str = ""


@runtime_checkable
class Module(Protocol):
    """The interface every capability module implements.

    Lifecycle is async (modules may open sockets, spawn processes, etc.).
    Implementations must be safe to `stop()` even if `start()` failed, and
    `health()` must never raise (return a `down`/`degraded` Health instead).
    """

    @property
    def spec(self) -> ModuleSpec: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def health(self) -> Health: ...


@dataclass
class BaseModule:
    """Optional convenience base: tracks a status field and gives sane defaults.

    Capability modules may subclass this or implement `Module` directly. It is
    deliberately tiny — the seam is the `Module` protocol, not this class.
    """

    _spec: ModuleSpec = field(default_factory=lambda: ModuleSpec(name="unnamed"))
    _status: HealthStatus = HealthStatus.absent
    _detail: str = ""

    @property
    def spec(self) -> ModuleSpec:
        return self._spec

    async def start(self) -> None:
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status, self._detail)
