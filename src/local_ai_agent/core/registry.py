"""Module registry — the health-aware authority over modules.

This is the mechanism that makes the project's loose-coupling / fault-isolation
rules real:
- modules are referenced **by name** and resolved here, never imported by one
  another;
- lifecycle is driven in **dependency order** with a per-module isolation
  boundary, so one failure cannot cascade;
- **capability availability** is computed from health: a capability is offered
  only when its providing module *and* every module it `depends_on` are
  available. A down/absent dependency gates exactly the dependent capabilities
  and nothing else.

The registry is the single source of truth for "is X available right now?"; the
API gateway just surfaces it.

stdlib-only — depends only on the `Module` seam.
"""
from __future__ import annotations

import logging

from .module import Health, HealthStatus, Module

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Registers modules, drives their lifecycle, and answers availability."""

    def __init__(self) -> None:
        self._by_name: dict[str, Module] = {}
        self._order: list[str] = [] # registration order
        self._started: list[str] = [] # names successfully started (start order)
        self.start_errors: dict[str, str] = {}

    # -- registration ---------------------------------------------------------
    def register(self, module: Module) -> None:
        """Register a module by `spec.name`. Duplicate names are rejected."""
        name = module.spec.name
        if name in self._by_name:
            raise ValueError(f"duplicate module name: {name!r}")
        self._by_name[name] = module
        self._order.append(name)

    @property
    def modules(self) -> list[Module]:
        return [self._by_name[n] for n in self._order]

    def get_module(self, name: str) -> Module | None:
        return self._by_name.get(name)

    # -- lifecycle ------------------------------------------------------------
    def _start_order(self) -> list[str]:
        """Registration order, reordered so dependencies precede dependents.

        Unknown dependency names are skipped (the dependent still starts);
        cycles are broken safely (a node already being visited is not recursed
        into again)."""
        order: list[str] = []
        done: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in done or name not in self._by_name:
                return
            if name in visiting: # cycle — stop descending
                return
            visiting.add(name)
            for dep in self._by_name[name].spec.depends_on:
                visit(dep)
            visiting.discard(name)
            done.add(name)
            order.append(name)

        for n in self._order:
            visit(n)
        return order

    async def start_all(self) -> None:
        """Start every module in dependency order, fault-isolated. A raising
        `start()` is recorded in `start_errors`; the rest still start."""
        self.start_errors.clear()
        self._started.clear()
        for name in self._start_order():
            module = self._by_name[name]
            try:
                await module.start()
                self._started.append(name)
            except Exception as exc: # noqa: BLE001 — isolation boundary
                self.start_errors[name] = repr(exc)
                logger.warning("module %r failed to start: %r", name, exc)

    async def stop_all(self) -> None:
        """Stop the modules that started, in reverse start order. Resilient."""
        for name in reversed(self._started):
            try:
                await self._by_name[name].stop()
            except Exception as exc: # noqa: BLE001
                logger.warning("module %r failed to stop: %r", name, exc)
        self._started.clear()

    # -- health ---------------------------------------------------------------
    def health(self, name: str) -> Health:
        """Health of one module. `absent` if not registered; `down` if it
        failed to start or its `health()` raises."""
        module = self._by_name.get(name)
        if module is None:
            return Health(HealthStatus.absent, "not registered")
        if name in self.start_errors:
            return Health(HealthStatus.down, self.start_errors[name])
        try:
            return module.health()
        except Exception as exc: # noqa: BLE001 — health() must not crash callers
            return Health(HealthStatus.down, f"health() raised: {exc!r}")

    def health_snapshot(self) -> dict[str, Health]:
        return {name: self.health(name) for name in self._order}

    # -- availability / capability gating -------------------------------------
    def is_module_available(self, name: str, _visiting: set[str] | None = None) -> bool:
        """True iff the module is healthy (ok|degraded) AND every module it
        `depends_on` is (recursively) available. Cycle-safe."""
        _visiting = _visiting if _visiting is not None else set()
        if name in _visiting: # cycle: this link is already being established
            return True
        module = self._by_name.get(name)
        if module is None:
            return False
        if not self.health(name).available:
            return False
        _visiting.add(name)
        try:
            return all(self.is_module_available(dep, _visiting) for dep in module.spec.depends_on)
        finally:
            _visiting.discard(name)

    def _provider_of(self, capability: str) -> str | None:
        for name in self._order:
            if capability in self._by_name[name].spec.capabilities:
                return name
        return None

    def is_capability_available(self, capability: str) -> bool:
        """True iff some registered module provides `capability` and that module
        is available (incl. its dependencies). Unknown capability → False."""
        provider = self._provider_of(capability)
        return provider is not None and self.is_module_available(provider)

    def available_capabilities(self) -> set[str]:
        """All capability names currently offered (provider + deps available)."""
        caps: set[str] = set()
        for name in self._order:
            if self.is_module_available(name):
                caps.update(self._by_name[name].spec.capabilities)
        return caps
