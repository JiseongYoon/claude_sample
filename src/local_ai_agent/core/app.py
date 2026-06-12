"""Application container — the object the composition root builds and runs.

`Application` is a thin lifecycle wrapper over a `ModuleRegistry`: the
registry is the authority for lifecycle, health, and capability availability;
`Application` just owns one, exposes a stable surface to `main`, and is where the
API server will attach. It does not construct modules — that is the
composition root's job (`main.build_application`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .module import Health, Module
from .registry import ModuleRegistry


@dataclass
class Application:
    """A composed system: a registry of modules plus lifecycle control."""

    modules: list[Module] = field(default_factory=list)
    registry: ModuleRegistry = field(init=False, repr=False)
    # opaque attach point for the agent bundle. Typed loosely so core
    # takes no dependency on the orchestrator/safety modules — the composition root sets it.
    agent_runtime: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.registry = ModuleRegistry()
        for m in self.modules:
            self.registry.register(m)

    # lifecycle (delegated) ---------------------------------------------------
    async def startup(self) -> None:
        await self.registry.start_all()

    async def shutdown(self) -> None:
        await self.registry.stop_all()

    # queries (delegated) -----------------------------------------------------
    @property
    def start_errors(self) -> dict[str, str]:
        return self.registry.start_errors

    def get_module(self, name: str) -> Module | None:
        return self.registry.get_module(name)

    def health_snapshot(self) -> dict[str, Health]:
        return self.registry.health_snapshot()

    def is_capability_available(self, capability: str) -> bool:
        return self.registry.is_capability_available(capability)

    def available_capabilities(self) -> set[str]:
        return self.registry.available_capabilities()
