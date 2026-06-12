"""PresentationModule — hostability demo (§A5).

**NOT a pre-built avatar seam.** This is the deliberate proof that a future presentation/Live2D module
needs no special machinery: it slots in via the **generic `Module` interface** (lifecycle + health +
declared capability) and is surfaced to clients through the **existing REST/WS API** (`/capabilities` +
the WS `capabilities` event — capability visibility by module health). Wiring it required only the
standard `enable_presentation_demo` one-flag block in the composition root — no new gateway route, no
registry change, no gate/dispatcher change.

A real Live2D module would replace this body (consume the agent's output, drive the avatar over the
WS / its own client) using the SAME interface. It is fault-isolated like every module: if its backend
is unavailable it reports `down`/`absent` and only the `presentation` capability is gated — unrelated
capabilities keep working.
"""
from __future__ import annotations

from ...core.module import Health, HealthStatus, ModuleSpec


class PresentationModule:
    """`Module` exposing the `presentation` capability (avatar/presentation hostability demo).

    `fail_start` is a test hook to exercise fault isolation (a real backend that won't come up)."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self._fail_start = fail_start
        self._started = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="presentation", version="0.1.0",
            capabilities=("presentation",), depends_on=(),
            description="presentation/avatar capability (hostability demo — §A5, not a pre-built seam)",
        )

    async def start(self) -> None:
        if self._fail_start:
            raise RuntimeError("presentation backend unavailable") # → isolated by the registry
        self._started = True

    async def stop(self) -> None:
        self._started = False

    def health(self) -> Health:
        return Health(HealthStatus.ok if self._started else HealthStatus.absent, "presentation demo")
