"""ModelManager — the engine-process controller wrapped as a platform Module.

Health semantics (deliberate): `health()` reports whether the **manager
subsystem** is operational, NOT whether a model is loaded. So `model-management`
(load/unload/switch) stays available even with no model loaded — you must be able
to issue a load command. Whether a model is actually *ready to serve* is separate
data (`engine_state` / `loaded_file`), which `llm-serving` and the param
API gate on.

Load-on-demand: `start()` makes the manager operational but does not load a model
(loading is ~minutes and must not block app startup); loading is triggered later
via `load()` (wired to the gateway in ).
"""
from __future__ import annotations

import asyncio
import logging

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .process import EngineProcessController, EngineState

logger = logging.getLogger(__name__)


class ModelBusy(RuntimeError):
    """Raised when a load/switch is requested while one is already in progress."""


class ModelManagerModule:
    """`Module` that owns the serving engine's process lifecycle."""

    def __init__(self, settings: Settings, controller: EngineProcessController | None = None) -> None:
        self._settings = settings
        self._controller = controller or EngineProcessController(settings)
        self._started = False
        self._tasks: set[asyncio.Task] = set()

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="model-manager", version="0.1.0",
            capabilities=("model-management",),
            description="dynamic load/unload/switch of the serving engine",
        )

    async def start(self) -> None:
        # load-on-demand: become operational without loading a model
        self._started = True

    async def stop(self) -> None:
        await self._controller.stop()
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if self._controller.state is EngineState.error:
            # subsystem is up (reload possible) but the engine errored — degraded,
            # still available so the operator can recover.
            return Health(HealthStatus.degraded, f"engine error: {self._controller.detail}")
        return Health(HealthStatus.ok, f"engine: {self._controller.state.value}")

    # -- load-state introspection (separate from module health) --------------
    @property
    def engine_state(self) -> EngineState:
        return self._controller.state

    @property
    def loaded_file(self) -> str | None:
        return self._controller.loaded_file

    @property
    def base_url(self) -> str:
        return self._controller.base_url

    @property
    def is_serving(self) -> bool:
        """True only when a model is loaded and the engine is ready to serve."""
        return self._controller.state is EngineState.ready

    # -- blocking lifecycle ops (awaitable; used by tests + the launch_* wrappers)
    async def load(self, gguf_file: str | None = None) -> EngineState:
        return await self._controller.start(gguf_file)

    async def unload(self) -> EngineState:
        return await self._controller.stop()

    async def switch(self, gguf_file: str) -> EngineState:
        return await self._controller.switch(gguf_file)

    # -- non-blocking launch (for the API; validates synchronously then defers) --
    def validate_target(self, gguf_file: str | None = None) -> None:
        """Raise ValueError if the requested GGUF is not an allowed model file."""
        self._controller.resolve_model_path(gguf_file)

    def launch_load(self, gguf_file: str | None = None) -> tuple[EngineState, bool]:
        """Validate, then start loading in the background (non-blocking). Returns
        (state, launched). No-op (launched=False) if already loading/ready."""
        self.validate_target(gguf_file)
        st = self.engine_state
        if st in (EngineState.loading, EngineState.ready):
            return st, False
        self._spawn(self.load(gguf_file))
        return EngineState.loading, True

    def launch_switch(self, gguf_file: str) -> tuple[EngineState, bool]:
        """Validate, then switch in the background. 409 (`ModelBusy`) if a load is
        already in progress."""
        self.validate_target(gguf_file)
        if self.engine_state is EngineState.loading:
            raise ModelBusy("a load/switch is already in progress")
        self._spawn(self.switch(gguf_file))
        return EngineState.loading, True

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(self._guarded(coro))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guarded(self, coro) -> None:
        try:
            await coro
        except Exception as exc: # noqa: BLE001 — background task must not crash the loop
            logger.warning("model lifecycle task failed: %r", exc)
