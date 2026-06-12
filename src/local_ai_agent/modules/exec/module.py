"""ExecModule — the registry-facing sandboxed-execution capability.

Builds the single contained path to the daemon — `GuardedExecutor` ( policy) over a
`DockerSandbox` ( broker, via the `docker` CLI) — and exposes the gated `Tool`s (`.tools`) the
composition root registers on the dispatcher (`run_command` → confirm via `_SHELL_TOOLS`,
`read_workspace_file` safe-listed, `write_workspace_file` → confirm via `_MUTATING_FILE_TOOLS`).

`depends_on=()` — exec needs no in-registry module (it talks to the Docker daemon, not llm-serving).
**Fail-closed (INV-3):** `start()` probes the daemon; if Docker is unreachable the module reports
`down`, so the registry stops advertising the `exec` capability — and tool calls still degrade
gracefully (the sandbox returns a typed `ExecUnavailable`). The agent never holds a raw executor.

v1 uses ONE module-scoped session container/volume (`local_ai_agent_exec_<id>` /
`local_ai_agent_ws_<id>`; per-task scoping is a later refinement). **Time-bomb guard:**
`start()` removes the stale workspace volume so each session begins fresh — a prior session's planted
state can't be inherited; within a session the workspace persists but every `run_command` is gated.
The v1 contract is non-interactive · time+byte-capped · cancellable (container kill) one-shot commands.
The hardened base image (`exec_image`) MUST own `/workspace` as the sandbox UID (else a fresh volume is
root-owned and the non-root sandbox can't write it) — a deploy precondition surfaced by the smoke.
"""
from __future__ import annotations

from ...config import Settings
from ...core.module import Health, HealthStatus, ModuleSpec
from .config import ExecConfig
from .policy import ExecPolicy, GuardedExecutor
from .sandbox import (
    CONTAINER_PREFIX,
    VOLUME_PREFIX,
    DockerSandbox,
    ProcRunner,
    gc_orphans,
    probe,
    remove_volume,
)
from .tools import build_tools


class ExecModule:
    """`Module` exposing the `exec` capability (sandboxed command/code execution).

    `runner` is injectable (defaults to the real `docker` subprocess runner) so tests can drive the
    module with a fake — no daemon required.
    """

    def __init__(self, settings: Settings, *, session_id: str = "main", runner: ProcRunner | None = None) -> None:
        cfg = ExecConfig.from_settings(settings)
        self._config = cfg
        self._runner = runner
        self._name = f"{CONTAINER_PREFIX}{session_id}"
        self._volume = f"{VOLUME_PREFIX}{session_id}"
        self._sandbox = DockerSandbox(name=self._name, runner=runner)
        self._guarded = GuardedExecutor(
            self._sandbox,
            policy=ExecPolicy(cfg),
            workspace_volume=self._volume,
            timeout=cfg.timeout,
            max_output_bytes=cfg.max_output_bytes,
            name="exec",
        )
        self._tools = build_tools(self._guarded, config=cfg)
        self._started = False
        self._docker_ok = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(
            name="exec", version="0.1.0",
            capabilities=("exec",), depends_on=(),
            description="sandboxed command/code execution (gated tools)",
        )

    @property
    def tools(self) -> list:
        return list(self._tools)

    async def start(self) -> None:
        # best-effort cleanup of orphaned exec containers, then **remove the stale workspace volume**
        # (time-bomb guard: a fresh workspace per session so a prior session's planted state can't be
        # inherited), then a daemon reachability probe. Order matters: gc the container that may hold
        # the volume BEFORE removing the volume.
        try:
            await gc_orphans(self._runner)
            await remove_volume(self._volume, self._runner)
        except Exception: # noqa: BLE001 — best-effort cleanup must never block start
            pass
        self._docker_ok = await probe(self._runner)
        self._started = True

    async def stop(self) -> None:
        try:
            await self._sandbox.close()
        except Exception: # noqa: BLE001 — best-effort teardown
            pass
        self._started = False

    def health(self) -> Health:
        if not self._started:
            return Health(HealthStatus.absent, "not started")
        if not self._docker_ok:
            # fail-closed: Docker unreachable → capability not served (tool calls also degrade gracefully)
            return Health(HealthStatus.down, "docker daemon unreachable")
        return Health(HealthStatus.ok, "ready")
