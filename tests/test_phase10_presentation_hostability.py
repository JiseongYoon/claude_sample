"""— presentation-module hostability (§A5).

Proves the §A5 claim: there is NO pre-built avatar seam — a future presentation/Live2D module slots in via
the GENERIC `Module` interface + the EXISTING REST/WS API, with ZERO core change beyond the standard
`enable_presentation_demo` one-flag wiring. The kept minimal `PresentationModule` registers, is
health-gated, is fault-isolated, and its `presentation` capability is surfaced automatically through the
existing `/capabilities` REST endpoint + the WS `capabilities` event — no dedicated route.

Run in conda `local-ai-agent-env-1`: `pytest tests/test_phase10_presentation_hostability.py`.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import build_application, create_app
from local_ai_agent.modules.presentation.module import PresentationModule

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _settings(**kw) -> Settings:
    return Settings(**_DIRS, **kw)


class _HealthyPeer:
    """An always-healthy unrelated module (capability `peer`) for the isolation test."""

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="peer", version="0.0.0", capabilities=("peer",), depends_on=())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def health(self) -> Health:
        return Health(HealthStatus.ok, "peer")


# --------------------------------------------------------------------------- #
# registers via the standard seam + flag; healthy; capability available
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_registers_and_healthy_via_standard_flag(tmp_path):
    app = build_application(_settings(enable_presentation_demo=True))
    await app.startup()
    try:
        assert app.get_module("presentation") is not None
        assert app.registry.health("presentation").status is HealthStatus.ok
        assert "presentation" in app.available_capabilities()
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_off_by_default(tmp_path):
    app = build_application(_settings()) # enable_presentation_demo defaults False
    await app.startup()
    try:
        assert app.get_module("presentation") is None # not loaded, no trace
        assert "presentation" not in app.available_capabilities()
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# surfaced over the EXISTING REST + WS API (no dedicated route)
# --------------------------------------------------------------------------- #
def test_capability_visible_over_rest():
    api = create_app(_settings(enable_presentation_demo=True, auth_enabled=False))
    with TestClient(api) as c:
        body = c.get("/capabilities").json()
        assert "presentation" in body["available"]
        assert "presentation" in body["all"]


def test_capability_visible_over_ws():
    api = create_app(_settings(enable_presentation_demo=True, auth_enabled=False))
    with TestClient(api) as c, c.websocket_connect("/ws") as ws:
        evt = ws.receive_json() # capabilities sent on connect
        assert evt["event"] == "capabilities"
        assert "presentation" in evt["available"]


def test_no_dedicated_presentation_route():
    # ZERO new gateway machinery: there is no presentation-specific route — hosting is purely via the
    # generic capability API. A made-up route 404s; the capability is still visible via /capabilities.
    api = create_app(_settings(enable_presentation_demo=True, auth_enabled=False))
    with TestClient(api) as c:
        assert c.get("/presentation").status_code == 404
        assert c.get("/presentation/state").status_code == 404
        assert "presentation" in c.get("/capabilities").json()["available"]


# --------------------------------------------------------------------------- #
# fault-isolated like every module (failure disables only itself)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_presentation_fault_is_isolated(tmp_path):
    app = Application(modules=[PresentationModule(fail_start=True), _HealthyPeer()])
    await app.startup() # must NOT raise — the registry isolates the failure
    try:
        reg = app.registry
        assert "presentation" in reg.start_errors
        assert reg.health("presentation").status is HealthStatus.down
        assert not reg.is_capability_available("presentation") # only presentation gated
        assert reg.is_capability_available("peer") # the unrelated peer is unaffected
    finally:
        await app.shutdown()
