"""acceptance gate: fault isolation + live capability gating,
proven end-to-end through the API.

Run inside conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import HealthStatus
from local_ai_agent.main import build_application
from local_ai_agent.modules.demo import DependentModule, EchoModule, FlakyModule


def _settings(**over) -> Settings:
    base = dict(
        _env_file=None,
        model_safetensors_dir="./models/s",
        model_gguf_dir="./models/g",
        auth_enabled=False, # isolate fault-isolation behaviour from auth
    )
    base.update(over)
    return Settings(**base)


def _client(modules):
    return TestClient(create_gateway(Application(modules=list(modules)), _settings()))


# --- all healthy ------------------------------------------------------------
def test_all_healthy_all_capabilities_available():
    with _client([EchoModule(), FlakyModule(), DependentModule()]) as c:
        caps = c.get("/capabilities").json()
        assert set(caps["available"]) == {"echo", "flaky", "dependent"}
        assert c.get("/health").status_code == 200
        for cap in ("echo", "flaky", "dependent"):
            assert c.post(f"/capabilities/{cap}/invoke").status_code == 200


# --- broken module isolates itself + dependents -----------------------------
def test_broken_module_isolates_self_and_dependents():
    with _client([EchoModule(), FlakyModule(broken=True), DependentModule()]) as c:
        caps = c.get("/capabilities").json()
        # only the unrelated module remains available
        assert caps["available"] == ["echo"]
        assert "flaky" not in caps["available"]
        assert "dependent" not in caps["available"] # cascade gated (depends on flaky)
        # unrelated capability still works; gated ones return 503 (not a crash)
        assert c.post("/capabilities/echo/invoke").status_code == 200
        assert c.post("/capabilities/flaky/invoke").status_code == 503
        assert c.post("/capabilities/dependent/invoke").status_code == 503
        # health rolls up to 503 (flaky down) but the server is up and serving
        assert c.get("/health").status_code == 503
        assert c.get("/capabilities").status_code == 200


# --- live health transition reflected immediately ---------------------------
def test_live_health_transition_regates_capabilities():
    flaky = FlakyModule()
    with _client([EchoModule(), flaky, DependentModule()]) as c:
        assert set(c.get("/capabilities").json()["available"]) == {"echo", "flaky", "dependent"}
        # flip flaky down at runtime → fresh query immediately re-gates
        flaky.set_status(HealthStatus.down)
        assert c.get("/capabilities").json()["available"] == ["echo"]
        assert c.get("/capabilities/dependent").json()["available"] is False
        # recover → capabilities return
        flaky.set_status(HealthStatus.ok)
        assert set(c.get("/capabilities").json()["available"]) == {"echo", "flaky", "dependent"}


def test_degraded_dependency_keeps_dependent_available():
    flaky = FlakyModule()
    with _client([EchoModule(), flaky, DependentModule()]) as c:
        flaky.set_status(HealthStatus.degraded) # degraded still counts as available
        caps = set(c.get("/capabilities").json()["available"])
        assert {"flaky", "dependent"} <= caps
        assert c.post("/capabilities/dependent/invoke").status_code == 200


# --- composition-root wiring ------------------------------------------------
def test_build_application_wires_demo_modules_only_when_enabled():
    off = build_application(_settings(enable_demo_modules=False))
    assert off.modules == []
    on = build_application(_settings(enable_demo_modules=True))
    assert {m.spec.name for m in on.modules} == {"echo", "flaky", "dependent"}


def test_build_application_demo_flaky_broken_flag():
    app = build_application(_settings(enable_demo_modules=True, demo_flaky_broken=True))
    api = create_gateway(app, _settings(enable_demo_modules=True, demo_flaky_broken=True))
    with TestClient(api) as c:
        assert c.get("/capabilities").json()["available"] == ["echo"]
