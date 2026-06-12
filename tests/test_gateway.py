"""Tests for the API gateway incl. the security baseline.

Uses FastAPI TestClient (httpx). Lifespan (Application startup/shutdown) runs via
the `with TestClient(...)` context. Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec


@dataclass
class M:
    name: str
    caps: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()
    fail_on_start: bool = False
    _status: HealthStatus = field(default=HealthStatus.absent, init=False)

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name=self.name, capabilities=self.caps, depends_on=self.deps)

    async def start(self) -> None:
        if self.fail_on_start:
            raise RuntimeError("boom-secret-path-/etc/x")
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status)


def _settings(**over) -> Settings:
    # auth is exercised in test_auth.py; these gateway-behaviour tests run with
    # auth disabled so they focus on routing/health/gating/security-headers.
    base = dict(
        _env_file=None,
        model_safetensors_dir="./models/gemma-4-safetensors",
        model_gguf_dir="./models/gemma-4-gguf",
        auth_enabled=False,
    )
    base.update(over)
    return Settings(**base)


def _client(modules, settings=None) -> TestClient:
    settings = settings or _settings()
    api = create_gateway(Application(modules=list(modules)), settings)
    return TestClient(api)


# --- health -----------------------------------------------------------------
def test_health_ok_when_all_healthy():
    with _client([M("svc", caps=("ping",))]) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["modules"]["svc"] == "ok" # redacted: plain string, no detail


def test_health_503_when_module_down():
    with _client([M("model", caps=("load",), fail_on_start=True)]) as c:
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json()["status"] == "down"


def test_health_detail_redacted_by_default():
    with _client([M("model", caps=("load",), fail_on_start=True)]) as c:
        body = c.get("/health").json()
        # value is a bare status string; the exception text must NOT leak
        assert body["modules"]["model"] == "down"
        assert "boom-secret-path" not in c.get("/health").text


def test_health_detail_exposed_when_enabled():
    s = _settings(expose_health_detail=True)
    with _client([M("model", caps=("load",), fail_on_start=True)], s) as c:
        body = c.get("/health").json()
        assert body["modules"]["model"]["status"] == "down"
        assert "boom-secret-path" in body["modules"]["model"]["detail"]


# --- capabilities + gating --------------------------------------------------
def test_capabilities_and_gating():
    mods = [
        M("svc", caps=("ping",)),
        M("model", caps=("load",), fail_on_start=True), # down
        M("tuner", caps=("tune",), deps=("model",)), # gated by model
    ]
    with _client(mods) as c:
        caps = c.get("/capabilities").json()
        assert "ping" in caps["available"]
        assert "tune" not in caps["available"]
        assert set(caps["all"]) == {"ping", "load", "tune"}
        assert c.get("/capabilities/tune").json()["available"] is False
        assert c.get("/capabilities/ping").json()["available"] is True


def test_invoke_gating_status_codes():
    mods = [
        M("svc", caps=("ping",)),
        M("model", caps=("load",), fail_on_start=True),
        M("tuner", caps=("tune",), deps=("model",)),
    ]
    with _client(mods) as c:
        assert c.post("/capabilities/ping/invoke", json={"x": 1}).status_code == 200
        assert c.post("/capabilities/tune/invoke").status_code == 503 # gated unavailable
        assert c.post("/capabilities/nope/invoke").status_code == 404 # unknown


def test_module_status_404_for_unknown():
    with _client([M("svc", caps=("ping",))]) as c:
        assert c.get("/modules/ghost").status_code == 404
        assert c.get("/modules/svc").json()["available"] is True


# --- security ---------------------------------------------------------------
def test_security_headers_present():
    with _client([M("svc")]) as c:
        h = c.get("/health").headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert h["Referrer-Policy"] == "no-referrer"


def test_cors_not_enabled_by_default():
    with _client([M("svc")]) as c:
        r = c.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_cors_preflight_allows_x_api_key_header():
    # A browser client mints a token (POST /auth/token) and makes api-key REST calls with the
    # `X-API-Key` header → the CORS preflight MUST allow it, else the UI can't authenticate cross-origin.
    s = _settings(cors_allow_origins=["http://ui.local"])
    with _client([M("svc")], s) as c:
        r = c.options(
            "/auth/token",
            headers={
                "Origin": "http://ui.local",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type, x-api-key",
            },
        )
        allowed = r.headers.get("access-control-allow-headers", "").lower()
        assert "x-api-key" in allowed
        assert r.headers.get("access-control-allow-origin") == "http://ui.local"


def test_untrusted_host_rejected():
    with _client([M("svc")]) as c:
        r = c.get("/health", headers={"host": "evil.example"})
        assert r.status_code == 400 # TrustedHostMiddleware


def test_generic_500_no_internal_leak():
    settings = _settings()
    api = create_gateway(Application(modules=[M("svc")]), settings)

    @api.get("/boom")
    async def boom(): # noqa: ANN202
        raise RuntimeError("super-secret internal detail")

    with TestClient(api, raise_server_exceptions=False) as c:
        r = c.get("/boom")
        assert r.status_code == 500
        assert r.json() == {"detail": "internal server error"}
        assert "super-secret" not in r.text
        # server stays up
        assert c.get("/health").status_code == 200


# --- websocket --------------------------------------------------------------
def test_websocket_scaffold():
    with _client([M("svc", caps=("ping",))]) as c:
        with c.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            assert first["event"] == "capabilities"
            assert "ping" in first["available"]
            ws.send_text("hi")
            echo = ws.receive_json()
            assert echo == {"event": "echo", "data": "hi"}
