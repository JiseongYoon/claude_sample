"""Tests for gateway authentication: API-Key + JWT + scopes.

Run inside conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.auth import create_access_token
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec

API_KEY = "test-api-key-1234567890"
# ≥32 bytes (PyJWT warns below that for HS256)
JWT_SECRET = "test-jwt-secret-0123456789abcdef0123456789"


@dataclass
class M:
    name: str
    caps: tuple[str, ...] = ()
    _status: HealthStatus = field(default=HealthStatus.absent, init=False)

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name=self.name, capabilities=self.caps)

    async def start(self) -> None:
        self._status = HealthStatus.ok

    async def stop(self) -> None:
        self._status = HealthStatus.absent

    def health(self) -> Health:
        return Health(self._status)


def _settings(**over) -> Settings:
    base = dict(
        _env_file=None,
        model_safetensors_dir="./models/s",
        model_gguf_dir="./models/g",
        auth_enabled=True,
        api_key=API_KEY,
        jwt_secret=JWT_SECRET,
    )
    base.update(over)
    return Settings(**base)


def _client(settings=None, modules=None) -> TestClient:
    settings = settings or _settings()
    mods = modules if modules is not None else [M("svc", caps=("ping",))]
    return TestClient(create_gateway(Application(modules=mods), settings))


# --- public vs protected ----------------------------------------------------
def test_health_is_public():
    with _client() as c:
        assert c.get("/health").status_code == 200 # no credential


def test_protected_route_401_without_credential():
    with _client() as c:
        r = c.get("/capabilities")
        assert r.status_code == 401
        assert "www-authenticate" in {k.lower() for k in r.headers}


def test_api_key_grants_access():
    with _client() as c:
        r = c.get("/capabilities", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200


def test_bad_api_key_rejected():
    with _client() as c:
        assert c.get("/capabilities", headers={"X-API-Key": "wrong"}).status_code == 401


# --- token issuance + JWT ---------------------------------------------------
def test_issue_token_requires_api_key():
    with _client() as c:
        assert c.post("/auth/token").status_code == 401 # no api key
        r = c.post("/auth/token", headers={"X-API-Key": API_KEY})
        assert r.status_code == 200
        body = r.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"]


def test_jwt_authenticates_protected_route():
    with _client() as c:
        token = c.post("/auth/token", headers={"X-API-Key": API_KEY},
                       json={"scopes": ["read", "invoke"]}).json()["access_token"]
        r = c.get("/capabilities", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


# --- scope enforcement on invoke -------------------------------------------
def test_invoke_requires_invoke_scope():
    with _client() as c:
        # token WITHOUT invoke scope → 403 on invoke
        read_only = c.post("/auth/token", headers={"X-API-Key": API_KEY},
                           json={"scopes": ["read"]}).json()["access_token"]
        r = c.post("/capabilities/ping/invoke",
                   headers={"Authorization": f"Bearer {read_only}"})
        assert r.status_code == 403
        # token WITH invoke scope → 200 (capability is available)
        full = c.post("/auth/token", headers={"X-API-Key": API_KEY},
                      json={"scopes": ["invoke"]}).json()["access_token"]
        r2 = c.post("/capabilities/ping/invoke",
                    headers={"Authorization": f"Bearer {full}"})
        assert r2.status_code == 200
        # API key has full scope → 200
        r3 = c.post("/capabilities/ping/invoke", headers={"X-API-Key": API_KEY})
        assert r3.status_code == 200


# --- error cases ------------------------------------------------------------
def test_expired_jwt_rejected():
    settings = _settings()
    token = create_access_token(settings, "client", ["read"], expires_minutes=-1)
    with _client(settings) as c:
        r = c.get("/capabilities", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401


def test_tampered_jwt_rejected():
    settings = _settings()
    token = create_access_token(settings, "client", ["read"]) + "tamper"
    with _client(settings) as c:
        assert c.get("/capabilities",
                     headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_jwt_signed_with_other_secret_rejected():
    other = _settings(jwt_secret="a-different-secret-0123456789abcdef0123")
    forged = create_access_token(other, "client", ["read", "invoke"])
    with _client(_settings()) as c: # server uses JWT_SECRET, token signed with other
        assert c.get("/capabilities",
                     headers={"Authorization": f"Bearer {forged}"}).status_code == 401


# --- auth disabled (back-compat) -------------------------------------------
def test_auth_disabled_opens_routes():
    with _client(_settings(auth_enabled=False)) as c:
        assert c.get("/capabilities").status_code == 200
        assert c.post("/capabilities/ping/invoke").status_code == 200


# --- websocket auth ---------------------------------------------------------
def _ws_close_code(c, url: str) -> int | None:
    """Open a WS that the gateway will reject; return the close code (or None). The gateway accepts
    then closes with 1008 so a browser receives a real close FRAME (not an HTTP-403 → 1006 that would
    make the client reconnect-loop)."""
    from starlette.websockets import WebSocketDisconnect

    try:
        with c.websocket_connect(url) as ws:
            ws.receive_json()
        return None
    except WebSocketDisconnect as exc:
        return exc.code


def test_ws_rejected_without_token():
    with _client() as c:
        with pytest.raises(Exception):
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()


def test_ws_no_token_closes_1008_frame():
    # the gateway accepts-then-closes (not a pre-accept 403) so the client gets a 1008 close FRAME.
    with _client() as c:
        assert _ws_close_code(c, "/ws") == 1008


def test_ws_api_key_via_query_rejected_1008():
    # THE OPERATOR-REPORTED SCENARIO: a browser WS can't send the x-api-key header, so it passes the
    # credential as ?token=. An API KEY (not a JWT) there is NOT accepted (?token= is a JWT channel) →
    # the gateway closes with a 1008 frame → the client surfaces auth_failed (no reconnect loop). The
    # API key must reach the WS only as a minted JWT (never raw in the URL — it would land in logs).
    with _client() as c:
        assert _ws_close_code(c, f"/ws?token={API_KEY}") == 1008


def test_ws_ok_with_token_query():
    with _client() as c:
        token = c.post("/auth/token", headers={"X-API-Key": API_KEY}).json()["access_token"]
        with c.websocket_connect(f"/ws?token={token}") as ws:
            snap = ws.receive_json()
            assert snap["event"] == "capabilities"


def test_ws_ok_with_api_key_header():
    with _client() as c:
        with c.websocket_connect("/ws", headers={"X-API-Key": API_KEY}) as ws:
            assert ws.receive_json()["event"] == "capabilities"
