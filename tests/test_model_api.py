"""Tests for the model-stack gateway routes incl. security.

Hermetic: model-manager backed by a fake launcher + immediate readiness;
llm-serving with a fake transport. Auth enabled (API-Key + JWT scopes).
Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.main import build_application
from local_ai_agent.modules.llm_serving import LLMServingModule
from local_ai_agent.modules.model_manager.module import ModelManagerModule
from local_ai_agent.modules.model_manager.process import EngineProcessController

GGUF = "gemma-4-31B-it-UD-Q6_K_XL.gguf"
GGUF_Q4 = "gemma-4-31B-it-UD-Q4_K_XL.gguf"
API_KEY = "test-api-key-1234567890"
JWT_SECRET = "test-jwt-secret-0123456789abcdef0123456789"


class _Handle:
    def __init__(self): self._r = True
    @property
    def pid(self): return 1
    def running(self): return self._r
    async def terminate(self, timeout: float = 10.0): self._r = False


class _Launcher:
    async def spawn(self, argv, env): return _Handle()


async def _ready(base_url): return True


class _Transport:
    async def __call__(self, base_url, payload):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "_payload": payload}


@pytest.fixture
def gguf_dir(tmp_path: Path) -> Path:
    for n in (GGUF, GGUF_Q4):
        (tmp_path / n).write_bytes(b"\x00")
    return tmp_path


def _settings(gguf_dir: Path, **over) -> Settings:
    base = dict(_env_file=None, model_safetensors_dir=str(gguf_dir), model_gguf_dir=str(gguf_dir),
                gguf_file=GGUF, auth_enabled=True, api_key=API_KEY, jwt_secret=JWT_SECRET)
    base.update(over)
    return Settings(**base)


def _stack_client(gguf_dir: Path, settings=None):
    settings = settings or _settings(gguf_dir)
    ctrl = EngineProcessController(settings, launcher=_Launcher(), readiness=_ready,
                                   readiness_timeout=5.0, poll_interval=0.01)
    manager = ModelManagerModule(settings, controller=ctrl)
    serving = LLMServingModule(settings, manager, transport=_Transport())
    return TestClient(create_gateway(Application(modules=[manager, serving]), settings))


def _admin(c) -> dict:
    return {"X-API-Key": API_KEY} # API key = full scope


def _token(c, scopes) -> dict:
    t = c.post("/auth/token", headers={"X-API-Key": API_KEY}, json={"scopes": scopes}).json()["access_token"]
    return {"Authorization": f"Bearer {t}"}


def _wait_ready(c, headers, tries=100) -> str:
    for _ in range(tries):
        st = c.get("/model/status", headers=headers).json()["state"]
        if st in ("ready", "error"):
            return st
    return st


# --- stack absent -----------------------------------------------------------
def test_routes_503_when_stack_absent():
    s = Settings(_env_file=None, model_safetensors_dir="./m", model_gguf_dir="./m",
                 auth_enabled=True, api_key=API_KEY, jwt_secret=JWT_SECRET)
    with TestClient(create_gateway(Application(modules=[]), s)) as c:
        assert c.get("/model/status", headers={"X-API-Key": API_KEY}).status_code == 503
        assert c.post("/chat", headers={"X-API-Key": API_KEY},
                      json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503


# --- status + auth ----------------------------------------------------------
def test_status_requires_auth_and_reports_unloaded(gguf_dir):
    with _stack_client(gguf_dir) as c:
        assert c.get("/model/status").status_code == 401 # no creds
        body = c.get("/model/status", headers=_admin(c)).json()
        assert body["state"] == "unloaded" and body["serving"] is False
        assert "detail" not in body # redacted by default


# --- lifecycle scope + load -------------------------------------------------
def test_load_requires_admin_scope(gguf_dir):
    with _stack_client(gguf_dir) as c:
        read_only = _token(c, ["read", "invoke"])
        assert c.post("/model/load", headers=read_only, json={}).status_code == 403
        r = c.post("/model/load", headers=_admin(c), json={})
        assert r.status_code == 202 and r.json()["accepted"] is True
        assert _wait_ready(c, _admin(c)) == "ready"


def test_invalid_gguf_is_400_generic(gguf_dir):
    with _stack_client(gguf_dir) as c:
        r = c.post("/model/load", headers=_admin(c), json={"gguf_file": "../escape.gguf"})
        assert r.status_code == 400
        assert "escape" not in r.text and "/" not in r.json()["detail"] # generic, no path leak


def test_switch_requires_gguf_and_admin(gguf_dir):
    with _stack_client(gguf_dir) as c:
        assert c.post("/model/load", headers=_admin(c), json={}).status_code == 202
        _wait_ready(c, _admin(c))
        assert c.post("/model/switch", headers=_token(c, ["invoke"]),
                      json={"gguf_file": GGUF_Q4}).status_code == 403
        assert c.post("/model/switch", headers=_admin(c), json={}).status_code == 400 # missing file
        assert c.post("/model/switch", headers=_admin(c),
                      json={"gguf_file": GGUF_Q4}).status_code in (202, 200)


# --- params gating ----------------------------------------------------------
def test_params_gated_on_loaded_model(gguf_dir):
    with _stack_client(gguf_dir) as c:
        # not serving yet → 409
        assert c.post("/model/params", headers=_admin(c),
                      json={"temperature": 0.5}).status_code == 409
        c.post("/model/load", headers=_admin(c), json={})
        _wait_ready(c, _admin(c))
        r = c.post("/model/params", headers=_admin(c), json={"temperature": 0.5, "top_p": 0.9})
        assert r.status_code == 200
        assert c.get("/model/params", headers=_admin(c)).json()["params"]["temperature"] == 0.5
        # range violation → 422 (pydantic)
        assert c.post("/model/params", headers=_admin(c), json={"temperature": 9}).status_code == 422


def test_params_set_requires_admin(gguf_dir):
    with _stack_client(gguf_dir) as c:
        c.post("/model/load", headers=_admin(c), json={})
        _wait_ready(c, _admin(c))
        assert c.post("/model/params", headers=_token(c, ["read", "invoke"]),
                      json={"temperature": 0.5}).status_code == 403


# --- chat -------------------------------------------------------------------
def test_chat_503_until_loaded_then_works(gguf_dir):
    with _stack_client(gguf_dir) as c:
        invoke = _token(c, ["invoke"])
        assert c.post("/chat", headers=invoke,
                      json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503
        c.post("/model/load", headers=_admin(c), json={})
        _wait_ready(c, _admin(c))
        r = c.post("/chat", headers=invoke, json={"messages": [{"role": "user", "content": "hi"}],
                                                  "temperature": 0.2})
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "ok"
        assert r.json()["_payload"]["temperature"] == 0.2


def test_chat_requires_invoke_scope(gguf_dir):
    with _stack_client(gguf_dir) as c:
        c.post("/model/load", headers=_admin(c), json={})
        _wait_ready(c, _admin(c))
        read_only = _token(c, ["read"])
        assert c.post("/chat", headers=read_only,
                      json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 403


def test_chat_max_tokens_clamped(gguf_dir):
    with _stack_client(gguf_dir, _settings(gguf_dir, chat_max_tokens_ceiling=128)) as c:
        c.post("/model/load", headers=_admin(c), json={})
        _wait_ready(c, _admin(c))
        r = c.post("/chat", headers=_admin(c),
                   json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 999999})
        assert r.json()["_payload"]["max_tokens"] == 128 # clamped to ceiling


# --- composition root wiring ------------------------------------------------
def test_build_application_wires_model_stack_only_when_enabled(gguf_dir):
    off = build_application(_settings(gguf_dir, enable_model_stack=False))
    assert off.modules == []
    on = build_application(_settings(gguf_dir, enable_model_stack=True))
    assert {m.spec.name for m in on.modules} == {"model-manager", "llm-serving"}
