"""acceptance gate: full model-stack flow end-to-end through the
gateway, hermetic (fake launcher + fake transport). Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
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
    def crash(self): self._r = False


class _Launcher:
    def __init__(self): self.last_handle: _Handle | None = None
    async def spawn(self, argv, env):
        self.last_handle = _Handle()
        return self.last_handle


async def _ready(base_url): return True


class _Transport:
    async def __call__(self, base_url, payload):
        return {"choices": [{"message": {"content": "ok"}}], "_payload": payload}


@pytest.fixture
def gguf_dir(tmp_path: Path) -> Path:
    for n in (GGUF, GGUF_Q4):
        (tmp_path / n).write_bytes(b"\x00")
    return tmp_path


def _build(gguf_dir: Path):
    s = Settings(_env_file=None, model_safetensors_dir=str(gguf_dir), model_gguf_dir=str(gguf_dir),
                 gguf_file=GGUF, auth_enabled=True, api_key=API_KEY, jwt_secret=JWT_SECRET)
    launcher = _Launcher()
    ctrl = EngineProcessController(s, launcher=launcher, readiness=_ready,
                                   readiness_timeout=5.0, poll_interval=0.01)
    manager = ModelManagerModule(s, controller=ctrl)
    serving = LLMServingModule(s, manager, transport=_Transport())
    client = TestClient(create_gateway(Application(modules=[manager, serving]), s))
    return client, launcher


KEY = {"X-API-Key": API_KEY}


def _wait_ready(c, tries=100):
    for _ in range(tries):
        st = c.get("/model/status", headers=KEY).json()["state"]
        if st in ("ready", "error"):
            return st
    return st


def test_full_lifecycle_end_to_end(gguf_dir):
    client, _ = _build(gguf_dir)
    with client as c:
        msg = {"messages": [{"role": "user", "content": "hi"}]}
        # before load: gated
        assert c.post("/chat", headers=KEY, json=msg).status_code == 503
        assert c.post("/model/params", headers=KEY, json={"temperature": 0.5}).status_code == 409
        # load
        assert c.post("/model/load", headers=KEY, json={}).status_code == 202
        assert _wait_ready(c) == "ready"
        status = c.get("/model/status", headers=KEY).json()
        assert status["serving"] is True and status["loaded_file"] == GGUF
        # set params → applied on chat
        assert c.post("/model/params", headers=KEY, json={"temperature": 0.3}).status_code == 200
        chat = c.post("/chat", headers=KEY, json=msg).json()
        assert chat["choices"][0]["message"]["content"] == "ok"
        assert chat["_payload"]["temperature"] == 0.3 # stored default applied
        # switch quant
        assert c.post("/model/switch", headers=KEY, json={"gguf_file": GGUF_Q4}).status_code in (202, 200)
        assert _wait_ready(c) == "ready"
        assert c.get("/model/status", headers=KEY).json()["loaded_file"] == GGUF_Q4
        # unload → gated again
        assert c.post("/model/unload", headers=KEY).json()["state"] == "unloaded"
        assert c.post("/chat", headers=KEY, json=msg).status_code == 503


def test_engine_crash_surfaced_over_api(gguf_dir):
    client, launcher = _build(gguf_dir)
    with client as c:
        c.post("/model/load", headers=KEY, json={})
        assert _wait_ready(c) == "ready"
        # kill the engine process underneath
        launcher.last_handle.crash()
        status = c.get("/model/status", headers=KEY).json()
        assert status["state"] == "error" and status["serving"] is False
        # chat is gated (503), not a 500
        assert c.post("/chat", headers=KEY,
                      json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 503
        # capability gated, server still serving other requests
        assert "chat" not in c.get("/capabilities", headers=KEY).json()["available"]
        assert c.get("/health").status_code in (200, 503) # up, not crashed


def test_scope_enforcement_across_lifecycle(gguf_dir):
    client, _ = _build(gguf_dir)
    with client as c:
        inv = c.post("/auth/token", headers=KEY, json={"scopes": ["read", "invoke"]}).json()["access_token"]
        invoke_hdr = {"Authorization": f"Bearer {inv}"}
        # lifecycle needs model:admin
        assert c.post("/model/load", headers=invoke_hdr, json={}).status_code == 403
        assert c.post("/model/unload", headers=invoke_hdr).status_code == 403
        assert c.post("/model/switch", headers=invoke_hdr, json={"gguf_file": GGUF_Q4}).status_code == 403
        # chat needs invoke (read-only denied)
        ro = c.post("/auth/token", headers=KEY, json={"scopes": ["read"]}).json()["access_token"]
        assert c.post("/chat", headers={"Authorization": f"Bearer {ro}"},
                      json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 403


def test_invalid_quant_rejected_over_api(gguf_dir):
    client, _ = _build(gguf_dir)
    with client as c:
        r = c.post("/model/load", headers=KEY, json={"gguf_file": "../../etc/shadow.gguf"})
        assert r.status_code == 400
        assert "etc" not in r.text # generic, no path leak
