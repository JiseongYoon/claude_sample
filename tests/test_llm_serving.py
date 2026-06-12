"""Tests for LLMServingModule.

Fake serving target (toggle is_serving) + fake transport (no real HTTP/model).
Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import HealthStatus, Module
from local_ai_agent.core.registry import ModuleRegistry
from local_ai_agent.modules.llm_serving import EngineNotReady, LLMServingModule
from local_ai_agent.modules.model_manager.module import ModelManagerModule
from local_ai_agent.modules.model_manager.process import EngineProcessController

GGUF = "gemma-4-31B-it-UD-Q6_K_XL.gguf"


class FakeTarget:
    def __init__(self, serving: bool = False, base_url: str = "http://127.0.0.1:8000") -> None:
        self._serving = serving
        self._base_url = base_url

    @property
    def is_serving(self) -> bool:
        return self._serving

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_serving(self, v: bool) -> None:
        self._serving = v


class FakeTransport:
    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        self.response = response or {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, base_url: str, payload: dict) -> dict:
        self.calls.append((base_url, payload))
        if self.error:
            raise self.error
        return self.response


def _settings() -> Settings:
    return Settings(_env_file=None, model_safetensors_dir="./m", model_gguf_dir="./m")


def _module(serving=False, transport=None):
    target = FakeTarget(serving=serving)
    mod = LLMServingModule(_settings(), target, transport=transport or FakeTransport())
    return mod, target


# --- spec / protocol --------------------------------------------------------
def test_spec_and_protocol():
    mod, _ = _module()
    assert isinstance(mod, Module)
    assert mod.spec.name == "llm-serving"
    assert mod.spec.capabilities == ("chat",)
    assert mod.spec.depends_on == ("model-manager",)


# --- health gated on serve-readiness ---------------------------------------
async def test_health_absent_before_start():
    mod, _ = _module(serving=True)
    assert mod.health().status is HealthStatus.absent


async def test_health_ok_only_when_serving():
    mod, target = _module(serving=False)
    await mod.start()
    assert mod.health().status is HealthStatus.absent # no model ready
    target.set_serving(True)
    assert mod.health().status is HealthStatus.ok
    target.set_serving(False)
    assert mod.health().status is HealthStatus.absent


# --- registry gating with the real model-manager ---------------------------
async def test_chat_capability_gated_on_model_load(tmp_path: Path):
    (tmp_path / GGUF).write_bytes(b"\x00")

    class _Handle:
        def __init__(self): self._r = True
        @property
        def pid(self): return 1
        def running(self): return self._r
        async def terminate(self, timeout=10.0): self._r = False

    class _Launcher:
        async def spawn(self, argv, env): return _Handle()

    async def _ready(b): return True

    settings = Settings(_env_file=None, model_safetensors_dir=str(tmp_path),
                        model_gguf_dir=str(tmp_path), gguf_file=GGUF)
    ctrl = EngineProcessController(settings, launcher=_Launcher(), readiness=_ready,
                                   readiness_timeout=5.0, poll_interval=0.01)
    manager = ModelManagerModule(settings, controller=ctrl)
    serving = LLMServingModule(settings, manager, transport=FakeTransport())

    reg = ModuleRegistry()
    reg.register(manager)
    reg.register(serving)
    await reg.start_all()

    # no model loaded yet → chat unavailable (manager subsystem ok, but not serving)
    assert reg.is_module_available("model-manager") is True
    assert "chat" not in reg.available_capabilities()

    await manager.load() # load a model
    assert manager.is_serving is True
    assert "chat" in reg.available_capabilities() # now available

    await manager.unload() # unload
    assert "chat" not in reg.available_capabilities()


# --- chat call --------------------------------------------------------------
async def test_chat_posts_expected_payload():
    transport = FakeTransport(response={"choices": [{"message": {"content": "42"}}]})
    mod, _ = _module(serving=True, transport=transport)
    await mod.start()
    msgs = [{"role": "user", "content": "q"}]
    out = await mod.chat(msgs, temperature=0.2, top_p=0.9, max_tokens=16)
    assert out["choices"][0]["message"]["content"] == "42"
    base_url, payload = transport.calls[0]
    assert base_url == "http://127.0.0.1:8000"
    assert payload["messages"] == msgs
    assert payload["model"] == "gemma-4-31b-it"
    assert payload["temperature"] == 0.2 and payload["top_p"] == 0.9 and payload["max_tokens"] == 16
    assert "top_k" not in payload # only supplied params included


async def test_chat_raises_when_not_serving():
    mod, _ = _module(serving=False)
    await mod.start()
    with pytest.raises(EngineNotReady):
        await mod.chat([{"role": "user", "content": "q"}])


async def test_transport_error_propagates_cleanly():
    transport = FakeTransport(error=RuntimeError("upstream 500"))
    mod, _ = _module(serving=True, transport=transport)
    await mod.start()
    with pytest.raises(RuntimeError, match="upstream 500"):
        await mod.chat([{"role": "user", "content": "q"}])
