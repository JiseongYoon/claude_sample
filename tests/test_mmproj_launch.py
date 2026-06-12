"""— `--mmproj` launch support + `multimodal` capability gating.

The vision projector is an OPERATOR PREREQUISITE: `multimodal` lights up ONLY when `model_mmproj_file`
is configured + present (and serving is up). These tests cover argv assembly, the projector file
validation (same containment as the GGUF, incl. symlink-escape), and the conditional capability advert.
Hermetic; conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.llm_serving import LLMServingModule
from local_ai_agent.modules.model_manager.process import EngineProcessController


def _settings(tmp_path, **over) -> Settings:
    gguf = tmp_path / "gguf"
    gguf.mkdir(exist_ok=True)
    (gguf / "model.gguf").write_bytes(b"\x00")
    base = dict(_env_file=None, model_safetensors_dir=str(gguf), model_gguf_dir=str(gguf),
                gguf_file="model.gguf")
    base.update(over)
    return Settings(**base)


class FakeTarget:
    def __init__(self, serving=True, base="http://127.0.0.1:8000"):
        self.is_serving = serving
        self.base_url = base


# --------------------------------------------------------------------------- #
# argv + capability — normal
# --------------------------------------------------------------------------- #
def test_argv_includes_mmproj_when_configured(tmp_path):
    s = _settings(tmp_path, model_mmproj_file="proj.gguf")
    (Path(s.model_gguf_dir) / "proj.gguf").write_bytes(b"\x00")
    ctrl = EngineProcessController(s)
    argv = ctrl.build_argv(Path(s.model_gguf_dir) / "model.gguf")
    assert "--mmproj" in argv
    assert argv[argv.index("--mmproj") + 1].endswith("proj.gguf")


def test_argv_no_mmproj_when_unset(tmp_path):
    ctrl = EngineProcessController(_settings(tmp_path)) # no model_mmproj_file
    argv = ctrl.build_argv(Path("/tmp/model.gguf"))
    assert "--mmproj" not in argv


def test_resolve_mmproj_none_when_unset(tmp_path):
    assert EngineProcessController(_settings(tmp_path)).resolve_mmproj_path() is None


def test_llm_serving_advertises_multimodal_when_configured(tmp_path):
    s = _settings(tmp_path, model_mmproj_file="proj.gguf")
    mod = LLMServingModule(s, FakeTarget())
    assert "multimodal" in mod.spec.capabilities and "chat" in mod.spec.capabilities


def test_llm_serving_no_multimodal_when_unset(tmp_path):
    mod = LLMServingModule(_settings(tmp_path), FakeTarget())
    assert mod.spec.capabilities == ("chat",)


# --------------------------------------------------------------------------- #
# projector validation — error / security
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["../escape.gguf", "sub/proj.gguf", "proj.bin", "proj"])
def test_bad_mmproj_name_rejected(tmp_path, bad):
    s = _settings(tmp_path, model_mmproj_file=bad)
    with pytest.raises(ValueError):
        EngineProcessController(s).resolve_mmproj_path()


def test_missing_mmproj_file_rejected(tmp_path):
    s = _settings(tmp_path, model_mmproj_file="absent.gguf") # not created
    with pytest.raises(ValueError):
        EngineProcessController(s).resolve_mmproj_path()


def test_symlinked_mmproj_escaping_dir_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "real.gguf").write_bytes(b"\x00")
    s = _settings(tmp_path, model_mmproj_file="link.gguf")
    (Path(s.model_gguf_dir) / "link.gguf").symlink_to(outside / "real.gguf")
    with pytest.raises(ValueError):
        EngineProcessController(s).resolve_mmproj_path()


def test_build_argv_raises_on_bad_mmproj(tmp_path):
    # a misconfigured projector fails fast at argv build (no silent text-only fallback)
    s = _settings(tmp_path, model_mmproj_file="../escape.gguf")
    with pytest.raises(ValueError):
        EngineProcessController(s).build_argv(Path(s.model_gguf_dir) / "model.gguf")
