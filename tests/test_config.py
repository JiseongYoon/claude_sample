"""Tests for the config layer.

Verifies Settings against Acceptance: normal (typed load + defaults) and
error (graceful ValidationError, no crash/corruption). Run inside conda
`local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from local_ai_agent.config import ServeEngine, Settings, SplitMode

# Required fields supplied directly (kwargs override env). _env_file=None isolates
# the test from any real .env on disk.
_BASE = dict(
    _env_file=None,
    model_safetensors_dir="./models/gemma-4-safetensors",
    model_gguf_dir="./models/gemma-4-gguf",
)


def test_typed_load_and_paths():
    s = Settings(**_BASE, serve_port=8001, tensor_parallel_size=2)
    assert isinstance(s.model_safetensors_dir, Path)
    assert isinstance(s.model_gguf_dir, Path)
    assert s.serve_port == 8001
    assert s.tensor_parallel_size == 2


def test_defaults_apply():
    s = Settings(**_BASE)
    assert s.serve_engine is ServeEngine.llamacpp # llama.cpp primary
    assert s.serve_host == "127.0.0.1"
    assert s.serve_port == 8000
    assert s.cuda_visible_devices == "0,1"
    assert s.openai_base_url == "http://127.0.0.1:8000/v1"


def test_llamacpp_serve_defaults():
    s = Settings(**_BASE)
    assert s.n_gpu_layers == 999
    assert s.split_mode is SplitMode.layer
    assert s.tensor_split == "1,1"
    assert s.ctx_size == 16384
    assert s.use_jinja is True


def test_engine_enum_parses_from_string():
    s = Settings(**_BASE, serve_engine="vllm")
    assert s.serve_engine is ServeEngine.vllm


def test_dropped_sglang_engine_rejected():
    # sglang was removed from the enum → must not parse.
    with pytest.raises(ValidationError):
        Settings(**_BASE, serve_engine="sglang")


def test_bad_ctx_size_rejected():
    with pytest.raises(ValidationError):
        Settings(**_BASE, ctx_size=0)


def test_bad_port_rejected():
    with pytest.raises(ValidationError):
        Settings(**_BASE, serve_port=99999)


def test_non_int_port_rejected():
    with pytest.raises(ValidationError):
        Settings(**_BASE, serve_port="not-a-number")


def test_tp_size_must_be_positive():
    with pytest.raises(ValidationError):
        Settings(**_BASE, tensor_parallel_size=0)


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_gguf_dir="./models/gemma-4-gguf")


def test_unknown_engine_rejected():
    with pytest.raises(ValidationError):
        Settings(**_BASE, serve_engine="tensorrt")


# --- follow-up: gateway api_port distinct from the model serve_port -------------------- #
def test_api_port_defaults_distinct_from_serve_port():
    s = Settings(**_BASE)
    assert s.api_port == 8080 and s.serve_port == 8000
    assert s.api_port != s.serve_port


def test_api_port_equal_serve_port_rejected():
    with pytest.raises(ValidationError):
        Settings(**_BASE, api_port=8000, serve_port=8000)


def test_api_port_range_validated():
    with pytest.raises(ValidationError):
        Settings(**_BASE, api_port=0)
