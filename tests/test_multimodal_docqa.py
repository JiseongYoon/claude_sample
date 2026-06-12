"""— the gated multimodal DocQA tool (answer_about_image).

Image → base64 vision part; scan-PDF → bounded per-page vision parts; → chat → answer. Registered +
reachable ONLY when a projector is configured (the visual path is otherwise absent). Hermetic with a
FAKE multimodal model (records the vision content it received). conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.docqa.ingest import UnknownIngestId
from local_ai_agent.modules.docqa.multimodal import AnswerAboutImageTool

fitz = pytest.importorskip("fitz")

_BOUNDS = dict(max_pages=20, max_page_px=4_000_000, max_render_bytes=50_000_000, render_timeout_s=30.0)


def _pdf(n=2):
    doc = fitz.open()
    for _ in range(n):
        doc.new_page(width=200, height=200).insert_text((20, 40), "hi")
    data = doc.tobytes()
    doc.close()
    return data


class _Rec:
    def __init__(self, ext):
        self.ext = ext


class FakeStore:
    def __init__(self, records):
        self._r = records # id -> (ext, bytes)

    def get(self, ingest_id):
        if ingest_id not in self._r:
            raise UnknownIngestId("unknown")
        return _Rec(self._r[ingest_id][0])

    def read_bytes(self, ingest_id):
        if ingest_id not in self._r:
            raise UnknownIngestId("unknown")
        return self._r[ingest_id][1]


class FakeVisionChat:
    def __init__(self, raises=False):
        self.last_messages = None
        self.raises = raises

    async def chat(self, messages, **params):
        if self.raises:
            raise RuntimeError("serving down")
        self.last_messages = messages
        return {"choices": [{"message": {"content": "a cat"}}]}


def _tool(store, chat):
    return AnswerAboutImageTool(store, chat, **_BOUNDS)


def _content(chat):
    return chat.last_messages[0]["content"]


# --------------------------------------------------------------------------- #
# normal
# --------------------------------------------------------------------------- #
async def test_image_becomes_one_vision_part():
    chat = FakeVisionChat()
    store = FakeStore({"img1": (".png", b"\x89PNGfake")})
    out = await _tool(store, chat).run({"attachment_id": "img1", "question": "what is this?"})
    assert out["ok"] is True and out["pages"] == 1
    parts = _content(chat)
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_jpeg_mime():
    chat = FakeVisionChat()
    out = await _tool(FakeStore({"j": (".jpg", b"\xff\xd8\xff")}), chat).run(
        {"attachment_id": "j", "question": "q"})
    assert out["ok"] and _content(chat)[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_pdf_becomes_per_page_vision_parts():
    chat = FakeVisionChat()
    store = FakeStore({"doc": (".pdf", _pdf(3))})
    out = await _tool(store, chat).run({"attachment_id": "doc", "question": "summarize"})
    assert out["ok"] and out["pages"] == 3
    parts = _content(chat)
    assert len(parts) == 1 + 3 # text + 3 page images
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts[1:])


# --------------------------------------------------------------------------- #
# error / security
# --------------------------------------------------------------------------- #
async def test_unknown_id_graceful():
    chat = FakeVisionChat()
    out = await _tool(FakeStore({}), chat).run({"attachment_id": "nope", "question": "q"})
    assert out["ok"] is False and "UnknownIngestId" in out["error"]
    assert chat.last_messages is None # model never called


async def test_unsupported_ext_graceful():
    out = await _tool(FakeStore({"t": (".txt", b"hi")}), FakeVisionChat()).run(
        {"attachment_id": "t", "question": "q"})
    assert out["ok"] is False


@pytest.mark.parametrize("args", [
    {"question": "q"}, # no attachment_id
    {"attachment_id": "x"}, # no question
    {"attachment_id": "", "question": "q"},
    {"attachment_id": "x", "question": " "},
    "not-a-dict",
])
async def test_bad_args_rejected(args):
    out = await _tool(FakeStore({"x": (".png", b"x")}), FakeVisionChat()).run(args)
    assert out["ok"] is False


async def test_serving_down_graceful():
    out = await _tool(FakeStore({"i": (".png", b"x")}), FakeVisionChat(raises=True)).run(
        {"attachment_id": "i", "question": "q"})
    assert out["ok"] is False and "model unavailable" in out["error"]


# --------------------------------------------------------------------------- #
# registration gating — the tool exists ONLY when a projector is configured
# --------------------------------------------------------------------------- #
class FakeServing:
    def __init__(self):
        self._s = False

    @property
    def spec(self):
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self):
        self._s = True

    async def stop(self):
        self._s = False

    def health(self):
        return Health(HealthStatus.ok if self._s else HealthStatus.absent, "fake")

    async def chat(self, messages, **p):
        return {"choices": [{"message": {"content": "x"}}]}


class ScriptedAgent:
    async def complete(self, messages, tools):
        from local_ai_agent.modules.orchestrator.loop import AssistantTurn
        return AssistantTurn(text="done")


def _settings(tmp_path, **over):
    gguf = tmp_path / "gguf"
    gguf.mkdir(exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(_env_file=None, model_safetensors_dir=str(gguf), model_gguf_dir=str(gguf),
                auth_enabled=False, enable_docqa=True, enable_agent=True, docs_root=str(docs))
    base.update(over)
    return Settings(**base)


def _roster_names(s):
    app = build_application(s, overrides=BuildOverrides(serving_module=FakeServing(), agent_model=ScriptedAgent()))
    roster = app.agent_runtime.dispatcher.roster()
    return {r["name"]: r["tier"] for r in roster}


def test_tool_registered_safe_when_multimodal(tmp_path):
    s = _settings(tmp_path, model_mmproj_file="proj.gguf")
    (Path(s.model_gguf_dir) / "proj.gguf").write_bytes(b"\x00")
    names = _roster_names(s)
    assert "answer_about_image" in names
    assert names["answer_about_image"] == "safe" # a read tool, safe-listed


def test_tool_absent_when_text_only(tmp_path):
    names = _roster_names(_settings(tmp_path)) # no projector
    assert "answer_about_image" not in names
