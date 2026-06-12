"""— regression (the phase completion gate).

Mechanism (operator-chosen): **hermetic composed integration with a FAKE multimodal model + capability-
absent fault isolation** (a real mmproj projector is an operator prerequisite, NOT in the repo, so an
in-suite real-model visual smoke is impossible — the operator runs it manually once they generate one).

Drives the REAL `build_application` and asserts the visual unit works as an INTEGRATED WHOLE:
  upload image/scan (real `/ingest`) → reference by id in `run_task` → the agent calls the **gated**
  `answer_about_image` (via the dispatcher/gate, INV-1) → the ingested visual content reaches the model as
  OpenAI vision parts → task_result.
AND the capability-absent path: with text-only serving the visual path is cleanly absent (image upload 415,
the tool absent from the roster, text DocQA + chat unaffected — fault isolation). Hermetic; conda env.
"""
from __future__ import annotations

import json

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall

fitz = pytest.importorskip("fitz")


def _pdf(n=2):
    doc = fitz.open()
    for _ in range(n):
        doc.new_page(width=200, height=200).insert_text((20, 40), "scan")
    data = doc.tobytes()
    doc.close()
    return data


class RecordingVisionServing:
    """llm-serving (Module + ChatModel) that records whether it received a VISION (image_url) part."""

    def __init__(self):
        self._s = False
        self.saw_image = False
        self.image_parts = 0

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
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                parts = [x for x in c if isinstance(x, dict) and x.get("type") == "image_url"]
                if parts:
                    self.saw_image = True
                    self.image_parts = max(self.image_parts, len(parts))
        return {"choices": [{"message": {"content": "it is a chart"}}]}


class VisionAgentModel:
    """Reads the attachment `id=` from the system prompt, calls the gated answer_about_image on it once."""

    def __init__(self):
        self.i = 0

    async def complete(self, messages, tools):
        self.i += 1
        if self.i == 1:
            import re

            sys = next((m["content"] for m in messages if m.get("role") == "system"), "") or ""
            m = re.search(r"id=(\S+)", sys)
            aid = m.group(1) if m else "missing"
            return AssistantTurn(tool_calls=[ToolCall(
                "t1", "answer_about_image", {"attachment_id": aid, "question": "what is in it?"})])
        return AssistantTurn(text="done")


def _settings(tmp_path, *, multimodal: bool, **over):
    gguf = tmp_path / "gguf"
    gguf.mkdir(exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(_env_file=None, model_safetensors_dir=str(gguf), model_gguf_dir=str(gguf),
                auth_enabled=False, enable_docqa=True, enable_agent=True, docs_root=str(docs))
    if multimodal:
        (gguf / "proj.gguf").write_bytes(b"\x00")
        base["model_mmproj_file"] = "proj.gguf"
    base.update(over)
    return Settings(**base)


def _build(s, serving, agent_model):
    app = build_application(s, overrides=BuildOverrides(serving_module=serving, agent_model=agent_model))
    return app, create_gateway(app, s)


def _run_ws_with_attachment(client, aid):
    events = []
    with client.websocket_connect("/ws") as ws: # auth off → anonymous full-scope
        ws.send_text(json.dumps({"action": "run_task", "task": "describe the attachment",
                                 "attachments": [aid]}))
        for _ in range(40):
            ev = ws.receive_json()
            events.append(ev)
            if ev.get("event") == "task_result":
                break
    return events


# --------------------------------------------------------------------------- #
# headline — composed visual ingest reaches the model via the GATED tool (INV-1)
# --------------------------------------------------------------------------- #
def test_stage5_image_answered_via_gated_tool(tmp_path):
    from fastapi.testclient import TestClient

    s = _settings(tmp_path, multimodal=True)
    serving = RecordingVisionServing()
    app, api = _build(s, serving, VisionAgentModel())
    with TestClient(api) as c:
        up = c.post("/ingest", files={"file": ("chart.png", b"\x89PNGchartbytes", "image/png")})
        assert up.status_code == 200, up.text
        events = _run_ws_with_attachment(c, up.json()["id"])
    result = next((e for e in events if e.get("event") == "task_result"), None)
    assert result is not None and result["status"] == "completed"
    assert serving.saw_image is True # the image reached the model as a vision part
    assert serving.image_parts == 1


def test_stage5_scan_pdf_rasterized_to_vision_parts(tmp_path):
    from fastapi.testclient import TestClient

    s = _settings(tmp_path, multimodal=True)
    serving = RecordingVisionServing()
    app, api = _build(s, serving, VisionAgentModel())
    with TestClient(api) as c:
        up = c.post("/ingest", files={"file": ("scan.pdf", _pdf(2), "application/pdf")})
        assert up.status_code == 200, up.text
        events = _run_ws_with_attachment(c, up.json()["id"])
    assert next((e for e in events if e.get("event") == "task_result"), {}).get("status") == "completed"
    assert serving.saw_image and serving.image_parts == 2 # 2 rendered pages → 2 vision parts


# --------------------------------------------------------------------------- #
# capability-absent fault isolation (text-only serving — no projector)
# --------------------------------------------------------------------------- #
def test_stage5_text_only_isolates_visual_path(tmp_path):
    from fastapi.testclient import TestClient

    s = _settings(tmp_path, multimodal=False) # no projector
    app, api = _build(s, RecordingVisionServing(), VisionAgentModel())
    roster = {r["name"] for r in app.agent_runtime.dispatcher.roster()}
    # the visual tool is ABSENT; the text DocQA tools are present (text path intact)
    assert "answer_about_image" not in roster
    assert {"summarize_document", "answer_question", "list_documents"} <= roster
    with TestClient(api) as c:
        # an image upload is rejected (415) — images are not in the text-only allowlist
        assert c.post("/ingest", files={"file": ("x.png", b"\x89PNG", "image/png")}).status_code == 415
        # a TEXT upload still works (the text ingest path is unaffected)
        assert c.post("/ingest", files={"file": ("note.txt", b"hello", "text/plain")}).status_code == 200


def test_stage5_multimodal_tool_present_when_configured(tmp_path):
    s = _settings(tmp_path, multimodal=True)
    app, _ = _build(s, RecordingVisionServing(), VisionAgentModel())
    roster = {r["name"]: r["tier"] for r in app.agent_runtime.dispatcher.roster()}
    assert roster.get("answer_about_image") == "safe" # gated tool, safe-listed read
