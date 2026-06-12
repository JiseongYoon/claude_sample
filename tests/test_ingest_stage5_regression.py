"""— regression (the phase completion gate).

Mechanism (operator-chosen): **hermetic composed integration + real-API e2e**. This file is the hermetic
half — it drives the REAL composition root (`build_application`) and asserts the ingest unit works as an
INTEGRATED WHOLE over the gated path:

  upload (POST /ingest, real endpoint + IngestStore) → reference by id → an ingested doc is read by the
  EXISTING gated DocQA tools through the dispatcher/gate (INV-1) AND by the direct-chat `load_text` path.

Plus the full adversarial battery against the live endpoint. The real-API e2e half lives in `web/e2e/`
(REAL ApiClient ↔ live core over a real socket). Hermetic; conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall

API_KEY = "k" * 40
JWT = "s" * 40
HDR = {"X-API-Key": API_KEY}
HP = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'


def _hwpx(marker: str) -> bytes:
    body = f"<hp:p><hp:run><hp:t>{marker}</hp:t></hp:run></hp:p>"
    section = f'<?xml version="1.0"?><hp:sec {HP}>{body}</hp:sec>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Contents/section0.xml", section)
    return buf.getvalue()


class RecordingServing:
    """llm-serving (Module + ChatModel) that ACCUMULATES every prompt text it is given, so a test can
    assert an ingested document's content actually reached the model through the gated DocQA tool."""

    def __init__(self) -> None:
        self.seen_text = ""

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    def health(self) -> Health:
        return Health(HealthStatus.ok, "fake")

    async def chat(self, messages, **params) -> dict:
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                self.seen_text += c + "\n"
        return {"choices": [{"message": {"content": "SUMMARY"}}]}


class IngestReadingModel:
    """A ToolCallingModel that reads the attachment relpath from the system prompt, calls the gated
    `summarize_document` on it once (turn 1), then returns a final answer (turn 2)."""

    def __init__(self) -> None:
        self.i = 0
        self.path_seen: str | None = None

    async def complete(self, messages, tools) -> AssistantTurn:
        self.i += 1
        if self.i == 1:
            import re

            sys = next((m["content"] for m in messages if m.get("role") == "system"), "") or ""
            m = re.search(r"_ingest/\S+", sys)
            self.path_seen = m.group(0) if m else None
            if self.path_seen:
                return AssistantTurn(tool_calls=[ToolCall("t1", "summarize_document", {"path": self.path_seen})])
        return AssistantTurn(text="done")


def _settings(tmp_path, **over) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(_env_file=None, model_safetensors_dir=str(tmp_path), model_gguf_dir=str(tmp_path),
                auth_enabled=True, api_key=API_KEY, jwt_secret=JWT,
                enable_docqa=True, enable_agent=True, docs_root=str(docs))
    base.update(over)
    return Settings(**base)


def _app(s, *, serving=None, agent_model=None):
    application = build_application(
        s, overrides=BuildOverrides(serving_module=serving or RecordingServing(), agent_model=agent_model),
    )
    return application, create_gateway(application, s)


# --------------------------------------------------------------------------- #
# headline integration — upload → gated DocQA reads the ingested file (INV-1)
# --------------------------------------------------------------------------- #
def test_stage5_ingest_then_gated_docqa_over_agent(tmp_path):
    s = _settings(tmp_path)
    serving = RecordingServing()
    model = IngestReadingModel()
    application, api = _app(s, serving=serving, agent_model=model)
    # mint a full-scope JWT so the WS (browser path = JWT) authenticates and the upload has `ingest`.
    with TestClient(api) as c:
        tok = c.post("/auth/token", headers=HDR,
                     json={"scopes": ["ingest", "agent:run", "approve", "read", "invoke"]}).json()["access_token"]
        up = c.post("/ingest", files={"file": ("notes.txt", b"MARKER-XYZZY-42", "text/plain")},
                    headers={"Authorization": f"Bearer {tok}"})
        assert up.status_code == 200, up.text
        aid = up.json()["id"]
        events = []
        with c.websocket_connect(f"/ws?token={tok}") as ws:
            ws.send_text(json.dumps({"action": "run_task", "task": "summarize the attachment",
                                     "attachments": [aid]}))
            for _ in range(30):
                ev = ws.receive_json()
                events.append(ev)
                if ev.get("event") == "task_result":
                    break
    # the agent read the relpath from the system prompt and called the gated tool on it
    assert model.path_seen and model.path_seen.startswith("_ingest/")
    result = next((e for e in events if e.get("event") == "task_result"), None)
    assert result is not None and result["status"] == "completed"
    # the ingested file's CONTENT actually reached the model through the gated DocQA summarizer
    assert "MARKER-XYZZY-42" in serving.seen_text


# --------------------------------------------------------------------------- #
# each format round-trips through the composed store + the contained loader
# --------------------------------------------------------------------------- #
def test_stage5_each_format_ingests_and_reads(tmp_path):
    s = _settings(tmp_path)
    application, api = _app(s)
    store = application.ingest_store
    cases = [
        ("a.txt", b"PLAIN-MARKER", "PLAIN-MARKER"),
        ("b.md", b"# MD-MARKER", "MD-MARKER"),
        ("c.html", b"<html><body><p>HTML-MARKER</p></body></html>", "HTML-MARKER"),
        ("d.hwpx", _hwpx("HWPX-MARKER"), "HWPX-MARKER"),
    ]
    with TestClient(api) as c:
        for name, data, marker in cases:
            r = c.post("/ingest", files={"file": (name, data, "application/octet-stream")}, headers=HDR)
            assert r.status_code == 200, f"{name}: {r.text}"
            aid = r.json()["id"]
            # the direct-chat read path (load_text → load_document → resolve_within) returns the content
            assert marker in store.load_text(aid), name


# --------------------------------------------------------------------------- #
# adversarial battery against the live endpoint (as an integrated whole)
# --------------------------------------------------------------------------- #
def test_stage5_adversarial_battery(tmp_path):
    s = _settings(tmp_path, ingest_max_file_bytes=64, ingest_max_files=3, ingest_max_total_bytes=120)
    application, api = _app(s)
    store = application.ingest_store
    with TestClient(api) as c:
        # oversized
        assert c.post("/ingest", files={"file": ("big.txt", b"x" * 200, "text/plain")}, headers=HDR).status_code == 413
        # wrong type + legacy .hwp
        assert c.post("/ingest", files={"file": ("m.exe", b"x", "application/octet-stream")}, headers=HDR).status_code == 415
        assert c.post("/ingest", files={"file": ("legacy.hwp", b"x", "application/octet-stream")}, headers=HDR).status_code == 415
        # traversal name → 200 but contained (basename only)
        tr = c.post("/ingest", files={"file": ("../../etc/passwd.txt", b"x", "text/plain")}, headers=HDR)
        assert tr.status_code == 200
        stored = store.resolve(tr.json()["id"])
        assert stored.startswith("_ingest/") and "passwd.txt" in stored and ".." not in stored
        # a malformed .hwpx UPLOADS (bytes are just stored — no parse at upload) but the LOADER's typed-
        # error discipline surfaces at READ time: load_text raises a DocError, never a raw exception/crash.
        # (The loader's decompression/zip-bomb bound itself is proven in 's verifying-code.)
        bad = c.post("/ingest", files={"file": ("corrupt.hwpx", b"not a zip at all", "application/octet-stream")},
                     headers=HDR)
        assert bad.status_code == 200 # storing bytes is fine; parsing happens only on read
        import pytest
        from local_ai_agent.modules.docqa.loaders import DocError
        with pytest.raises(DocError):
            store.load_text(bad.json()["id"])
        # count cap: we already stored 2 (traversal + corrupt); a 3rd ok, a 4th → 429
        assert c.post("/ingest", files={"file": ("e.txt", b"x", "text/plain")}, headers=HDR).status_code == 200
        assert c.post("/ingest", files={"file": ("f.txt", b"x", "text/plain")}, headers=HDR).status_code == 429


def test_stage5_unknown_attachment_id_is_rejected(tmp_path):
    s = _settings(tmp_path)
    application, _ = _app(s)
    from local_ai_agent.modules.docqa.ingest import UnknownIngestId
    import pytest
    with pytest.raises(UnknownIngestId):
        application.ingest_store.resolve("not-a-real-id")
