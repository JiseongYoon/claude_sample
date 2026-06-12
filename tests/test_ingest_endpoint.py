"""— the gated upload endpoint `POST /ingest`.

Drives the REAL composition root (`build_application`) with a fake serving leaf so DocQA +
the ingest store are wired without a live model. Covers: happy path (id returned, file written
under `_ingest/`), the `ingest` scope gate, the count/total/per-file caps, wrong-type/bad-name
rejections (typed, no-leak), and 503 when ingestion is absent. Hermetic; conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.auth import create_access_token
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application

API_KEY = "k" * 40
JWT = "s" * 40
HDR = {"X-API-Key": API_KEY} # api-key principal has the `*` scope → covers `ingest`


class FakeServing:
    """llm-serving stand-in (Module + ChatModel) so DocQA's dep resolves without a real model."""

    def __init__(self) -> None:
        self._s = False

    @property
    def spec(self) -> ModuleSpec:
        return ModuleSpec(name="llm-serving", version="0", capabilities=("chat",), depends_on=())

    async def start(self) -> None:
        self._s = True

    async def stop(self) -> None:
        self._s = False

    def health(self) -> Health:
        return Health(HealthStatus.ok if self._s else HealthStatus.absent, "fake")

    async def chat(self, messages, **params) -> dict:
        return {"choices": [{"message": {"content": "x"}}]}


def _settings(tmp_path, **over) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(
        _env_file=None, model_safetensors_dir=str(tmp_path), model_gguf_dir=str(tmp_path),
        auth_enabled=True, api_key=API_KEY, jwt_secret=JWT,
        enable_docqa=True, docs_root=str(docs),
    )
    base.update(over)
    return Settings(**base)


def _app(s):
    application = build_application(s, overrides=BuildOverrides(serving_module=FakeServing()))
    return create_gateway(application, s)


def _token(s, scopes) -> dict:
    return {"Authorization": f"Bearer {create_access_token(s, 'client', scopes)}"}


def _ingest_root(s):
    from pathlib import Path
    return Path(s.docs_root) / "_ingest"


# --------------------------------------------------------------------------- #
# normal
# --------------------------------------------------------------------------- #
def test_upload_returns_id_and_writes_file(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("report.txt", b"hello world", "text/plain")}, headers=HDR)
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"id", "filename", "size", "ext"}
        assert body["filename"] == "report.txt" and body["ext"] == ".txt" and body["size"] == 11
        # no server path in the response
        assert "/" not in str(body) and "\\" not in str(body)
        # the file exists under _ingest/<id>/
        stored = _ingest_root(s) / body["id"] / "report.txt"
        assert stored.is_file() and stored.read_bytes() == b"hello world"


def test_two_uploads_distinct_ids(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        a = c.post("/ingest", files={"file": ("a.txt", b"aaa", "text/plain")}, headers=HDR).json()
        b = c.post("/ingest", files={"file": ("a.txt", b"bbbb", "text/plain")}, headers=HDR).json()
        assert a["id"] != b["id"]


def test_api_key_scope_star_allows_ingest(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        assert c.post("/ingest", files={"file": ("x.md", b"# hi", "text/markdown")},
                      headers=HDR).status_code == 200


def test_minted_token_with_ingest_scope_allows(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        hdr = _token(s, ["read", "ingest"])
        assert c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")},
                      headers=hdr).status_code == 200


# --------------------------------------------------------------------------- #
# error / security
# --------------------------------------------------------------------------- #
def test_missing_ingest_scope_rejected(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        hdr = _token(s, ["read", "invoke", "agent:run"]) # no `ingest`
        r = c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")}, headers=hdr)
        assert r.status_code == 403


def test_unauthenticated_rejected(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")})
        assert r.status_code in (401, 403)


def test_oversized_rejected_413(tmp_path):
    s = _settings(tmp_path, ingest_max_file_bytes=10)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("big.txt", b"x" * 50, "text/plain")}, headers=HDR)
        assert r.status_code == 413
        # nothing written
        assert not any(_ingest_root(s).glob("*/*")) if _ingest_root(s).exists() else True


def test_wrong_type_rejected_415(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        for name in ("malware.exe", "legacy.hwp", "archive.zip"):
            r = c.post("/ingest", files={"file": (name, b"data", "application/octet-stream")}, headers=HDR)
            assert r.status_code == 415, name
            assert name not in r.text # no filename echo


def test_bad_filename_rejected_400(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        # a traversal name whose basename keeps a supported ext is stripped to a safe name (200);
        # a dotfile/empty-after-strip name is rejected 400. Use a dotfile to force 400.
        r = c.post("/ingest", files={"file": (".secret.txt", b"x", "text/plain")}, headers=HDR)
        assert r.status_code == 400
        assert "secret" not in r.text # generic detail, no echo


def test_traversal_filename_is_contained(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("../../etc/passwd.txt", b"x", "text/plain")}, headers=HDR)
        assert r.status_code == 200 # stripped to a safe basename
        stored = _ingest_root(s) / r.json()["id"] / "passwd.txt"
        assert stored.is_file()
        assert stored.resolve().is_relative_to(_ingest_root(s).resolve()) # never escaped


def test_count_cap_429(tmp_path):
    s = _settings(tmp_path, ingest_max_files=1)
    with TestClient(_app(s)) as c:
        assert c.post("/ingest", files={"file": ("a.txt", b"a", "text/plain")}, headers=HDR).status_code == 200
        r = c.post("/ingest", files={"file": ("b.txt", b"b", "text/plain")}, headers=HDR)
        assert r.status_code == 429


def test_total_byte_cap_413(tmp_path):
    s = _settings(tmp_path, ingest_max_file_bytes=1000, ingest_max_total_bytes=10)
    with TestClient(_app(s)) as c:
        assert c.post("/ingest", files={"file": ("a.txt", b"x" * 6, "text/plain")}, headers=HDR).status_code == 200
        r = c.post("/ingest", files={"file": ("b.txt", b"x" * 6, "text/plain")}, headers=HDR) # 12 > 10
        assert r.status_code == 413


def test_503_when_ingestion_absent(tmp_path):
    # docqa off → no ingest store → 503
    s = _settings(tmp_path, enable_docqa=False, docs_root=None)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("x.txt", b"hi", "text/plain")}, headers=HDR)
        assert r.status_code == 503


def test_empty_filename_rejected(tmp_path):
    s = _settings(tmp_path)
    with TestClient(_app(s)) as c:
        r = c.post("/ingest", files={"file": ("", b"hi", "text/plain")}, headers=HDR)
        assert r.status_code in (400, 415, 422) # empty/▽no-ext → rejected, never written
