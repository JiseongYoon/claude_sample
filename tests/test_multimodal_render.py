"""— bounded PDF→image rasterization + multimodal-gated image ingest allowlist.

The rasterizer is the ONE new server-side parser surface (UNTRUSTED PDF) → hard bounds (pages · per-page
pixels · total bytes · timeout). The ingest allowlist admits image types ONLY when a vision projector is
configured (multimodal); text-only serving → 415. PDFs are BUILT in-test with PyMuPDF (no committed binary).
Hermetic; conda `local-ai-agent-env-1` (needs the [multimodal] extra: pymupdf).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.core.module import Health, HealthStatus, ModuleSpec
from local_ai_agent.main import BuildOverrides, build_application
from local_ai_agent.modules.docqa.render import (
    IMAGE_EXTS,
    MultimodalUnavailable,
    RenderError,
    render_pdf_to_images,
)

fitz = pytest.importorskip("fitz") # the [multimodal] extra

API_KEY = "k" * 40
HDR = {"X-API-Key": API_KEY}
_BOUNDS = dict(max_pages=20, max_page_px=4_000_000, max_total_bytes=50_000_000, timeout_s=30.0)


def _pdf(n_pages=1, *, width=200, height=200, text="hello") -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=width, height=height)
        page.insert_text((20, 40), text)
    data = doc.tobytes()
    doc.close()
    return data


# --------------------------------------------------------------------------- #
# rasterizer — normal
# --------------------------------------------------------------------------- #
def test_render_returns_one_png_per_page():
    out = render_pdf_to_images(_pdf(2), **_BOUNDS)
    assert len(out) == 2
    assert all(b.startswith(b"\x89PNG") for b in out) # real PNGs, in order


def test_render_caps_page_count():
    out = render_pdf_to_images(_pdf(5), **{**_BOUNDS, "max_pages": 2})
    assert len(out) == 2 # only the first 2 pages, never all 5


def test_render_zooms_down_oversized_page():
    # a huge page must be zoomed down so the raster stays under the pixel cap (no giant bitmap)
    cap = 200_000
    out = render_pdf_to_images(_pdf(1, width=8000, height=8000), **{**_BOUNDS, "max_page_px": cap})
    assert len(out) == 1
    assert out[0].startswith(b"\x89PNG")
    # the DECODED raster stays under the pixel cap (PNG IHDR: width@16:20, height@20:24, big-endian)
    import struct

    w, h = struct.unpack(">II", out[0][16:24])
    assert w * h <= cap
    assert len(out[0]) < 2_000_000 # a small PNG, not a multi-MB full-res bitmap


# --------------------------------------------------------------------------- #
# rasterizer — error / security (bounds + bad input)
# --------------------------------------------------------------------------- #
def test_render_total_bytes_cap():
    with pytest.raises(RenderError):
        render_pdf_to_images(_pdf(5), **{**_BOUNDS, "max_total_bytes": 50})


def test_render_timeout():
    # a tiny timeout trips at the top of the loop after the first page renders
    with pytest.raises(RenderError):
        render_pdf_to_images(_pdf(4), **{**_BOUNDS, "timeout_s": 1e-6})


def test_render_not_a_pdf():
    with pytest.raises(RenderError):
        render_pdf_to_images(b"this is definitely not a pdf", **_BOUNDS)


def test_render_empty_bytes():
    with pytest.raises(RenderError):
        render_pdf_to_images(b"", **_BOUNDS)


def test_render_non_bytes():
    with pytest.raises(RenderError):
        render_pdf_to_images("not bytes", **_BOUNDS)


def test_render_encrypted_pdf():
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    doc.close()
    with pytest.raises(RenderError):
        render_pdf_to_images(data, **_BOUNDS)


def test_render_multimodal_unavailable(monkeypatch):
    # simulate the [multimodal] extra being absent → typed MultimodalUnavailable, not an import crash
    monkeypatch.setitem(sys.modules, "fitz", None)
    with pytest.raises(MultimodalUnavailable):
        render_pdf_to_images(b"%PDF-1.4", **_BOUNDS)


# --------------------------------------------------------------------------- #
# ingest allowlist — image types gated on multimodal availability
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


def _settings(tmp_path, **over):
    gguf = tmp_path / "gguf"
    gguf.mkdir(exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    base = dict(_env_file=None, model_safetensors_dir=str(gguf), model_gguf_dir=str(gguf),
                auth_enabled=True, api_key=API_KEY, jwt_secret="s" * 40,
                enable_docqa=True, docs_root=str(docs))
    base.update(over)
    return Settings(**base)


def _api(s):
    app = build_application(s, overrides=BuildOverrides(serving_module=FakeServing()))
    return create_gateway(app, s)


def test_image_exts_cover_common_types():
    assert {".png", ".jpg", ".jpeg", ".webp"} <= IMAGE_EXTS


def test_ingest_accepts_image_when_multimodal_configured(tmp_path):
    from fastapi.testclient import TestClient

    s = _settings(tmp_path, model_mmproj_file="proj.gguf")
    (Path(s.model_gguf_dir) / "proj.gguf").write_bytes(b"\x00")
    with TestClient(_api(s)) as c:
        r = c.post("/ingest", files={"file": ("chart.png", b"\x89PNG\r\n\x1a\nfake", "image/png")}, headers=HDR)
        assert r.status_code == 200, r.text
        assert r.json()["ext"] == ".png"


def test_ingest_rejects_image_when_text_only(tmp_path):
    from fastapi.testclient import TestClient

    s = _settings(tmp_path) # no model_mmproj_file → text-only → images not in the allowlist
    with TestClient(_api(s)) as c:
        r = c.post("/ingest", files={"file": ("chart.png", b"\x89PNG", "image/png")}, headers=HDR)
        assert r.status_code == 415
