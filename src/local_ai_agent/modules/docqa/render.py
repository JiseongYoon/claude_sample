"""Bounded PDF→image rasterization for the visual-multimodal path.

A scan-PDF (image-based, no text layer) is rendered to page PNGs so the model can SEE it. The PDF is
UNTRUSTED, so this is the one new server-side parser surface in — it is **hard-bounded**:
- **page count** — at most `max_pages` pages are rendered;
- **per-page pixels** — each page is zoomed DOWN so its raster stays under `max_page_px` (bounds the
  render cost of a page that declares a huge MediaBox), with a post-render belt-and-braces check;
- **total bytes** — the cumulative PNG size is capped (`max_total_bytes`);
- **timeout** — a wall-clock backstop across pages.
A breach / encrypted / malformed / not-a-PDF input raises a typed `RenderError`, never an OOM/hang/crash.

PyMuPDF (`fitz`) is lazy-imported behind the `[multimodal]` extra; absent → `MultimodalUnavailable`
(the visual path degrades, never an import crash). Raster IMAGE files (png/jpg/webp) are NOT decoded here
— they pass through as base64 (no image-parser surface on our side); only PDFs are rasterized.
"""
from __future__ import annotations

import time

# raster image types the multimodal path accepts as direct vision input (no server-side decode).
IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


class RenderError(Exception):
    """A PDF could not be rasterized within the bounds (typed, generic — no path/stack leak)."""


class MultimodalUnavailable(RenderError):
    """The optional rasterization dependency (PyMuPDF / `[multimodal]` extra) is not installed."""


def render_pdf_to_images(
    data: bytes,
    *,
    max_pages: int,
    max_page_px: int,
    max_total_bytes: int,
    timeout_s: float,
) -> list[bytes]:
    """Render an UNTRUSTED PDF (`data`) to a list of PNG byte blobs (one per page, in order),
    hard-bounded. Raises `RenderError` (or `MultimodalUnavailable`) on any breach / bad input."""
    try:
        import fitz # PyMuPDF
    except ImportError as exc:
        raise MultimodalUnavailable("pdf rasterization requires the [multimodal] extra (pymupdf)") from exc
    if not isinstance(data, (bytes, bytearray)):
        raise RenderError("data must be bytes")
    if max_pages <= 0 or max_page_px <= 0 or max_total_bytes <= 0 or timeout_s <= 0:
        raise RenderError("invalid render bounds")
    start = time.monotonic()
    try:
        doc = fitz.open(stream=bytes(data), filetype="pdf")
    except Exception as exc: # noqa: BLE001 — any open failure → typed, never raw
        raise RenderError("could not open PDF") from exc
    try:
        if getattr(doc, "needs_pass", False):
            raise RenderError("encrypted PDF not supported")
        page_count = doc.page_count
        if page_count <= 0:
            raise RenderError("PDF has no pages")
        out: list[bytes] = []
        total = 0
        for i in range(min(page_count, max_pages)):
            if time.monotonic() - start > timeout_s:
                raise RenderError("PDF render timed out")
            try:
                page = doc.load_page(i)
                rect = page.rect
                base_px = max(1.0, float(rect.width) * float(rect.height))
                # zoom DOWN so the rasterized pixel count stays under the cap (sqrt scales both axes).
                # A 0.95 safety margin absorbs the pixmap's round-up so the post-render hard-cap never
                # trips on a rounding overshoot — only on a gross miscalculation.
                zoom = 1.0 if base_px <= max_page_px else (max_page_px * 0.95 / base_px) ** 0.5
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            except RenderError:
                raise
            except Exception as exc: # noqa: BLE001
                raise RenderError(f"could not render page {i}") from exc
            if pix.width * pix.height > max_page_px: # belt-and-braces post-render bound
                raise RenderError("rasterized page exceeds the pixel cap")
            png = pix.tobytes("png")
            total += len(png)
            if total > max_total_bytes:
                raise RenderError("rasterized PDF exceeds the total-byte cap")
            out.append(png)
        return out
    finally:
        try:
            doc.close()
        except Exception: # noqa: BLE001 — close must never mask the result/raise
            pass
