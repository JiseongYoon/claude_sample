"""Multimodal DocQA — the gated `answer_about_image` tool.

Answers a question over an ingested IMAGE or scan-PDF by sending it to the vision-capable model as OpenAI
vision content (`image_url` data-URLs). Registered + reachable ONLY when the `multimodal` capability is
available (an mmproj projector is configured) — so even a crafted `run_task`/invoke can't reach a visual
path the model can't serve. Routes through the gate/dispatcher like every other tool (INV-1).

- image (png/jpg/webp) → ONE base64 vision part (no server-side decode — the bytes pass through).
- scan-PDF → `render_pdf_to_images` (BOUNDED) → one vision part per rendered page.
Reuses `IngestStore.read_bytes` (contained read) and `llm_serving.chat` (opaque passthrough → vision works
with no client change). Every failure → a typed error dict (graceful; no path/stack leak), mirroring the
QA tool.
"""
from __future__ import annotations

import base64
from typing import Any

from .render import IMAGE_EXTS, RenderError, render_pdf_to_images

# the read-only multimodal tool name → safe-listed on the gate (a read, like answer_question).
MULTIMODAL_SAFE_TOOL_NAMES = frozenset({"answer_about_image"})

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _answer_text(reply: Any) -> str:
    """Pull the assistant text out of an OpenAI-style chat reply (tolerant)."""
    if isinstance(reply, dict):
        choices = reply.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""


class AnswerAboutImageTool:
    """`answer_about_image` — vision QA over an ingested image/scan, gated behind `multimodal`."""

    name = "answer_about_image"
    description = ("Answer a question about an uploaded IMAGE or scanned PDF, identified by its "
                   "attachment id, using the vision model. Use this for images/scans (a text "
                   "document uses the DocQA tools by path instead).")
    parameters = {
        "type": "object",
        "properties": {
            "attachment_id": {"type": "string", "description": "The uploaded attachment's opaque id."},
            "question": {"type": "string", "description": "The question about the image/scan."},
        },
        "required": ["attachment_id", "question"],
    }

    def __init__(self, store, chat, *, max_pages: int, max_page_px: int,
                 max_render_bytes: int, render_timeout_s: float,
                 answer_max_tokens: int = 512) -> None:
        self._store = store # IngestStore (resolve + contained read_bytes)
        self._chat = chat # ChatModel (llm_serving) — opaque passthrough → vision content
        self._max_pages = max_pages
        self._max_page_px = max_page_px
        self._max_render_bytes = max_render_bytes
        self._render_timeout_s = render_timeout_s
        self._answer_max_tokens = answer_max_tokens

    def _vision_parts(self, ext: str, data: bytes) -> list[dict]:
        if ext in IMAGE_EXTS:
            return [{"type": "image_url", "image_url": {"url": _data_url(_MIME[ext], data)}}]
        if ext == ".pdf":
            pages = render_pdf_to_images(
                data, max_pages=self._max_pages, max_page_px=self._max_page_px,
                max_total_bytes=self._max_render_bytes, timeout_s=self._render_timeout_s,
            )
            return [{"type": "image_url", "image_url": {"url": _data_url("image/png", p)}} for p in pages]
        raise RenderError(f"unsupported visual type: {ext!r}")

    async def run(self, args: dict) -> Any:
        question = args.get("question") if isinstance(args, dict) else None
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "ValueError: 'question' must be a non-empty string"}
        attachment_id = args.get("attachment_id")
        if not isinstance(attachment_id, str) or not attachment_id:
            return {"ok": False, "error": "ValueError: 'attachment_id' must be a non-empty string"}
        try:
            rec = self._store.get(attachment_id) # raises UnknownIngestId
            data = self._store.read_bytes(attachment_id) # contained, bounded
            parts = self._vision_parts(rec.ext, data) # image → 1 part; pdf → bounded pages
        except Exception as exc: # noqa: BLE001 — UnknownIngestId / DocError / RenderError → typed, no leak
            return {"ok": False, "error": f"{type(exc).__name__}: visual content unavailable"}
        if not parts:
            return {"ok": False, "error": "RenderError: no renderable content"}
        messages = [{"role": "user", "content": [{"type": "text", "text": question}, *parts]}]
        try:
            reply = await self._chat.chat(messages, max_tokens=self._answer_max_tokens)
        except Exception as exc: # noqa: BLE001 — serving down / transport error → graceful
            return {"ok": False, "error": f"{type(exc).__name__}: model unavailable"}
        return {"ok": True, "answer": _answer_text(reply), "pages": len(parts)}
