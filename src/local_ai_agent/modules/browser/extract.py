"""Main-content extraction.

Turns fetched HTML into clean title + body text via trafilatura (lazy-imported, so a missing dep
disables only extraction — never an import crash). Never raises a raw exception: a failed/empty
extraction yields an empty `ExtractResult`; a missing library yields a typed `BrowserUnavailable`.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fetcher import BrowserUnavailable


@dataclass(frozen=True)
class ExtractResult:
    """Extracted main content. `title`/`text` are empty strings when nothing could be extracted."""

    title: str
    text: str


def _charset_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip('"') or None
    return None


def _import_trafilatura():
    try:
        import trafilatura # noqa: PLC0415 — lazy: optional [browser] dep
    except Exception as exc: # noqa: BLE001
        raise BrowserUnavailable(f"extractor not available: {type(exc).__name__}") from exc
    return trafilatura


def extract_main_text(
    content: bytes | str,
    *,
    content_type: str | None = None,
    url: str | None = None,
) -> ExtractResult:
    """Extract main article text + title. `content` may be bytes (decoded via the `content_type`
    charset, else utf-8/replace) or str. Returns an empty `ExtractResult` when extraction yields
    nothing; raises `BrowserUnavailable` only if trafilatura is not installed."""
    if isinstance(content, bytes):
        enc = _charset_from_content_type(content_type) or "utf-8"
        try:
            text = content.decode(enc, errors="replace")
        except LookupError: # unknown charset name
            text = content.decode("utf-8", errors="replace")
    else:
        text = content or ""

    if not text.strip():
        return ExtractResult(title="", text="")

    trafilatura = _import_trafilatura()
    try:
        doc = trafilatura.bare_extraction(text, with_metadata=True, url=url)
    except Exception: # noqa: BLE001 — any parser hiccup degrades to "nothing extracted"
        return ExtractResult(title="", text="")

    if doc is None:
        return ExtractResult(title="", text="")

    # trafilatura 2.x returns a Document (attrs); older/dict forms tolerated.
    if isinstance(doc, dict):
        title = doc.get("title") or ""
        body = doc.get("text") or ""
    else:
        title = getattr(doc, "title", "") or ""
        body = getattr(doc, "text", "") or ""
    return ExtractResult(title=str(title).strip(), text=str(body).strip())
