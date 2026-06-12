"""Browser capability config.

A small frozen view over the main `Settings`, consumed by the browser module/tools (mirrors docqa's
`DocQAConfig.from_settings`). The SearXNG endpoint is operator config (a trusted host) — the agent
controls only the *query*, never the search host, so there is no per-query SSRF surface there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING: # avoid an import cycle / coupling at runtime
    from ...config import Settings


@dataclass(frozen=True)
class BrowserConfig:
    """Resolved browser settings. `allow_hosts` empty → blocklist mode (any public host allowed);
    non-empty → strict allowlist (only those exact hosts)."""

    searxng_url: str | None
    allow_hosts: tuple[str, ...]
    max_bytes: int
    per_fetch_timeout: float
    total_timeout: float
    max_redirects: int
    top_n: int
    max_text_chars: int
    # `web_answer` synthesis bounds — reuse the docqa QA settings (present on Settings
    # regardless of `enable_docqa`); defaulted so direct construction in tests stays terse.
    qa_top_k: int = 5
    qa_max_context_chars: int = 12000
    qa_answer_max_tokens: int = 512
    qa_chunk_chars: int = 2000
    qa_max_chunks: int = 128

    @classmethod
    def from_settings(cls, settings: "Settings") -> "BrowserConfig":
        return cls(
            searxng_url=settings.browser_searxng_url,
            allow_hosts=tuple(settings.browser_allow_hosts or ()),
            max_bytes=settings.browser_max_bytes,
            per_fetch_timeout=settings.browser_per_fetch_timeout,
            total_timeout=settings.browser_total_timeout,
            max_redirects=settings.browser_max_redirects,
            top_n=settings.browser_top_n,
            max_text_chars=settings.browser_max_text_chars,
            qa_top_k=settings.docqa_qa_top_k,
            qa_max_context_chars=settings.docqa_qa_max_context_chars,
            qa_answer_max_tokens=settings.docqa_answer_max_tokens,
            qa_chunk_chars=settings.docqa_qa_chunk_chars,
            qa_max_chunks=settings.docqa_max_chunks,
        )
