"""Map-reduce document summarizer.

The first DocQA step that calls the model. Chunk the text, summarize each chunk
(**map**), then combine the partial summaries (**reduce**) — hierarchically, packing
summaries into context-budget groups, until a single summary remains. A one-chunk document
takes a fast path (one map call, no reduce).

Bounded by design (aligns with INV-5 when driven by the agent loop):
  * per-call output capped via `max_summary_tokens` (keeps reduce inputs from blowing up);
  * reduce is hard-capped at `reduce_max_passes` *and* each pass strictly shrinks the working
    set (or makes a single terminal combine) — no unbounded recursion;
  * a `` truncation (doc exceeded `max_chunks`) is propagated as
    `SummaryResult.truncated`, never silently dropped.

The model is reached only through an injected `ChatModel` (the `LLMServingModule`
satisfies it); tests inject a deterministic fake — no real model needed. Every model/transport
failure maps to a typed `SummarizeError` (fail-fast); a not-ready engine maps to
`ServingUnavailable`. No raw library exception escapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..llm_serving import EngineNotReady
from .chunker import chunk_text

_JOIN = "\n\n"

_SYSTEM_PROMPT = (
    "You are a precise summarizer. Summarize the user's text faithfully and concisely, "
    "in the same language as the text. Do not add information that is not present."
)
_MAP_USER = "Summarize the following document section:\n\n{text}"
_REDUCE_USER = "Combine the following partial summaries into one coherent summary:\n\n{text}"


class ChatModel(Protocol):
    """Minimal seam over `LLMServingModule.chat` — inject a fake in tests."""

    async def chat(self, messages: list[dict], **params) -> dict: ...


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    truncated: bool # True iff capped the doc at max_chunks (partial summary)
    chunk_count: int
    reduce_passes: int


# --------------------------------------------------------------------------- #
# errors — every failure is a typed SummarizeError (never a raw exception)
# --------------------------------------------------------------------------- #
class SummarizeError(Exception):
    """Base for summarization failures."""


class ServingUnavailable(SummarizeError):
    """The serving engine is not ready (maps `EngineNotReady`) → 'capability unavailable'."""


# --------------------------------------------------------------------------- #
# response handling
# --------------------------------------------------------------------------- #
def _extract_content(response: object) -> str:
    """Pull `choices[0].message.content` with per-level guards (mirrors the orchestrator
    adapter). A non-string/missing content → `SummarizeError`. An empty string is valid."""
    if not isinstance(response, dict):
        raise SummarizeError("malformed response: not a dict")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SummarizeError("malformed response: no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise SummarizeError("malformed response: bad choice")
    message = first.get("message")
    if not isinstance(message, dict):
        raise SummarizeError("malformed response: no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise SummarizeError("malformed response: content is not a string")
    return content


async def _summarize_one(chat: ChatModel, text: str, *, max_summary_tokens: int, reduce: bool) -> str:
    """One model call. Any chat failure → fail-fast typed error (never a raw exception)."""
    template = _REDUCE_USER if reduce else _MAP_USER
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": template.format(text=text)},
    ]
    try:
        response = await chat.chat(messages, max_tokens=max_summary_tokens)
    except EngineNotReady as exc:
        raise ServingUnavailable(str(exc) or "serving engine not ready") from exc
    except SummarizeError:
        raise
    except Exception as exc: # noqa: BLE001 — transport/engine failure → fail-fast typed error
        raise SummarizeError(f"chat call failed: {type(exc).__name__}") from exc
    return _extract_content(response)


def _pack(items: list[str], budget: int) -> list[list[str]]:
    """Greedily group `items` so each group's joined length stays within `budget` chars.
    An item larger than `budget` becomes its own (oversize) group."""
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for it in items:
        add = len(it) + (len(_JOIN) if cur else 0)
        if cur and cur_len + add > budget:
            groups.append(cur)
            cur, cur_len = [it], len(it)
        else:
            cur.append(it)
            cur_len += add
    if cur:
        groups.append(cur)
    return groups


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
async def summarize_text(
    text: str,
    *,
    chat: ChatModel,
    chunk_chars: int,
    overlap: int = 0,
    max_chunks: int,
    max_summary_tokens: int = 512,
    reduce_max_passes: int = 5,
) -> SummaryResult:
    """Summarize `text` via map-reduce over context-sized chunks. Empty/whitespace text →
    empty summary. A truncated chunking is propagated. Invalid params → `ValueError`; any
    model/transport failure → `SummarizeError` (fail-fast)."""
    if max_summary_tokens <= 0:
        raise ValueError("max_summary_tokens must be > 0")
    if reduce_max_passes <= 0:
        raise ValueError("reduce_max_passes must be > 0")

    chunked = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap, max_chunks=max_chunks)
    chunks = chunked.chunks
    if not chunks:
        return SummaryResult(summary="", truncated=chunked.truncated, chunk_count=0, reduce_passes=0)

    # map: one summary per chunk
    summaries = [
        await _summarize_one(chat, c.text, max_summary_tokens=max_summary_tokens, reduce=False)
        for c in chunks
    ]

    # 1-chunk fast path: the single map result IS the summary (no reduce)
    if len(summaries) == 1:
        return SummaryResult(summary=summaries[0], truncated=chunked.truncated, chunk_count=1, reduce_passes=0)

    # reduce: pack into budget-sized groups, summarize each, repeat until one remains
    current = summaries
    passes = 0
    while len(current) > 1:
        if passes >= reduce_max_passes:
            raise SummarizeError(f"reduce did not converge within {reduce_max_passes} passes")
        groups = _pack(current, chunk_chars)
        if len(groups) >= len(current):
            # packing can't shrink the set (each summary already exceeds the budget) → one
            # terminal combine of everything, then stop. Guarantees termination.
            combined = _JOIN.join(current)
            current = [await _summarize_one(chat, combined, max_summary_tokens=max_summary_tokens, reduce=True)]
            passes += 1
            break
        current = [
            await _summarize_one(chat, _JOIN.join(g), max_summary_tokens=max_summary_tokens, reduce=True)
            for g in groups
        ]
        passes += 1

    return SummaryResult(
        summary=current[0], truncated=chunked.truncated, chunk_count=len(chunks), reduce_passes=passes
    )
