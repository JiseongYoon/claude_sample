"""Text chunker + token budgeting.

Split the plain text produced by `loaders.load_document` into model-context-sized
chunks for the summarizer and QA. Pure, deterministic, dependency-free.

Two separated concerns:

  * **token budgeting** — `derive_chunk_chars` / `derive_overlap` translate a token
    budget (`ctx_size` minus a prompt reserve) into a *character* budget using a
    `chars_per_token` heuristic (no `tiktoken`; char-based is deterministic and
    language-agnostic). A conservative ratio keeps chunks safely inside the context.
  * **splitting** — `chunk_text` cuts the text into ≤ `chunk_chars` windows, preferring
    natural boundaries (paragraph → line → whitespace) and falling back to a hard cut,
    with `overlap` carried between consecutive chunks and a hard `max_chunks` cap.

Invariants the splitter guarantees for *every* input:
  * **termination** — forward progress is guaranteed (no infinite loop), even for text
    with no boundaries, a single mega-token, or pathological whitespace.
  * **offset fidelity** — `chunk.text == source[chunk.start:chunk.end]` exactly, so
     can cite a chunk back to the source.
  * **bounded** — at most `max_chunks` chunks; if the text would need more, the tail is
    dropped and `ChunkResult.truncated` is set True (signalled, never silent).
  * **determinism** — identical input + params → identical output.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One contiguous slice of the source text. `text == source[start:end]`."""

    index: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ChunkResult:
    """Result of chunking. `truncated` is True iff the `max_chunks` cap dropped tail text."""

    chunks: tuple[Chunk, ...]
    truncated: bool
    total_chars: int


# --------------------------------------------------------------------------- #
# token budgeting (token-budget → char-budget heuristic)
# --------------------------------------------------------------------------- #
def derive_chunk_chars(
    ctx_size: int, *, prompt_reserve_tokens: int, chars_per_token: float
) -> int:
    """Characters per chunk = (ctx_size − prompt_reserve_tokens) × chars_per_token.

    `prompt_reserve_tokens` holds back room for the system/instruction prompt and the
    model's own output. A misconfiguration that leaves no content budget (reserve ≥
    ctx_size, or a degenerate ratio) raises `ValueError` rather than yielding a useless
    zero-width chunk size."""
    if ctx_size <= 0:
        raise ValueError("ctx_size must be > 0")
    if prompt_reserve_tokens < 0:
        raise ValueError("prompt_reserve_tokens must be >= 0")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    budget_tokens = ctx_size - prompt_reserve_tokens
    if budget_tokens <= 0:
        raise ValueError("prompt_reserve_tokens leaves no room for content (>= ctx_size)")
    chars = int(budget_tokens * chars_per_token)
    if chars <= 0:
        raise ValueError("derived chunk size is zero (chars_per_token too small)")
    return chars


def derive_overlap(chunk_chars: int, *, ratio: float) -> int:
    """Overlap chars = floor(chunk_chars × ratio). `ratio` in [0, 1) guarantees
    overlap < chunk_chars (a prerequisite for forward progress in `chunk_text`)."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if not 0 <= ratio < 1:
        raise ValueError("ratio must be in [0, 1)")
    return int(chunk_chars * ratio)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def _find_break(text: str, pos: int, end: int, chunk_chars: int) -> int:
    """Pick a cut point in (pos, end] preferring a natural boundary. Searches the tail
    window [min_fill, end) so chunks are not cut absurdly early, preferring paragraph
    (`\\n\\n`) → line (`\\n`) → whitespace; falls back to a hard cut at `end`. The
    returned point is always > pos and <= end (forward progress + within budget)."""
    min_fill = pos + max(1, chunk_chars // 2)
    if min_fill >= end: # tiny budget: don't force an early boundary
        return end
    window = text[min_fill:end]
    i = window.rfind("\n\n")
    if i != -1:
        return min_fill + i + 2
    i = window.rfind("\n")
    if i != -1:
        return min_fill + i + 1
    i = window.rfind(" ")
    if i != -1:
        return min_fill + i + 1
    i = window.rfind("\t")
    if i != -1:
        return min_fill + i + 1
    return end # no boundary in window → hard cut


def chunk_text(
    text: str, *, chunk_chars: int, overlap: int = 0, max_chunks: int
) -> ChunkResult:
    """Split `text` into ≤ `chunk_chars`-char chunks with `overlap` carried between
    consecutive chunks, capped at `max_chunks`.

    Empty/whitespace-only text → zero chunks (`truncated=False`). Invalid params raise
    `ValueError`. Forward progress is guaranteed for every input."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_chars:
        raise ValueError("overlap must be < chunk_chars (else no forward progress)")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be > 0")

    n = len(text)
    if not text.strip():
        return ChunkResult(chunks=(), truncated=False, total_chars=n)

    chunks: list[Chunk] = []
    pos = 0
    truncated = False
    while pos < n:
        if len(chunks) >= max_chunks: # cap reached with text remaining → truncated
            truncated = True
            break
        end = pos + chunk_chars
        chunk_end = n if end >= n else _find_break(text, pos, end, chunk_chars)
        chunks.append(Chunk(index=len(chunks), text=text[pos:chunk_end], start=pos, end=chunk_end))
        if chunk_end >= n:
            break
        next_pos = chunk_end - overlap
        if next_pos <= pos: # overlap would not advance → drop overlap this step
            next_pos = chunk_end
        pos = next_pos

    return ChunkResult(chunks=tuple(chunks), truncated=truncated, total_chars=n)
