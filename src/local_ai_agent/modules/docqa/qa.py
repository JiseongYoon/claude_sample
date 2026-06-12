"""Naive-retrieval question answering over local document chunks.

Answer a question grounded **only** in retrieved chunks, with source citations. Retrieval is
keyword-based (no embeddings) — a deterministic seam that RAG (sub-phase 4.1) can replace.

Pure core (no filesystem I/O): it operates on a pre-built `SourcedChunk` list. The glue that
loads documents and chunks them into `SourcedChunk`s lives in 's tool wiring (alongside the
gate). `sourced_chunks` is a pure helper that labels a `ChunkResult` with its source.

Strict grounding:
  * zero retrieval (no chunk shares any query term) → `answer_found=False` with **no model call**;
  * otherwise the model is told to reply exactly `NO_ANSWER` when the context lacks the answer →
    `answer_found=False`. Citations are reported only for a found answer.

The model is reached only through the `ChatModel` seam (fake in tests). Every model/
transport/parse failure maps to a typed `QAError` (fail-fast); a not-ready engine →
`ServingUnavailable`. No raw library exception escapes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm_serving import EngineNotReady
from .chunker import ChunkResult
from .summarizer import ChatModel

_NO_ANSWER = "NO_ANSWER"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_SYSTEM_PROMPT = (
    "You answer the user's question using ONLY the provided context. Each context block is "
    "tagged with its source as [source#index]. If the answer is not contained in the context, "
    f"reply with exactly {_NO_ANSWER} and nothing else. Answer in the question's language; do "
    "not invent information beyond the context."
)
_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


@dataclass(frozen=True)
class SourcedChunk:
    """A chunk labelled with the document it came from. `text == source_doc[start:end]`."""

    source: str # document identifier (e.g. root-relative path)
    index: int # chunk index within that document
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Citation:
    source: str
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class QAResult:
    answer: str
    answer_found: bool
    citations: tuple[Citation, ...]


# --------------------------------------------------------------------------- #
# errors — typed, never raw
# --------------------------------------------------------------------------- #
class QAError(Exception):
    """Base for question-answering failures."""


class ServingUnavailable(QAError):
    """The serving engine is not ready (maps `EngineNotReady`) → 'capability unavailable'."""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def sourced_chunks(source: str, result: ChunkResult) -> list[SourcedChunk]:
    """Label a `ChunkResult` with its source. Pure — no I/O."""
    return [SourcedChunk(source, c.index, c.text, c.start, c.end) for c in result.chunks]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def select_chunks(
    question: str, chunks: list[SourcedChunk], *, top_k: int, max_context_chars: int
) -> list[SourcedChunk]:
    """Rank chunks by keyword overlap with `question` and return up to `top_k` that fit within
    `max_context_chars` (the top-ranked chunk is always included, even if oversize). Returns
    `[]` when nothing matches. Fully deterministic (tie-break: distinct-terms desc, matched
    frequency desc, source asc, original index asc)."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be > 0")
    q_terms = set(_tokenize(question))
    if not q_terms:
        return []
    scored = []
    for ci, c in enumerate(chunks):
        toks = _tokenize(c.text)
        matched = q_terms.intersection(toks)
        if not matched:
            continue
        distinct = len(matched)
        freq = sum(toks.count(t) for t in matched)
        scored.append((-distinct, -freq, c.source, c.index, ci, c))
    scored.sort(key=lambda x: x[:5])
    selected: list[SourcedChunk] = []
    used = 0
    for *_key, c in scored:
        if len(selected) >= top_k:
            break
        add = len(c.text)
        if selected and used + add > max_context_chars:
            continue # skip; a smaller lower-ranked chunk may still fit
        selected.append(c) # first (top-ranked) chunk is always included
        used += add
    return selected


# --------------------------------------------------------------------------- #
# response handling
# --------------------------------------------------------------------------- #
def _extract_content(response: object) -> str:
    """Pull `choices[0].message.content` with per-level guards. Malformed → `QAError`."""
    if not isinstance(response, dict):
        raise QAError("malformed response: not a dict")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise QAError("malformed response: no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise QAError("malformed response: bad choice")
    message = first.get("message")
    if not isinstance(message, dict):
        raise QAError("malformed response: no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise QAError("malformed response: content is not a string")
    return content


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
async def answer_question(
    question: str,
    chunks: list[SourcedChunk],
    *,
    chat: ChatModel,
    top_k: int = 5,
    max_context_chars: int = 12000,
    max_answer_tokens: int = 512,
) -> QAResult:
    """Answer `question` grounded in retrieved `chunks`, with citations. Empty question →
    `ValueError`. No retrieval → `answer_found=False` (no model call). A `NO_ANSWER` reply →
    `answer_found=False`. Any model/transport failure → `QAError` (fail-fast)."""
    if not question or not question.strip():
        raise ValueError("question must be non-empty")
    if max_answer_tokens <= 0:
        raise ValueError("max_answer_tokens must be > 0")

    selected = select_chunks(question, chunks, top_k=top_k, max_context_chars=max_context_chars)
    if not selected:
        return QAResult(answer="", answer_found=False, citations=())

    context = "\n\n".join(f"[{c.source}#{c.index}]\n{c.text}" for c in selected)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(context=context, question=question)},
    ]
    try:
        response = await chat.chat(messages, max_tokens=max_answer_tokens)
    except EngineNotReady as exc:
        raise ServingUnavailable(str(exc) or "serving engine not ready") from exc
    except QAError:
        raise
    except Exception as exc: # noqa: BLE001 — transport/engine failure → fail-fast typed error
        raise QAError(f"chat call failed: {type(exc).__name__}") from exc

    answer = _extract_content(response)
    if answer.strip() == _NO_ANSWER:
        return QAResult(answer="", answer_found=False, citations=())
    citations = tuple(Citation(c.source, c.index, c.start, c.end) for c in selected)
    return QAResult(answer=answer, answer_found=True, citations=citations)
