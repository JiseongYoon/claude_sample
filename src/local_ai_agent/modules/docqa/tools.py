"""DocQA gated tools + the load/chunk glue.

Three read-only `Tool` adapters the agent reaches through the dispatcher:
`list_documents`, `summarize_document`, `answer_question`. Each ties the earlier pure pieces
together — loaders → chunker → `sourced_chunks`/summarizer/qa (steps 3–4) —
behind the `docs_root` sandbox.

Containment is double: the gate ( wiring) classifies the path arg, AND every read here
goes through `resolve_within`. A tool NEVER raises for a normal failure: it catches its
own typed errors (`DocError`, `SummarizeError`, `QAError`, incl. their `ServingUnavailable`
subclasses) and returns `{"ok": False, "error": …}` so the agent gets a clean observation. Only
a truly unexpected exception propagates — and the dispatcher catches that too.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...config import Settings
from .chunker import chunk_text, derive_chunk_chars, derive_overlap
from .loaders import DocError, list_documents, load_document
from .qa import QAError, SourcedChunk, answer_question, sourced_chunks
from .summarizer import ChatModel, SummarizeError, summarize_text

# the typed-error families a tool turns into a graceful structured result
_GRACEFUL = (DocError, SummarizeError, QAError)


@dataclass(frozen=True)
class DocQAConfig:
    """Concrete params the tools need — decoupled from `Settings` for easy testing.

    `summarize_chunk_chars` is the (large) per-map context budget; `qa_chunk_chars` is smaller
    so several chunks fit the QA retrieval budget."""

    docs_root: Path
    max_doc_bytes: int
    summarize_chunk_chars: int
    qa_chunk_chars: int
    overlap_ratio: float
    max_chunks: int
    summary_max_tokens: int
    reduce_max_passes: int
    qa_top_k: int
    qa_max_context_chars: int
    answer_max_tokens: int
    max_docs_per_query: int
    max_total_chunks: int

    @classmethod
    def from_settings(cls, s: Settings) -> "DocQAConfig":
        if s.docs_root is None:
            raise ValueError("docqa enabled but docs_root is not configured")
        return cls(
            docs_root=Path(s.docs_root),
            max_doc_bytes=s.docqa_max_doc_bytes,
            summarize_chunk_chars=derive_chunk_chars(
                s.ctx_size, prompt_reserve_tokens=s.docqa_prompt_reserve_tokens,
                chars_per_token=s.docqa_chars_per_token),
            qa_chunk_chars=s.docqa_qa_chunk_chars,
            overlap_ratio=s.docqa_chunk_overlap_ratio,
            max_chunks=s.docqa_max_chunks,
            summary_max_tokens=s.docqa_summary_max_tokens,
            reduce_max_passes=s.docqa_reduce_max_passes,
            qa_top_k=s.docqa_qa_top_k,
            qa_max_context_chars=s.docqa_qa_max_context_chars,
            answer_max_tokens=s.docqa_answer_max_tokens,
            max_docs_per_query=s.docqa_max_docs_per_query,
            max_total_chunks=s.docqa_max_total_chunks,
        )


def _err(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# load + chunk glue
# --------------------------------------------------------------------------- #
def gather_sourced_chunks(
    root: Path,
    *,
    path: str | None = None,
    subdir: str | None = None,
    recursive: bool = True,
    chunk_chars: int,
    overlap: int,
    max_chunks: int,
    max_bytes: int,
    max_docs: int,
    max_total_chunks: int,
) -> tuple[list[SourcedChunk], bool]:
    """Load + chunk a single document (`path`) or every supported doc under `subdir`, returning
    `(chunks, truncated)`. Each chunk is labelled by its root-relative source. Exactly one of
    `path`/`subdir` must be given. Bounded by `max_docs` / `max_total_chunks` (sets `truncated`).
    Reads are contained to `root`. Raises typed `DocError` on a bad read; `ValueError`
    on a bad path/subdir combination (a caller bug — tools validate args first)."""
    if (path is None) == (subdir is None):
        raise ValueError("exactly one of path/subdir is required")

    if path is not None:
        text = load_document(root, path, max_bytes=max_bytes)
        result = chunk_text(text, chunk_chars=chunk_chars, overlap=overlap, max_chunks=max_chunks)
        return sourced_chunks(path, result), result.truncated

    docs = list_documents(root, subdir, recursive=recursive)
    truncated = False
    if len(docs) > max_docs:
        docs, truncated = docs[:max_docs], True
    out: list[SourcedChunk] = []
    for rel in docs:
        result = chunk_text(
            load_document(root, rel, max_bytes=max_bytes),
            chunk_chars=chunk_chars, overlap=overlap, max_chunks=max_chunks,
        )
        if result.truncated:
            truncated = True
        for sc in sourced_chunks(rel, result):
            if len(out) >= max_total_chunks:
                return out, True
            out.append(sc)
    return out, truncated


def _resolve_target(args: dict) -> tuple[str | None, str | None] | dict:
    """Validate the `path`/`subdir` selector. Returns `(path, subdir)` or an error dict.
    Neither given → whole-root `subdir="."`; both given → error."""
    path = args.get("path")
    subdir = args.get("subdir")
    if path is not None and not isinstance(path, str):
        return {"ok": False, "error": "ValueError: path must be a string"}
    if subdir is not None and not isinstance(subdir, str):
        return {"ok": False, "error": "ValueError: subdir must be a string"}
    if path is not None and subdir is not None:
        return {"ok": False, "error": "ValueError: provide 'path' or 'subdir', not both"}
    if path is None and subdir is None:
        subdir = "."
    return path, subdir


# --------------------------------------------------------------------------- #
# tools — read-only; name matches the gate's (5b-extended) safe allowlist
# --------------------------------------------------------------------------- #
class ListDocumentsTool:
    name = "list_documents"
    description = ("List the supported documents under a subdirectory of the local document root "
                   "(root-relative path; '.' = the whole root).")
    parameters = {
        "type": "object",
        "properties": {
            "subdir": {"type": "string", "description": "Root-relative subdirectory ('.' = whole root).",
                       "default": "."},
            "recursive": {"type": "boolean", "description": "Recurse into nested folders.", "default": True},
        },
    }

    def __init__(self, config: DocQAConfig) -> None:
        self._cfg = config

    async def run(self, args: dict) -> Any:
        subdir = args.get("subdir", ".")
        if not isinstance(subdir, str):
            return {"ok": False, "error": "ValueError: subdir must be a string"}
        recursive = bool(args.get("recursive", True))
        try:
            docs = list_documents(self._cfg.docs_root, subdir, recursive=recursive)
        except _GRACEFUL as exc:
            return _err(exc)
        return {"ok": True, "documents": docs}


class SummarizeDocumentTool:
    name = "summarize_document"
    description = ("Summarize a single local document by its root-relative path, or every document "
                   "under a subdirectory. Provide 'path' OR 'subdir' (not both; neither = whole root).")
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Root-relative path of a single document."},
            "subdir": {"type": "string", "description": "Root-relative subdirectory to summarize."},
        },
    }

    def __init__(self, config: DocQAConfig, chat: ChatModel) -> None:
        self._cfg = config
        self._chat = chat

    async def run(self, args: dict) -> Any:
        target = _resolve_target(args)
        if isinstance(target, dict):
            return target
        path, subdir = target
        c = self._cfg
        try:
            chunks, truncated = gather_sourced_chunks(
                c.docs_root, path=path, subdir=subdir,
                chunk_chars=c.summarize_chunk_chars,
                overlap=derive_overlap(c.summarize_chunk_chars, ratio=c.overlap_ratio),
                max_chunks=c.max_chunks, max_bytes=c.max_doc_bytes,
                max_docs=c.max_docs_per_query, max_total_chunks=c.max_total_chunks,
            )
            text = "\n\n".join(sc.text for sc in chunks)
            result = await summarize_text(
                text, chat=self._chat,
                chunk_chars=c.summarize_chunk_chars,
                overlap=derive_overlap(c.summarize_chunk_chars, ratio=c.overlap_ratio),
                max_chunks=c.max_chunks, max_summary_tokens=c.summary_max_tokens,
                reduce_max_passes=c.reduce_max_passes,
            )
        except _GRACEFUL as exc:
            return _err(exc)
        return {"ok": True, "summary": result.summary,
                "truncated": truncated or result.truncated, "chunk_count": result.chunk_count}


class AnswerQuestionTool:
    name = "answer_question"
    description = ("Answer a question grounded in the local documents (a single 'path' or a 'subdir'), "
                   "returning an answer with citations.")
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to answer."},
            "path": {"type": "string", "description": "Root-relative path of a single document."},
            "subdir": {"type": "string", "description": "Root-relative subdirectory to search."},
        },
        "required": ["question"],
    }

    def __init__(self, config: DocQAConfig, chat: ChatModel) -> None:
        self._cfg = config
        self._chat = chat

    async def run(self, args: dict) -> Any:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "ValueError: 'question' must be a non-empty string"}
        target = _resolve_target(args)
        if isinstance(target, dict):
            return target
        path, subdir = target
        c = self._cfg
        try:
            chunks, truncated = gather_sourced_chunks(
                c.docs_root, path=path, subdir=subdir,
                chunk_chars=c.qa_chunk_chars,
                overlap=derive_overlap(c.qa_chunk_chars, ratio=c.overlap_ratio),
                max_chunks=c.max_chunks, max_bytes=c.max_doc_bytes,
                max_docs=c.max_docs_per_query, max_total_chunks=c.max_total_chunks,
            )
            result = await answer_question(
                question, chunks, chat=self._chat,
                top_k=c.qa_top_k, max_context_chars=c.qa_max_context_chars,
                max_answer_tokens=c.answer_max_tokens,
            )
        except _GRACEFUL as exc:
            return _err(exc)
        return {
            "ok": True,
            "answer": result.answer,
            "answer_found": result.answer_found,
            "truncated": truncated,
            "citations": [
                {"source": ct.source, "index": ct.index, "start": ct.start, "end": ct.end}
                for ct in result.citations
            ],
        }


def build_tools(config: DocQAConfig, chat: ChatModel) -> list:
    """The 3 DocQA tools, in registration order."""
    return [
        ListDocumentsTool(config),
        SummarizeDocumentTool(config, chat),
        AnswerQuestionTool(config, chat),
    ]


# the tool names that are read-only and therefore safe to allowlist on the gate
SAFE_TOOL_NAMES = frozenset({"list_documents", "summarize_document", "answer_question"})
