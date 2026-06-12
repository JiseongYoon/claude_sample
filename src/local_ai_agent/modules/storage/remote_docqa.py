"""Remote → DocQA gated tools.

Bridge the storage capability and the DocQA pipeline: fetch a CONTAINED
remote file via a `GuardedConnector`, decode its bytes to text (format-aware `loaders.load_bytes`),
and run the summarizer / QA over it. Read-only (fetch + summarize/answer; no remote write).

These tools need a `ChatModel` + DocQA params, so the composition root registers them ONLY when
both `enable_storage` and `enable_docqa` are set. Containment, read-only, and size are enforced by
the same `GuardedConnector` the plain storage tools use; every known failure (`StorageError` /
`DocError` / `SummarizeError` / `QAError`) → a graceful `{"ok": False, "error": ...}`.
"""
from __future__ import annotations

from typing import Any

from ..docqa.chunker import chunk_text, derive_overlap
from ..docqa.loaders import DocError, load_bytes
from ..docqa.qa import QAError, answer_question, sourced_chunks
from ..docqa.summarizer import ChatModel, SummarizeError, summarize_text
from ..docqa.tools import DocQAConfig
from .connector import GuardedConnector, StorageError
from .tools import _BaseStorageTool, _err

# read-only → safe to allowlist on the gate (wiring adds these only when both flags are on)
REMOTE_DOCQA_SAFE_TOOL_NAMES = frozenset({"summarize_remote", "answer_remote"})

_GRACEFUL = (StorageError, DocError, SummarizeError, QAError)


class _RemoteDocQABase(_BaseStorageTool):
    def __init__(self, connectors: dict[str, GuardedConnector], chat: ChatModel, config: DocQAConfig) -> None:
        super().__init__(connectors)
        self._chat = chat
        self._cfg = config

    async def _fetch_text(self, conn: GuardedConnector, path: str) -> str:
        data = await conn.read_bytes(path) # contained + size-capped
        return load_bytes(path, data, max_bytes=self._cfg.max_doc_bytes)


class SummarizeRemoteTool(_RemoteDocQABase):
    name = "summarize_remote"
    description = "Summarize a document stored on a remote storage connector (read + summarize)."
    parameters = {
        "type": "object",
        "properties": {
            "connector": {"type": "string", "description": "Name of the configured storage connector."},
            "path": {"type": "string", "description": "Connector-relative document path."},
        },
        "required": ["connector", "path"],
    }

    async def run(self, args: dict) -> Any:
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        c = self._cfg
        try:
            text = await self._fetch_text(conn, path)
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
                "truncated": result.truncated, "chunk_count": result.chunk_count}


class AnswerRemoteTool(_RemoteDocQABase):
    name = "answer_remote"
    description = "Answer a question grounded in a document stored on a remote storage connector."
    parameters = {
        "type": "object",
        "properties": {
            "connector": {"type": "string", "description": "Name of the configured storage connector."},
            "path": {"type": "string", "description": "Connector-relative document path."},
            "question": {"type": "string", "description": "The question to answer."},
        },
        "required": ["connector", "path", "question"],
    }

    async def run(self, args: dict) -> Any:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "ValueError: 'question' must be a non-empty string"}
        conn = self._resolve(args)
        if isinstance(conn, dict):
            return conn
        path = self._str_arg(args, "path")
        if isinstance(path, dict):
            return path
        c = self._cfg
        try:
            text = await self._fetch_text(conn, path)
            chunked = chunk_text(
                text, chunk_chars=c.qa_chunk_chars,
                overlap=derive_overlap(c.qa_chunk_chars, ratio=c.overlap_ratio),
                max_chunks=c.max_chunks,
            )
            result = await answer_question(
                question, sourced_chunks(path, chunked), chat=self._chat,
                top_k=c.qa_top_k, max_context_chars=c.qa_max_context_chars,
                max_answer_tokens=c.answer_max_tokens,
            )
        except _GRACEFUL as exc:
            return _err(exc)
        return {
            "ok": True, "answer": result.answer, "answer_found": result.answer_found,
            "truncated": chunked.truncated,
            "citations": [
                {"source": ct.source, "index": ct.index, "start": ct.start, "end": ct.end}
                for ct in result.citations
            ],
        }


def build_remote_tools(connectors: dict[str, GuardedConnector], chat: ChatModel, config: DocQAConfig) -> list:
    return [SummarizeRemoteTool(connectors, chat, config), AnswerRemoteTool(connectors, chat, config)]
