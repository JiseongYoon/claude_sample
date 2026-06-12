"""Tests for the remote → DocQA tools.

Composed with a fake `RemoteTransport` connector + a fake `ChatModel` (no network, no model).
Focus: a contained remote read → `load_bytes` (format-aware) → the summarize/QA pipeline;
graceful failure for storage-down / llm-down / unknown connector / bad args; containment + gate
interplay over remote paths. Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.modules.docqa.tools import DocQAConfig
from local_ai_agent.modules.llm_serving import EngineNotReady
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from local_ai_agent.modules.storage.connector import GuardedConnector
from local_ai_agent.modules.storage.remote_docqa import (
    REMOTE_DOCQA_SAFE_TOOL_NAMES,
    AnswerRemoteTool,
    SummarizeRemoteTool,
    build_remote_tools,
)
from tests.test_docqa_loaders import _make_pdf
from tests.test_storage_connector import FakeTransport

_ROOT = "/srv/share"


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeChat:
    def __init__(self, *, answer="SUMMARY", raise_exc=None):
        self.calls = []
        self._answer = answer
        self._raise = raise_exc

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self._raise is not None:
            raise self._raise
        return _resp(self._answer)


def _cfg():
    return DocQAConfig(
        docs_root=Path("/unused"), max_doc_bytes=10_000_000,
        summarize_chunk_chars=2000, qa_chunk_chars=500, overlap_ratio=0.1,
        max_chunks=64, summary_max_tokens=256, reduce_max_passes=5,
        qa_top_k=5, qa_max_context_chars=4000, answer_max_tokens=256,
        max_docs_per_query=20, max_total_chunks=128,
    )


def _connectors(*, read_only=True):
    t = FakeTransport(
        files={
            f"{_ROOT}/doc.txt": b"alpha remote content discussing revenue figures in detail.",
            f"{_ROOT}/report.pdf": _make_pdf("Hello PDF revenue"),
        },
        dirs={_ROOT},
    )
    return {"nas": GuardedConnector(t, allowed_root=_ROOT, read_only=read_only, max_bytes=10_000_000)}, t


def _summarize_tool(**kw):
    return SummarizeRemoteTool(_connectors()[0], FakeChat(**kw), _cfg())


# --------------------------------------------------------------------------- #
# summarize_remote
# --------------------------------------------------------------------------- #
async def test_summarize_remote_txt():
    res = await _summarize_tool(answer="A summary.").run({"connector": "nas", "path": "doc.txt"})
    assert res["ok"] is True and res["summary"] == "A summary." and res["chunk_count"] >= 1


async def test_summarize_remote_pdf_format_aware():
    # the PDF bytes are parsed via load_bytes (format-aware) before summarizing
    fake = FakeChat(answer="pdf summary")
    res = await SummarizeRemoteTool(_connectors()[0], fake, _cfg()).run(
        {"connector": "nas", "path": "report.pdf"})
    assert res["ok"] is True and res["summary"] == "pdf summary"
    assert fake.calls # model reached → PDF decoded to text successfully


async def test_summarize_remote_missing_file_graceful():
    res = await _summarize_tool().run({"connector": "nas", "path": "nope.txt"})
    assert res["ok"] is False and "StorageNotFound" in res["error"]


async def test_summarize_remote_escape_contained():
    res = await _summarize_tool().run({"connector": "nas", "path": "../../etc/shadow"})
    assert res["ok"] is False and "StorageAccessError" in res["error"]


async def test_summarize_remote_unknown_connector():
    res = await _summarize_tool().run({"connector": "ghost", "path": "doc.txt"})
    assert res["ok"] is False and "unknown connector" in res["error"]


async def test_summarize_remote_llm_down_graceful():
    res = await _summarize_tool(raise_exc=EngineNotReady("no model")).run(
        {"connector": "nas", "path": "doc.txt"})
    assert res["ok"] is False and "ServingUnavailable" in res["error"]


# --------------------------------------------------------------------------- #
# answer_remote
# --------------------------------------------------------------------------- #
async def test_answer_remote_with_citation():
    tool = AnswerRemoteTool(_connectors()[0], FakeChat(answer="Revenue is up."), _cfg())
    res = await tool.run({"question": "what about revenue", "connector": "nas", "path": "doc.txt"})
    assert res["ok"] is True and res["answer_found"] is True
    assert res["answer"] == "Revenue is up."
    assert res["citations"] and res["citations"][0]["source"] == "doc.txt"


async def test_answer_remote_no_match_not_found_no_model_call():
    fake = FakeChat()
    tool = AnswerRemoteTool(_connectors()[0], fake, _cfg())
    res = await tool.run({"question": "zzznonexistent", "connector": "nas", "path": "doc.txt"})
    assert res["ok"] is True and res["answer_found"] is False and res["citations"] == []
    assert fake.calls == [] # strict grounding: zero retrieval → no model call


@pytest.mark.parametrize("q", [None, "", " ", 123])
async def test_answer_remote_invalid_question(q):
    tool = AnswerRemoteTool(_connectors()[0], FakeChat(), _cfg())
    res = await tool.run({"question": q, "connector": "nas", "path": "doc.txt"})
    assert res["ok"] is False and "question" in res["error"]


async def test_answer_remote_storage_down_graceful():
    class Boom:
        async def stat(self, p):
            raise RuntimeError("host=10.0.0.9 pw=secret")
        async def list_dir(self, p): ...
        async def read_bytes(self, p): ...
        async def write_bytes(self, p, d): ...
        async def delete(self, p): ...
        async def move(self, s, d): ...
        async def realpath(self, p): return p
        async def close(self): ...

    connectors = {"nas": GuardedConnector(Boom(), allowed_root=_ROOT, read_only=True, max_bytes=10_000_000)}
    tool = AnswerRemoteTool(connectors, FakeChat(), _cfg())
    res = await tool.run({"question": "revenue", "connector": "nas", "path": "doc.txt"})
    assert res["ok"] is False
    assert "secret" not in res["error"] and "password" not in res["error"] # no cred leak


# --------------------------------------------------------------------------- #
# gate interplay via the dispatcher
# --------------------------------------------------------------------------- #
def _dispatcher():
    connectors, _ = _connectors()
    tools = build_remote_tools(connectors, FakeChat(answer="S"), _cfg())
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | REMOTE_DOCQA_SAFE_TOOL_NAMES)
    return ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=tools)


def test_remote_tools_safe_listed():
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | REMOTE_DOCQA_SAFE_TOOL_NAMES)
    for name in ("summarize_remote", "answer_remote"):
        assert gate.classify(Action(name, {"connector": "nas", "path": "doc.txt"})).verdict is Verdict.safe


async def test_remote_summarize_executes_via_dispatcher():
    res = await _dispatcher().dispatch(Action("summarize_remote", {"connector": "nas", "path": "doc.txt"}), "t")
    assert res.outcome is Outcome.executed and res.result["ok"] is True


async def test_remote_secret_path_blocked():
    res = await _dispatcher().dispatch(
        Action("summarize_remote", {"connector": "nas", "path": "/etc/shadow"}), "t")
    assert res.outcome is Outcome.refused and res.result is None
