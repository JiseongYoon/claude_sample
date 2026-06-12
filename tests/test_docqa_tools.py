"""Tests for DocQA tools + module + load/chunk glue.

Filesystem reads use real `tmp_path` doc fixtures through the sandbox; the model is an
injected fake `ChatModel`. Focus: read-only behaviour, double containment (escape → no read),
bounded directory queries, graceful structured errors (never raise for a known failure), tool
result shapes, and the `DocQAModule` spec/health/tools. Run in conda `local-ai-agent-env-1`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.docqa.module import DocQAModule
from local_ai_agent.modules.docqa.tools import (
    AnswerQuestionTool,
    DocQAConfig,
    ListDocumentsTool,
    SummarizeDocumentTool,
    build_tools,
    gather_sourced_chunks,
)
from local_ai_agent.modules.llm_serving import EngineNotReady


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeChat:
    def __init__(self, *, answer="ANSWER", raise_exc=None):
        self.calls = []
        self._answer = answer
        self._raise = raise_exc

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self._raise is not None:
            raise self._raise
        return _resp(self._answer)


def _cfg(root, **over):
    base = dict(
        docs_root=root, max_doc_bytes=10_000_000,
        summarize_chunk_chars=2000, qa_chunk_chars=500, overlap_ratio=0.1,
        max_chunks=64, summary_max_tokens=256, reduce_max_passes=5,
        qa_top_k=5, qa_max_context_chars=4000, answer_max_tokens=256,
        max_docs_per_query=20, max_total_chunks=128,
    )
    base.update(over)
    return DocQAConfig(**base)


@pytest.fixture
def docs_root(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta gamma. the report covers alpha topics.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\ndelta epsilon. beta appears here too.", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("nested zeta content discussing gamma in detail.", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# list_documents
# --------------------------------------------------------------------------- #
async def test_list_documents(docs_root):
    tool = ListDocumentsTool(_cfg(docs_root))
    out = await tool.run({})
    assert out["ok"] is True
    assert out["documents"] == ["a.txt", "b.md", "sub/c.txt"]


async def test_list_documents_non_recursive(docs_root):
    out = await ListDocumentsTool(_cfg(docs_root)).run({"subdir": ".", "recursive": False})
    assert out["documents"] == ["a.txt", "b.md"]


async def test_list_documents_bad_subdir_type(docs_root):
    out = await ListDocumentsTool(_cfg(docs_root)).run({"subdir": 123})
    assert out["ok"] is False and "subdir" in out["error"]


# --------------------------------------------------------------------------- #
# summarize_document
# --------------------------------------------------------------------------- #
async def test_summarize_single_doc(docs_root):
    fake = FakeChat(answer="A short summary.")
    out = await SummarizeDocumentTool(_cfg(docs_root), fake).run({"path": "a.txt"})
    assert out["ok"] is True
    assert out["summary"] == "A short summary."
    assert out["chunk_count"] >= 1
    assert len(fake.calls) >= 1


async def test_summarize_directory(docs_root):
    fake = FakeChat(answer="combined")
    out = await SummarizeDocumentTool(_cfg(docs_root), fake).run({"subdir": "."})
    assert out["ok"] is True and out["summary"] == "combined"


async def test_summarize_defaults_to_whole_root(docs_root):
    fake = FakeChat(answer="x")
    out = await SummarizeDocumentTool(_cfg(docs_root), fake).run({})
    assert out["ok"] is True


async def test_summarize_both_path_and_subdir_errors(docs_root):
    out = await SummarizeDocumentTool(_cfg(docs_root), FakeChat()).run({"path": "a.txt", "subdir": "."})
    assert out["ok"] is False and "not both" in out["error"]


async def test_summarize_missing_file(docs_root):
    out = await SummarizeDocumentTool(_cfg(docs_root), FakeChat()).run({"path": "nope.txt"})
    assert out["ok"] is False and "DocNotFound" in out["error"]


async def test_summarize_escaping_path_contained(docs_root, tmp_path):
    # a sibling file outside the root must NOT be readable (double containment)
    (tmp_path.parent / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    out = await SummarizeDocumentTool(_cfg(docs_root), FakeChat()).run({"path": "../secret.txt"})
    assert out["ok"] is False and "DocAccessError" in out["error"]


async def test_summarize_serving_unavailable_graceful(docs_root):
    fake = FakeChat(raise_exc=EngineNotReady("no model"))
    out = await SummarizeDocumentTool(_cfg(docs_root), fake).run({"path": "a.txt"})
    assert out["ok"] is False and "ServingUnavailable" in out["error"]


# --------------------------------------------------------------------------- #
# answer_question
# --------------------------------------------------------------------------- #
async def test_answer_single_doc_with_citations(docs_root):
    fake = FakeChat(answer="Alpha topics are covered.")
    out = await AnswerQuestionTool(_cfg(docs_root), fake).run({"question": "what about alpha", "path": "a.txt"})
    assert out["ok"] is True and out["answer_found"] is True
    assert out["answer"] == "Alpha topics are covered."
    assert out["citations"] and out["citations"][0]["source"] == "a.txt"
    assert all({"source", "index", "start", "end"} <= c.keys() for c in out["citations"])


async def test_answer_directory_cites_multiple_sources(docs_root):
    fake = FakeChat(answer="gamma is discussed in several docs")
    out = await AnswerQuestionTool(_cfg(docs_root), fake).run({"question": "gamma", "subdir": "."})
    assert out["ok"] is True and out["answer_found"] is True
    sources = {c["source"] for c in out["citations"]}
    assert "a.txt" in sources and "sub/c.txt" in sources


async def test_answer_no_retrieval_no_model_call(docs_root):
    fake = FakeChat()
    out = await AnswerQuestionTool(_cfg(docs_root), fake).run({"question": "zzznonexistent", "path": "a.txt"})
    assert out["ok"] is True and out["answer_found"] is False
    assert out["citations"] == []
    assert fake.calls == [] # strict grounding


async def test_answer_no_answer_sentinel(docs_root):
    fake = FakeChat(answer="NO_ANSWER")
    out = await AnswerQuestionTool(_cfg(docs_root), fake).run({"question": "alpha", "path": "a.txt"})
    assert out["ok"] is True and out["answer_found"] is False and out["citations"] == []
    assert len(fake.calls) == 1


@pytest.mark.parametrize("q", [None, "", " ", 123])
async def test_answer_invalid_question(docs_root, q):
    out = await AnswerQuestionTool(_cfg(docs_root), FakeChat()).run({"question": q, "path": "a.txt"})
    assert out["ok"] is False and "question" in out["error"]


async def test_answer_serving_unavailable_graceful(docs_root):
    fake = FakeChat(raise_exc=EngineNotReady("down"))
    out = await AnswerQuestionTool(_cfg(docs_root), fake).run({"question": "alpha", "path": "a.txt"})
    assert out["ok"] is False and "ServingUnavailable" in out["error"]


# --------------------------------------------------------------------------- #
# gather_sourced_chunks glue
# --------------------------------------------------------------------------- #
def test_gather_single_doc(docs_root):
    chunks, truncated = gather_sourced_chunks(
        docs_root, path="a.txt", chunk_chars=500, overlap=10, max_chunks=64,
        max_bytes=10_000_000, max_docs=20, max_total_chunks=128)
    assert chunks and all(c.source == "a.txt" for c in chunks)
    assert not truncated


def test_gather_directory_labels_sources(docs_root):
    chunks, _ = gather_sourced_chunks(
        docs_root, subdir=".", chunk_chars=500, overlap=10, max_chunks=64,
        max_bytes=10_000_000, max_docs=20, max_total_chunks=128)
    assert {c.source for c in chunks} == {"a.txt", "b.md", "sub/c.txt"}


def test_gather_max_docs_truncates(docs_root):
    _, truncated = gather_sourced_chunks(
        docs_root, subdir=".", chunk_chars=500, overlap=10, max_chunks=64,
        max_bytes=10_000_000, max_docs=2, max_total_chunks=128)
    assert truncated is True # 3 docs, cap 2


def test_gather_max_total_chunks_truncates(docs_root):
    chunks, truncated = gather_sourced_chunks(
        docs_root, subdir=".", chunk_chars=10, overlap=0, max_chunks=64,
        max_bytes=10_000_000, max_docs=20, max_total_chunks=3)
    assert len(chunks) == 3 and truncated is True


def test_gather_requires_exactly_one_target(docs_root):
    with pytest.raises(ValueError):
        gather_sourced_chunks(docs_root, chunk_chars=500, overlap=10, max_chunks=64,
                              max_bytes=10_000_000, max_docs=20, max_total_chunks=128)
    with pytest.raises(ValueError):
        gather_sourced_chunks(docs_root, path="a.txt", subdir=".", chunk_chars=500, overlap=10,
                              max_chunks=64, max_bytes=10_000_000, max_docs=20, max_total_chunks=128)


# --------------------------------------------------------------------------- #
# DocQAModule + build_tools + DocQAConfig.from_settings
# --------------------------------------------------------------------------- #
def test_build_tools_names(docs_root):
    tools = build_tools(_cfg(docs_root), FakeChat())
    assert [t.name for t in tools] == ["list_documents", "summarize_document", "answer_question"]


async def test_module_spec_and_health(docs_root):
    from local_ai_agent.config import Settings

    settings = Settings(
        model_safetensors_dir="/tmp/st", model_gguf_dir="/tmp/gguf",
        enable_docqa=True, docs_root=docs_root,
    )
    mod = DocQAModule(settings, FakeChat())
    assert mod.spec.name == "docqa"
    assert mod.spec.capabilities == ("docqa",)
    assert mod.spec.depends_on == ("llm-serving",)
    assert [t.name for t in mod.tools] == ["list_documents", "summarize_document", "answer_question"]
    assert mod.health().status.value == "absent"
    await mod.start()
    assert mod.health().status.value == "ok"
    await mod.stop()
    assert mod.health().status.value == "absent"


def test_config_from_settings_requires_docs_root():
    from local_ai_agent.config import Settings

    settings = Settings(model_safetensors_dir="/tmp/st", model_gguf_dir="/tmp/gguf",
                        enable_docqa=True, docs_root=None)
    with pytest.raises(ValueError):
        DocQAConfig.from_settings(settings)


def test_config_from_settings_derives_chunk_sizes(docs_root):
    from local_ai_agent.config import Settings

    settings = Settings(model_safetensors_dir="/tmp/st", model_gguf_dir="/tmp/gguf",
                        enable_docqa=True, docs_root=docs_root, ctx_size=16384)
    cfg = DocQAConfig.from_settings(settings)
    assert cfg.summarize_chunk_chars == (16384 - 2048) * 3 # derive_chunk_chars
    assert cfg.qa_chunk_chars == 2000
    assert cfg.docs_root == docs_root
