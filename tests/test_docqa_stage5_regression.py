"""regression — DocQA as an integrated whole (hermetic).

The phase-completion gate: verify the composed DocQA capability works end-to-end across ALL
five formats (txt/md/pdf/docx/html) through the SAME composition the root wires under
`enable_docqa` — `DocQAModule` → tools registered on a `ToolDispatcher` behind a `SafetyGate`
whose `safe_tools` is extended with the 3 DocQA names — with an injected fake `ChatModel` (no
real model; that is the separate real-model smoke, A/B). Confirms loaders→chunker→summarizer/qa
flow for every format past the gate, plus the gate's safe/confirm/blocked routing over doc paths.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.docqa.module import DocQAModule
from local_ai_agent.modules.docqa.tools import SAFE_TOOL_NAMES
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict

from tests.test_docqa_loaders import _make_pdf # reuse the minimal-PDF builder

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeChat:
    def __init__(self, *, answer="SUMMARY"):
        self.calls = []
        self._answer = answer

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        return _resp(self._answer)


@pytest.fixture
def multi_format_root(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("alpha covers quarterly revenue figures.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Report\n\nbeta section about revenue growth.", encoding="utf-8")
    (tmp_path / "c.html").write_text(
        "<html><body><h1>Gamma</h1><script>x()</script><p>revenue outlook</p></body></html>",
        encoding="utf-8")
    (tmp_path / "d.pdf").write_bytes(_make_pdf("delta revenue PDF body"))
    import docx
    doc = docx.Document()
    doc.add_paragraph("epsilon revenue in the docx document")
    doc.save(str(tmp_path / "e.docx"))
    return tmp_path


def _compose(root):
    """Mirror build_application's enable_docqa branch with an injectable fake model."""
    settings = Settings(**_DIRS, enable_docqa=True, docs_root=root)
    chat = FakeChat()
    module = DocQAModule(settings, chat=chat)
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | SAFE_TOOL_NAMES)
    dispatcher = ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300),
                                tools=module.tools)
    return dispatcher, gate, chat


# --------------------------------------------------------------------------- #
# integrated whole — all formats flow through the composed gated path
# --------------------------------------------------------------------------- #
async def test_list_all_formats_via_dispatcher(multi_format_root):
    dispatcher, _, _ = _compose(multi_format_root)
    res = await dispatcher.dispatch(Action("list_documents", {"subdir": "."}), "stage5")
    assert res.outcome is Outcome.executed and res.result["ok"] is True
    assert res.result["documents"] == ["a.txt", "b.md", "c.html", "d.pdf", "e.docx"]


@pytest.mark.parametrize("doc", ["a.txt", "b.md", "c.html", "d.pdf", "e.docx"])
async def test_summarize_each_format_end_to_end(multi_format_root, doc):
    dispatcher, _, chat = _compose(multi_format_root)
    res = await dispatcher.dispatch(Action("summarize_document", {"path": doc}), "stage5")
    assert res.outcome is Outcome.executed # gate=safe → ran
    assert res.result["ok"] is True
    assert res.result["summary"] == "SUMMARY"
    assert res.result["chunk_count"] >= 1
    assert chat.calls # model was reached through the tool


async def test_answer_across_corpus_with_citations(multi_format_root):
    dispatcher, _, _ = _compose(multi_format_root)
    res = await dispatcher.dispatch(
        Action("answer_question", {"question": "what about revenue", "subdir": "."}), "stage5")
    assert res.outcome is Outcome.executed and res.result["ok"] is True
    assert res.result["answer_found"] is True
    sources = {c["source"] for c in res.result["citations"]}
    assert len(sources) >= 2 # revenue appears in several formats


# --------------------------------------------------------------------------- #
# gate routing over doc paths holds in the composed whole
# --------------------------------------------------------------------------- #
async def test_secret_blocked_in_composed_path(multi_format_root):
    dispatcher, _, chat = _compose(multi_format_root)
    res = await dispatcher.dispatch(Action("summarize_document", {"path": "/etc/shadow"}), "stage5")
    assert res.outcome is Outcome.refused and res.result is None
    assert chat.calls == [] # tool never ran → model never reached


async def test_absolute_path_confirm_in_composed_path(multi_format_root):
    dispatcher, _, _ = _compose(multi_format_root)
    res = await dispatcher.dispatch(Action("summarize_document", {"path": "/abs/x.pdf"}), "stage5")
    assert res.outcome is Outcome.pending and res.approval_id


def test_safe_routing_over_relative_paths(multi_format_root):
    _, gate, _ = _compose(multi_format_root)
    for name in ("list_documents", "summarize_document", "answer_question"):
        assert gate.classify(Action(name, {"path": "report.md"})).verdict is Verdict.safe
