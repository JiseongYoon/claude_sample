"""Integration: DocQA wired into the composition root.

Exercises the COMPOSED path — `build_application` under `enable_docqa` — not the per-component
batteries (steps 1–5a already cover those). Focus: the 3 tools register on the dispatcher and are safe-listed; gate interplay over doc paths (in-workspace→safe / absolute→
confirm / secret→blocked); **double containment** (an approved confirm action is still refused by
the `docs_root` sandbox); fault isolation (llm-serving with no model → graceful, capability gated).
Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.safety.dispatcher import Outcome
from local_ai_agent.modules.safety.gate import Action, Verdict

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


@pytest.fixture
def docs_root(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta gamma about quarterly reports.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("# c\n\ndelta epsilon content.", encoding="utf-8")
    return tmp_path


def _app(docs_root, **kw):
    return build_application(Settings(**_DIRS, enable_agent=True, enable_docqa=True,
                                      docs_root=docs_root, **kw))


# --------------------------------------------------------------------------- #
# registration + safe-listing
# --------------------------------------------------------------------------- #
def test_docqa_tools_safe_listed(docs_root):
    gate = _app(docs_root).agent_runtime.gate
    for name in ("list_documents", "summarize_document", "answer_question"):
        d = gate.classify(Action(name, {"path": "report.txt"})) # in-workspace relative
        assert d.verdict is Verdict.safe, name


async def test_list_documents_executes_via_dispatcher(docs_root):
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("list_documents", {"subdir": "."}), "tester")
    assert res.outcome is Outcome.executed
    assert res.result["ok"] is True
    assert res.result["documents"] == ["a.txt", "sub/c.md"]


# --------------------------------------------------------------------------- #
# gate interplay over doc paths
# --------------------------------------------------------------------------- #
async def test_secret_path_blocked_tool_never_runs(docs_root):
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("summarize_document", {"path": "/etc/shadow"}), "t")
    assert res.outcome is Outcome.refused
    assert res.result is None # tool never invoked


async def test_absolute_path_needs_confirmation(docs_root):
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("answer_question", {"question": "q", "path": "/abs/x.txt"}), "t")
    assert res.outcome is Outcome.pending and res.approval_id


# --------------------------------------------------------------------------- #
# double containment: gate-approved confirm action is still sandbox-refused
# --------------------------------------------------------------------------- #
async def test_double_containment_approved_escape_refused_by_sandbox(docs_root):
    disp = _app(docs_root).agent_runtime.dispatcher
    action = Action("summarize_document", {"path": "/abs/escaping.txt"})
    res = await disp.dispatch(action, "t")
    assert res.outcome is Outcome.pending # gate: confirm.nonlocal_path
    disp.approvals.approve(res.approval_id, action, _Approver())
    res2 = await disp.execute_approved(res.approval_id, action)
    assert res2.outcome is Outcome.executed # approval consumed, tool ran
    assert res2.result["ok"] is False # but docs_root sandbox refused the read
    assert "DocAccessError" in res2.result["error"]


# --------------------------------------------------------------------------- #
# fault isolation: no model loaded
# --------------------------------------------------------------------------- #
async def test_serving_down_summarize_graceful(docs_root):
    # no real model → llm-serving not ready → tool returns graceful unavailable, no crash
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("summarize_document", {"path": "a.txt"}), "t")
    assert res.outcome is Outcome.executed # gate=safe, tool ran...
    assert res.result["ok"] is False and "ServingUnavailable" in res.result["error"]


async def test_serving_down_answer_question_graceful(docs_root):
    # a question that DOES overlap the doc → retrieval succeeds → the model IS needed →
    # with no model loaded the tool degrades gracefully (the real fault-isolation path).
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("answer_question", {"question": "quarterly reports", "path": "a.txt"}), "t")
    assert res.outcome is Outcome.executed
    assert res.result["ok"] is False and "ServingUnavailable" in res.result["error"]


async def test_answer_no_match_is_not_found_without_model(docs_root):
    # strict grounding + zero-retrieval short-circuit (by design): when NO chunk matches the
    # question, the honest answer is "not found" and NO model call is made — so this returns a
    # valid ok=True/answer_found=False result even with no model loaded. The model is genuinely
    # unnecessary here, so this is correct (not a fault-isolation gap).
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("answer_question", {"question": "zzznonexistentterm", "path": "a.txt"}), "t")
    assert res.outcome is Outcome.executed
    assert res.result["ok"] is True and res.result["answer_found"] is False
    assert res.result["citations"] == []


async def test_double_containment_answer_question(docs_root):
    # the same approved-escape containment must hold for answer_question, not just summarize
    disp = _app(docs_root).agent_runtime.dispatcher
    action = Action("answer_question", {"question": "q", "path": "/abs/escaping.txt"})
    res = await disp.dispatch(action, "t")
    assert res.outcome is Outcome.pending
    disp.approvals.approve(res.approval_id, action, _Approver())
    res2 = await disp.execute_approved(res.approval_id, action)
    assert res2.outcome is Outcome.executed
    assert res2.result["ok"] is False and "DocAccessError" in res2.result["error"]


async def test_symlink_escape_refused_no_leak(docs_root, tmp_path):
    # a symlink INSIDE docs_root pointing OUTSIDE → resolve_within follows it, finds it escapes,
    # refuses. The path arg is relative ("link.txt") so the gate classifies it safe → the tool
    # runs → the sandbox refuses → graceful, and the secret content never leaks.
    secret = tmp_path.parent / "outside_secret.txt"
    secret.write_text("ULTRA_SECRET_TOKEN_9z", encoding="utf-8")
    try:
        (docs_root / "link.txt").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    disp = _app(docs_root).agent_runtime.dispatcher
    res = await disp.dispatch(Action("summarize_document", {"path": "link.txt"}), "t")
    assert res.outcome is Outcome.executed
    assert res.result["ok"] is False and "DocAccessError" in res.result["error"]
    assert "ULTRA_SECRET_TOKEN_9z" not in str(res.result)


async def test_consumed_approval_replay_aborted(docs_root):
    # an approval is single-use: re-running execute_approved after it consumed → aborted
    disp = _app(docs_root).agent_runtime.dispatcher
    action = Action("summarize_document", {"path": "/abs/escaping.txt"})
    res = await disp.dispatch(action, "t")
    disp.approvals.approve(res.approval_id, action, _Approver())
    first = await disp.execute_approved(res.approval_id, action)
    assert first.outcome is Outcome.executed
    replay = await disp.execute_approved(res.approval_id, action)
    assert replay.outcome is Outcome.aborted


async def test_docqa_capability_gated_when_no_model(docs_root):
    app = _app(docs_root)
    await app.startup()
    try:
        known = {c for m in app.registry.modules for c in m.spec.capabilities}
        assert "docqa" in known # module registered
        assert app.is_capability_available("docqa") is False # gated: llm-serving has no model
    finally:
        await app.shutdown()


# --------------------------------------------------------------------------- #
# no regression to the agent path / flag independence
# --------------------------------------------------------------------------- #
def test_default_safe_tools_unchanged(docs_root):
    gate = _app(docs_root).agent_runtime.gate
    assert gate.classify(Action("read_file", {"path": "x.txt"})).verdict is Verdict.safe
    assert gate.classify(Action("write_file", {"path": "x.txt"})).verdict is Verdict.needs_confirmation


def test_agent_without_docqa_does_not_safelist_docqa():
    app = build_application(Settings(**_DIRS, enable_agent=True, enable_docqa=False))
    gate = app.agent_runtime.gate
    # not safe-listed → falls through to the default needs_confirmation
    assert gate.classify(Action("summarize_document", {"path": "x.txt"})).verdict is not Verdict.safe


def test_docqa_without_agent_registers_module_no_runtime(docs_root):
    app = build_application(Settings(**_DIRS, enable_agent=False, enable_docqa=True, docs_root=docs_root))
    assert app.agent_runtime is None # no agent loop/WS
    known = {c for m in app.registry.modules for c in m.spec.capabilities}
    assert "docqa" in known # but the capability module is registered
