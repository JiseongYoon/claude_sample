"""Tests for naive-retrieval QA + citations.

Pure retrieval (`select_chunks`) is deterministic; `answer_question` reaches the model only
through an injected fake `ChatModel`. Focus: keyword ranking + tie-break determinism, top_k /
budget selection, strict grounding (zero-retrieval short-circuit + NO_ANSWER sentinel), citation
accuracy, and fail-fast typed errors. Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.docqa.chunker import chunk_text
from local_ai_agent.modules.docqa.qa import (
    Citation,
    QAError,
    QAResult,
    ServingUnavailable,
    SourcedChunk,
    answer_question,
    select_chunks,
    sourced_chunks,
)
from local_ai_agent.modules.llm_serving import EngineNotReady as _EngineNotReady


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _sc(source, index, text, start=0, end=None):
    return SourcedChunk(source, index, text, start, end if end is not None else len(text))


class FakeChat:
    def __init__(self, *, answer="ANSWER", raise_exc=None, response=None):
        self.calls = []
        self._answer = answer
        self._raise = raise_exc
        self._response = response

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self._raise is not None:
            raise self._raise
        if self._response is not None:
            return self._response
        return _resp(self._answer)


# --------------------------------------------------------------------------- #
# select_chunks — pure retrieval (normal)
# --------------------------------------------------------------------------- #
def test_ranking_by_distinct_terms():
    a = _sc("a.md", 0, "the cat sat on the mat") # matches: the, cat -> distinct 2
    b = _sc("b.md", 0, "a dog ran fast") # matches: none
    c = _sc("c.md", 0, "the cat and the cat sit here") # matches: the, cat, sit -> distinct 3
    sel = select_chunks("where did the cat sit", [a, b, c], top_k=5, max_context_chars=10000)
    assert [s.source for s in sel] == ["c.md", "a.md"] # c (3) ranks above a (2); b excluded


def test_tie_break_is_deterministic():
    # both match exactly {alpha} -> distinct 1; differ by frequency, then source, then index
    one = _sc("z.md", 0, "alpha") # freq 1
    two = _sc("a.md", 0, "alpha alpha") # freq 2 -> ranks first (freq desc)
    three = _sc("a.md", 1, "alpha beta") # freq 1, source a, index 1
    four = _sc("b.md", 0, "alpha gamma") # freq 1, source b
    sel = select_chunks("alpha", [one, two, three, four], top_k=5, max_context_chars=10000)
    # two (freq2) , then freq-1 group ordered by source asc then index: a#1, b#0, z#0
    assert [(s.source, s.index) for s in sel] == [("a.md", 0), ("a.md", 1), ("b.md", 0), ("z.md", 0)]


def test_top_k_cap():
    chunks = [_sc("d.md", i, "alpha word%d" % i) for i in range(10)]
    sel = select_chunks("alpha", chunks, top_k=3, max_context_chars=10000)
    assert len(sel) == 3


def test_budget_limits_total_and_keeps_top():
    big = _sc("big.md", 0, "alpha " + "x" * 500) # top match, large
    small = _sc("s.md", 0, "alpha small") # also matches, small
    sel = select_chunks("alpha", [big, small], top_k=5, max_context_chars=100)
    # budget 100 < big alone; top-ranked (more freq? both distinct1) -> tie freq: big freq1, small freq1
    # source order: big.md < s.md, so big is top and always included even though oversize
    assert sel[0].source == "big.md"
    assert sum(len(s.text) for s in sel[1:]) <= 100 # subsequent must fit remaining budget


def test_oversize_top_chunk_still_included():
    big = _sc("big.md", 0, "alpha " + "y" * 1000)
    sel = select_chunks("alpha", [big], top_k=5, max_context_chars=50)
    assert len(sel) == 1 and sel[0] is big


def test_no_overlap_returns_empty():
    chunks = [_sc("a.md", 0, "completely unrelated content")]
    assert select_chunks("xyzzy plugh", chunks, top_k=5, max_context_chars=10000) == []


def test_punctuation_only_question_no_terms():
    chunks = [_sc("a.md", 0, "real content here")]
    assert select_chunks("??? ...", chunks, top_k=5, max_context_chars=10000) == []


@pytest.mark.parametrize("top_k,budget", [(0, 100), (-1, 100), (5, 0), (5, -10)])
def test_select_invalid_params(top_k, budget):
    with pytest.raises(ValueError):
        select_chunks("alpha", [_sc("a.md", 0, "alpha")], top_k=top_k, max_context_chars=budget)


def test_sourced_chunks_helper():
    res = chunk_text("alpha beta gamma delta " * 50, chunk_chars=120, overlap=0, max_chunks=64)
    out = sourced_chunks("doc.md", res)
    assert len(out) == len(res.chunks)
    assert all(isinstance(s, SourcedChunk) and s.source == "doc.md" for s in out)
    assert [(s.index, s.text, s.start, s.end) for s in out] == \
           [(c.index, c.text, c.start, c.end) for c in res.chunks]


# --------------------------------------------------------------------------- #
# answer_question (normal)
# --------------------------------------------------------------------------- #
async def test_answer_found_with_citations():
    chunks = [_sc("doc.md", 2, "alpha beta", start=10, end=20),
              _sc("doc.md", 7, "alpha gamma", start=80, end=91)]
    fake = FakeChat(answer="The answer is 42.")
    res = await answer_question("alpha", chunks, chat=fake, top_k=5, max_context_chars=10000)
    assert res.answer_found and res.answer == "The answer is 42."
    assert set(res.citations) == {Citation("doc.md", 2, 10, 20), Citation("doc.md", 7, 80, 91)}
    assert len(fake.calls) == 1


async def test_no_retrieval_short_circuits_without_model_call():
    fake = FakeChat()
    res = await answer_question("xyzzy", [_sc("a.md", 0, "unrelated")], chat=fake,
                                top_k=5, max_context_chars=10000)
    assert res == QAResult(answer="", answer_found=False, citations=())
    assert fake.calls == [] # strict grounding: no model call


async def test_no_answer_sentinel():
    fake = FakeChat(answer="NO_ANSWER")
    res = await answer_question("alpha", [_sc("a.md", 0, "alpha content")], chat=fake,
                                top_k=5, max_context_chars=10000)
    assert not res.answer_found
    assert res.answer == "" and res.citations == ()
    assert len(fake.calls) == 1 # model WAS consulted, said no


async def test_max_tokens_passed():
    fake = FakeChat()
    await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                          top_k=5, max_context_chars=10000, max_answer_tokens=321)
    assert fake.calls[0][1].get("max_tokens") == 321


async def test_prompt_contains_source_tags():
    fake = FakeChat()
    await answer_question("alpha", [_sc("rep.md", 3, "alpha detail")], chat=fake,
                          top_k=5, max_context_chars=10000)
    user_msg = fake.calls[0][0][-1]["content"]
    assert "[rep.md#3]" in user_msg and "alpha detail" in user_msg


async def test_determinism():
    chunks = [_sc("a.md", i, "alpha term%d here" % (i % 3)) for i in range(8)]
    a, b = FakeChat(answer="X"), FakeChat(answer="X")
    ra = await answer_question("alpha term1", chunks, chat=a, top_k=3, max_context_chars=10000)
    rb = await answer_question("alpha term1", chunks, chat=b, top_k=3, max_context_chars=10000)
    assert ra == rb
    assert a.calls[0][0] == b.calls[0][0] # identical prompt


# --------------------------------------------------------------------------- #
# answer_question (error / edge)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", ["", " ", "\n\t "])
async def test_empty_question_raises(q):
    fake = FakeChat()
    with pytest.raises(ValueError):
        await answer_question(q, [_sc("a.md", 0, "alpha")], chat=fake, top_k=5, max_context_chars=10000)
    assert fake.calls == []


async def test_empty_chunks_no_answer():
    fake = FakeChat()
    res = await answer_question("alpha", [], chat=fake, top_k=5, max_context_chars=10000)
    assert res == QAResult(answer="", answer_found=False, citations=())
    assert fake.calls == []


async def test_invalid_answer_tokens_raises():
    fake = FakeChat()
    with pytest.raises(ValueError):
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                              top_k=5, max_context_chars=10000, max_answer_tokens=0)


async def test_engine_not_ready_maps_to_serving_unavailable():
    fake = FakeChat(raise_exc=_EngineNotReady("down"))
    with pytest.raises(ServingUnavailable):
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                              top_k=5, max_context_chars=10000)


async def test_plain_runtime_error_is_not_serving_unavailable():
    fake = FakeChat(raise_exc=RuntimeError("x"))
    with pytest.raises(QAError) as ei:
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                              top_k=5, max_context_chars=10000)
    assert not isinstance(ei.value, ServingUnavailable)


async def test_transport_error_maps_to_qa_error():
    fake = FakeChat(raise_exc=ConnectionError("boom"))
    with pytest.raises(QAError):
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                              top_k=5, max_context_chars=10000)


@pytest.mark.parametrize(
    "response",
    [
        "not a dict",
        {},
        {"choices": []},
        {"choices": ["bad"]},
        {"choices": [{}]},
        {"choices": [{"message": "x"}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 123}}]},
    ],
)
async def test_malformed_response_raises(response):
    fake = FakeChat(response=response)
    with pytest.raises(QAError):
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=fake,
                              top_k=5, max_context_chars=10000)


async def test_none_response_raises():
    # FakeChat's `response is not None` guard can't express a None return; use a dedicated fake
    class _NoneChat:
        async def chat(self, messages, **params):
            return None

    with pytest.raises(QAError):
        await answer_question("alpha", [_sc("a.md", 0, "alpha")], chat=_NoneChat(),
                              top_k=5, max_context_chars=10000)
