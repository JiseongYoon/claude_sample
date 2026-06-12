"""Tests for the DocQA map-reduce summarizer.

The model is reached only through an injected `ChatModel`; a deterministic `FakeChat`
stands in (no real model). Focus: map/reduce call structure, 1-chunk fast path, the
`reduce_max_passes` + terminal-combine bounds (termination), truncation propagation, and
fail-fast typed errors (engine down / transport error / malformed response). Run in conda
`local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.docqa.chunker import chunk_text
from local_ai_agent.modules.docqa.summarizer import (
    ServingUnavailable,
    SummarizeError,
    SummaryResult,
    summarize_text,
)
from local_ai_agent.modules.llm_serving import EngineNotReady

_MAP_PREFIX = "Summarize the following document section:"
_REDUCE_PREFIX = "Combine the following partial summaries"


def _resp(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeChat:
    """Records calls; returns a controllable summary. `summary_len` sets output size so a
    test can force single-pass, hierarchical, or terminal-combine reduce behaviour."""

    def __init__(self, *, summary_len=20, raise_exc=None, response=None):
        self.calls: list[tuple[list[dict], dict]] = []
        self._len = summary_len
        self._raise = raise_exc
        self._response = response

    async def chat(self, messages, **params):
        self.calls.append((messages, params))
        if self._raise is not None:
            raise self._raise
        if self._response is not None:
            return self._response
        return _resp("s" * self._len)

    # introspection helpers
    def _user(self, i):
        return self.calls[i][0][-1]["content"]

    @property
    def map_calls(self):
        return [c for c in self.calls if c[0][-1]["content"].startswith(_MAP_PREFIX)]

    @property
    def reduce_calls(self):
        return [c for c in self.calls if c[0][-1]["content"].startswith(_REDUCE_PREFIX)]


# --------------------------------------------------------------------------- #
# normal class
# --------------------------------------------------------------------------- #
async def test_empty_text_empty_summary():
    fake = FakeChat()
    res = await summarize_text(" \n ", chat=fake, chunk_chars=100, overlap=0, max_chunks=10)
    assert res == SummaryResult(summary="", truncated=False, chunk_count=0, reduce_passes=0)
    assert fake.calls == [] # no model call for empty input


async def test_single_chunk_fast_path():
    fake = FakeChat(summary_len=12)
    text = "a short document that fits in one chunk"
    res = await summarize_text(text, chat=fake, chunk_chars=500, overlap=0, max_chunks=10)
    assert res.chunk_count == 1
    assert res.reduce_passes == 0
    assert res.summary == "s" * 12
    assert len(fake.calls) == 1 # exactly one map call, no reduce
    assert len(fake.map_calls) == 1 and len(fake.reduce_calls) == 0
    assert text in fake._user(0) # the chunk was sent


async def test_multi_chunk_map_then_reduce():
    fake = FakeChat(summary_len=20)
    text = "word " * 600 # many chunks
    expected_chunks = len(chunk_text(text, chunk_chars=300, overlap=0, max_chunks=64).chunks)
    assert expected_chunks > 1
    res = await summarize_text(text, chat=fake, chunk_chars=300, overlap=0, max_chunks=64)
    assert res.chunk_count == expected_chunks
    assert res.reduce_passes >= 1
    assert not res.truncated
    assert len(fake.map_calls) == expected_chunks # one map per chunk
    assert len(fake.reduce_calls) >= 1


async def test_hierarchical_reduce_multiple_passes():
    # summaries (20 chars) pack ~2 per 60-char budget -> several reduce passes to converge
    fake = FakeChat(summary_len=20)
    text = "alpha beta gamma " * 200
    res = await summarize_text(text, chat=fake, chunk_chars=120, overlap=0, max_chunks=64)
    assert res.chunk_count > 4
    assert res.reduce_passes >= 2 # genuinely hierarchical
    assert res.summary == "s" * 20


async def test_max_tokens_passed_on_every_call():
    fake = FakeChat()
    await summarize_text("word " * 400, chat=fake, chunk_chars=200, overlap=0,
                         max_chunks=64, max_summary_tokens=333)
    assert fake.calls
    assert all(params.get("max_tokens") == 333 for _, params in fake.calls)


async def test_truncation_propagated():
    fake = FakeChat()
    text = "w" * 50_000 # far exceeds max_chunks * chunk_chars
    res = await summarize_text(text, chat=fake, chunk_chars=100, overlap=0, max_chunks=3)
    assert res.truncated
    assert res.chunk_count == 3
    assert len(fake.map_calls) == 3


async def test_determinism_same_calls():
    a = FakeChat(summary_len=15)
    b = FakeChat(summary_len=15)
    text = "Lorem ipsum dolor sit amet. " * 100
    ra = await summarize_text(text, chat=a, chunk_chars=200, overlap=20, max_chunks=64)
    rb = await summarize_text(text, chat=b, chunk_chars=200, overlap=20, max_chunks=64)
    assert ra == rb
    assert [u[-1]["content"] for u, _ in a.calls] == [u[-1]["content"] for u, _ in b.calls]


# --------------------------------------------------------------------------- #
# error / edge class
# --------------------------------------------------------------------------- #
async def test_engine_not_ready_maps_to_serving_unavailable():
    fake = FakeChat(raise_exc=EngineNotReady("no model loaded"))
    with pytest.raises(ServingUnavailable):
        await summarize_text("word " * 100, chat=fake, chunk_chars=200, overlap=0, max_chunks=64)


async def test_transport_error_maps_to_summarize_error():
    fake = FakeChat(raise_exc=ConnectionError("boom"))
    with pytest.raises(SummarizeError) as ei:
        await summarize_text("word " * 100, chat=fake, chunk_chars=200, overlap=0, max_chunks=64)
    assert not isinstance(ei.value, ServingUnavailable)


async def test_map_failure_is_fail_fast():
    # fails on the very first call -> no further calls made
    fake = FakeChat(raise_exc=RuntimeError("x"))
    with pytest.raises(SummarizeError):
        await summarize_text("word " * 400, chat=fake, chunk_chars=100, overlap=0, max_chunks=64)
    assert len(fake.calls) == 1


async def test_plain_runtime_error_is_not_serving_unavailable():
    # EngineNotReady subclasses RuntimeError; a plain RuntimeError must NOT be misclassified
    fake = FakeChat(raise_exc=RuntimeError("generic"))
    with pytest.raises(SummarizeError) as ei:
        await summarize_text("word " * 100, chat=fake, chunk_chars=200, overlap=0, max_chunks=64)
    assert not isinstance(ei.value, ServingUnavailable)


class _FailOnReduce:
    """Succeeds for every map call, raises on the first reduce call."""

    def __init__(self):
        self.map_n = 0
        self.reduce_n = 0

    async def chat(self, messages, **params):
        if messages[-1]["content"].startswith(_REDUCE_PREFIX):
            self.reduce_n += 1
            raise ConnectionError("reduce down")
        self.map_n += 1
        return _resp("s" * 20)


async def test_fail_fast_on_reduce_call():
    fake = _FailOnReduce()
    text = "word " * 600
    n_chunks = len(chunk_text(text, chunk_chars=300, overlap=0, max_chunks=64).chunks)
    assert n_chunks > 1
    with pytest.raises(SummarizeError):
        await summarize_text(text, chat=fake, chunk_chars=300, overlap=0, max_chunks=64)
    assert fake.map_n == n_chunks # all maps ran
    assert fake.reduce_n == 1 # then the first reduce failed and aborted


@pytest.mark.parametrize(
    "response",
    [
        "not a dict",
        {}, # no choices
        {"choices": []}, # empty choices
        {"choices": ["bad"]}, # choice not a dict
        {"choices": [{}]}, # no message
        {"choices": [{"message": "x"}]}, # message not a dict
        {"choices": [{"message": {}}]}, # no content
        {"choices": [{"message": {"content": 123}}]},# content not a string
    ],
)
async def test_malformed_response_raises(response):
    fake = FakeChat(response=response)
    with pytest.raises(SummarizeError):
        await summarize_text("a single chunk doc", chat=fake, chunk_chars=500, overlap=0, max_chunks=10)


async def test_empty_string_content_is_valid():
    fake = FakeChat(response=_resp(""))
    res = await summarize_text("a single chunk doc", chat=fake, chunk_chars=500, overlap=0, max_chunks=10)
    assert res.summary == ""
    assert res.chunk_count == 1


async def test_reduce_non_convergence_raises():
    fake = FakeChat(summary_len=20)
    text = "alpha beta gamma " * 200 # many chunks -> >1 group after 1 pass
    with pytest.raises(SummarizeError):
        await summarize_text(text, chat=fake, chunk_chars=80, overlap=0,
                             max_chunks=64, reduce_max_passes=1)


async def test_terminal_combine_when_summaries_exceed_budget():
    # each summary (1000 chars) is larger than the 120-char budget -> packing can't shrink
    # -> exactly one terminal combine, must still terminate
    fake = FakeChat(summary_len=1000)
    text = "z" * 500 # 5 hard-cut chunks of 100
    res = await summarize_text(text, chat=fake, chunk_chars=120, overlap=0, max_chunks=64)
    assert res.chunk_count == 5
    assert res.reduce_passes == 1 # single terminal combine
    assert len(fake.map_calls) == 5 and len(fake.reduce_calls) == 1
    assert res.summary == "s" * 1000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_chars": 100, "overlap": 0, "max_chunks": 10, "max_summary_tokens": 0},
        {"chunk_chars": 100, "overlap": 0, "max_chunks": 10, "max_summary_tokens": -1},
        {"chunk_chars": 100, "overlap": 0, "max_chunks": 10, "reduce_max_passes": 0},
        {"chunk_chars": 0, "overlap": 0, "max_chunks": 10}, # delegated to chunk_text
        {"chunk_chars": 100, "overlap": 100, "max_chunks": 10}, # overlap >= chunk_chars
        {"chunk_chars": 100, "overlap": 0, "max_chunks": 0},
    ],
)
async def test_invalid_params_raise_valueerror(kwargs):
    fake = FakeChat()
    with pytest.raises(ValueError):
        await summarize_text("some text", chat=fake, **kwargs)
