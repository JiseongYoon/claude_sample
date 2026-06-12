"""Tests for the DocQA chunker + token budgeting.

Pure, deterministic, dependency-free. Focus: termination (no infinite loop for any
input), offset fidelity (chunk.text == source[start:end]), determinism, the max_chunks
cap (truncated signalled), and ValueError on invalid params. Run in conda
`local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.docqa.chunker import (
    Chunk,
    ChunkResult,
    chunk_text,
    derive_chunk_chars,
    derive_overlap,
)


# --------------------------------------------------------------------------- #
# helpers — the structural invariants every successful chunking must satisfy
# --------------------------------------------------------------------------- #
def _assert_invariants(text: str, result: ChunkResult, *, chunk_chars: int) -> None:
    assert result.total_chars == len(text)
    chunks = result.chunks
    if not chunks:
        return
    for i, c in enumerate(chunks):
        assert c.index == i # contiguous 0..N-1
        assert c.text == text[c.start : c.end] # offset fidelity
        assert 0 <= c.start <= c.end <= len(text)
        assert len(c.text) <= chunk_chars # within budget
        assert c.start < c.end # forward progress (no empty chunk)
    assert chunks[0].start == 0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.start < nxt.start # strictly advancing
        assert nxt.start <= prev.end # no gap (overlap or abut)
    if not result.truncated:
        assert chunks[-1].end == len(text) # full coverage when not capped


# --------------------------------------------------------------------------- #
# derive_chunk_chars
# --------------------------------------------------------------------------- #
def test_derive_chunk_chars_normal():
    # (16384 - 2048) * 3.0
    assert derive_chunk_chars(16384, prompt_reserve_tokens=2048, chars_per_token=3.0) == 43008


@pytest.mark.parametrize(
    "ctx,reserve,cpt",
    [
        (0, 100, 3.0), # ctx_size <= 0
        (-5, 0, 3.0),
        (1000, -1, 3.0), # reserve < 0
        (1000, 0, 0), # cpt <= 0
        (1000, 0, -1.0),
        (1000, 1000, 3.0), # reserve == ctx -> no room
        (1000, 2000, 3.0), # reserve > ctx
        (10, 9, 0.05), # budget 1 token * 0.05 -> int 0 -> ValueError
    ],
)
def test_derive_chunk_chars_invalid(ctx, reserve, cpt):
    with pytest.raises(ValueError):
        derive_chunk_chars(ctx, prompt_reserve_tokens=reserve, chars_per_token=cpt)


# --------------------------------------------------------------------------- #
# derive_overlap
# --------------------------------------------------------------------------- #
def test_derive_overlap_normal():
    assert derive_overlap(1000, ratio=0.1) == 100
    assert derive_overlap(1000, ratio=0.0) == 0
    assert derive_overlap(1, ratio=0.1) == 0 # floor keeps it < chunk_chars
    # overlap is always strictly < chunk_chars (precondition for progress)
    for cc in (1, 2, 7, 100, 9999):
        assert derive_overlap(cc, ratio=0.999) < cc


@pytest.mark.parametrize("cc,ratio", [(0, 0.1), (-1, 0.1), (100, 1.0), (100, 1.5), (100, -0.1)])
def test_derive_overlap_invalid(cc, ratio):
    with pytest.raises(ValueError):
        derive_overlap(cc, ratio=ratio)


# --------------------------------------------------------------------------- #
# chunk_text — normal class
# --------------------------------------------------------------------------- #
def test_multi_paragraph_full_coverage():
    paras = ["Paragraph %d. %s" % (i, "word " * 40) for i in range(20)]
    text = "\n\n".join(paras)
    res = chunk_text(text, chunk_chars=300, overlap=30, max_chunks=64)
    assert not res.truncated
    assert len(res.chunks) > 1
    _assert_invariants(text, res, chunk_chars=300)


def test_tiny_text_single_chunk():
    text = "just a short note"
    res = chunk_text(text, chunk_chars=300, overlap=30, max_chunks=64)
    assert len(res.chunks) == 1
    assert res.chunks[0].text == text
    assert res.chunks[0].start == 0 and res.chunks[0].end == len(text)
    assert not res.truncated


def test_text_exactly_chunk_size_single_chunk():
    text = "x" * 100
    res = chunk_text(text, chunk_chars=100, overlap=10, max_chunks=64)
    assert len(res.chunks) == 1
    assert res.chunks[0].end == 100
    assert not res.truncated


def test_overlap_region_is_shared():
    text = "abcdefghij " * 50 # spaces present so boundaries exist
    res = chunk_text(text, chunk_chars=120, overlap=40, max_chunks=64)
    assert len(res.chunks) >= 2
    _assert_invariants(text, res, chunk_chars=120)
    # at least one consecutive pair genuinely overlaps, and the shared region matches
    overlapped = False
    for prev, nxt in zip(res.chunks, res.chunks[1:]):
        if nxt.start < prev.end:
            overlapped = True
            assert text[nxt.start : prev.end] == prev.text[nxt.start - prev.start :]
    assert overlapped


def test_prefers_paragraph_boundary():
    # a clean paragraph break sits inside the window; the chunk should end at it
    head = "A" * 200
    tail = "B" * 200
    text = head + "\n\n" + tail
    res = chunk_text(text, chunk_chars=260, overlap=0, max_chunks=64)
    assert res.chunks[0].end == len(head) + 2 # break right after "\n\n"
    assert res.chunks[0].text == head + "\n\n"
    _assert_invariants(text, res, chunk_chars=260)


def test_zero_overlap_partitions_exactly():
    text = "z" * 1000 # no boundaries -> hard cuts of 250
    res = chunk_text(text, chunk_chars=250, overlap=0, max_chunks=64)
    assert len(res.chunks) == 4
    assert [c.start for c in res.chunks] == [0, 250, 500, 750]
    assert "".join(c.text for c in res.chunks) == text # exact partition, no overlap
    _assert_invariants(text, res, chunk_chars=250)


def test_determinism():
    text = ("Lorem ipsum dolor sit amet. " * 200) + "\n\n" + ("Foo bar baz. " * 200)
    a = chunk_text(text, chunk_chars=333, overlap=37, max_chunks=64)
    b = chunk_text(text, chunk_chars=333, overlap=37, max_chunks=64)
    assert a == b
    assert [(c.start, c.end, c.text) for c in a.chunks] == [(c.start, c.end, c.text) for c in b.chunks]


# --------------------------------------------------------------------------- #
# chunk_text — error / edge class
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", ["", " ", "\n\n\n", "\t \t", " \n \t \n "])
def test_empty_or_whitespace_yields_zero_chunks(text):
    res = chunk_text(text, chunk_chars=100, overlap=10, max_chunks=64)
    assert res.chunks == ()
    assert not res.truncated
    assert res.total_chars == len(text)


def test_huge_text_is_capped_and_flagged():
    text = "w" * 100_000 # would need 1000 chunks of 100
    res = chunk_text(text, chunk_chars=100, overlap=0, max_chunks=10)
    assert len(res.chunks) == 10
    assert res.truncated
    assert res.chunks[-1].end < len(text) # tail genuinely dropped
    _assert_invariants(text, res, chunk_chars=100)


def test_single_mega_token_no_boundaries_terminates():
    # one "word" far longer than the budget, no whitespace at all -> hard cuts only
    text = "Z" * 5000
    res = chunk_text(text, chunk_chars=128, overlap=16, max_chunks=64)
    assert len(res.chunks) > 1
    _assert_invariants(text, res, chunk_chars=128)


def test_overlap_near_chunk_size_still_progresses():
    # overlap one below chunk_chars: the guard must keep advancing (no infinite loop)
    text = "Y" * 3000
    res = chunk_text(text, chunk_chars=100, overlap=99, max_chunks=200)
    assert len(res.chunks) > 1
    _assert_invariants(text, res, chunk_chars=100)


def test_chunk_chars_one_terminates():
    text = "abcde fghij"
    res = chunk_text(text, chunk_chars=1, overlap=0, max_chunks=1000)
    assert len(res.chunks) == len(text)
    assert all(len(c.text) == 1 for c in res.chunks)
    _assert_invariants(text, res, chunk_chars=1)


def test_unicode_terminates():
    text = ("한국어 문서 테스트. " * 100) + "🙂🙂🙂\n\n" + ("混合内容 " * 100)
    res = chunk_text(text, chunk_chars=90, overlap=9, max_chunks=64)
    assert res.chunks
    _assert_invariants(text, res, chunk_chars=90)


@pytest.mark.parametrize(
    "chunk_chars,overlap,max_chunks",
    [
        (0, 0, 10), # chunk_chars <= 0
        (-5, 0, 10),
        (100, -1, 10), # overlap < 0
        (100, 100, 10), # overlap >= chunk_chars
        (100, 150, 10),
        (100, 10, 0), # max_chunks <= 0
        (100, 10, -1),
    ],
)
def test_invalid_params_raise(chunk_chars, overlap, max_chunks):
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_chars=chunk_chars, overlap=overlap, max_chunks=max_chunks)


def test_chunk_is_frozen():
    c = Chunk(index=0, text="x", start=0, end=1)
    with pytest.raises(Exception):
        c.index = 5 # type: ignore[misc]
