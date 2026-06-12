"""— ingest containment crux (IngestPolicy + IngestStore).

The SECURITY boundary: an UNTRUSTED filename + size is turned into a safe, contained
destination, or refused with a typed IngestError. These tests are the adversarial battery —
traversal / wrong-type / oversized / count+total caps / overwrite / containment / bad id —
plus the id↔path store roundtrip. Pure (no model, no network); a deterministic id_factory.
"""
from __future__ import annotations

import itertools

import pytest

from local_ai_agent.modules.docqa.ingest import (
    BadFilename,
    ContainmentError,
    IngestPolicy,
    IngestStore,
    QuotaExceeded,
    TooLarge,
    TooMany,
    UnknownIngestId,
    UnsupportedType,
)


def _counter_ids():
    c = itertools.count()
    return lambda: f"id{next(c)}"


def _policy(tmp_path, **kw):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    defaults = dict(
        max_file_bytes=1_000_000, max_files=3, max_total_bytes=2_000_000,
        id_factory=_counter_ids(),
    )
    defaults.update(kw)
    return IngestPolicy(docs, **defaults)


# --------------------------------------------------------------------------- #
# normal
# --------------------------------------------------------------------------- #
def test_valid_upload_prepares_contained_destination(tmp_path):
    pol = _policy(tmp_path)
    prep = pol.validate("report.pdf", 1234)
    assert prep.stored_name == "report.pdf"
    assert prep.ext == ".pdf"
    assert prep.relpath == f"_ingest/{prep.ingest_id}/report.pdf"
    dest = pol.resolve_destination(prep)
    assert dest.is_relative_to(pol.ingest_root)
    assert not dest.exists()


@pytest.mark.parametrize("name,stored,ext", [
    ("Notes.TXT", "Notes.txt", ".txt"),
    ("a.DOCX", "a.docx", ".docx"),
    ("paper.HwpX", "paper.hwpx", ".hwpx"),
    ("page.HTML", "page.html", ".html"),
])
def test_extension_normalized_lowercase(tmp_path, name, stored, ext):
    prep = _policy(tmp_path).validate(name, 10)
    assert prep.stored_name == stored and prep.ext == ext


def test_store_register_resolve_roundtrip(tmp_path):
    pol = _policy(tmp_path)
    store = IngestStore(pol)
    prep = pol.validate("doc.txt", 100)
    store.register(prep, 100)
    assert store.resolve(prep.ingest_id) == prep.relpath
    assert store.count() == 1 and store.total_bytes() == 100


def test_admission_passes_under_caps(tmp_path):
    pol = _policy(tmp_path)
    pol.check_admission(current_count=0, current_total_bytes=0, new_size=500) # no raise


# --------------------------------------------------------------------------- #
# error / security — traversal & bad filenames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "../../etc/passwd.txt", "/etc/passwd.txt", "a/b.txt", "..\\..\\win.txt",
    "../secret.pdf", "sub/dir/file.txt", "\\\\server\\share\\f.txt",
])
def test_traversal_filenames_stripped_or_rejected(tmp_path, bad):
    pol = _policy(tmp_path)
    # either rejected, or the directory structure is fully stripped (no separators survive)
    try:
        prep = pol.validate(bad, 10)
    except (BadFilename, UnsupportedType):
        return
    assert "/" not in prep.stored_name and "\\" not in prep.stored_name
    assert ".." not in prep.stored_name.split("/")
    dest = pol.resolve_destination(prep)
    assert dest.is_relative_to(pol.ingest_root)


@pytest.mark.parametrize("bad", ["", ".", "..", " ", "\x00evil.txt", ".env", ".secret.txt",
                                 "a\tb.txt", "a\nb.txt", "x" * 300 + ".txt"])
def test_bad_filenames_rejected(tmp_path, bad):
    with pytest.raises((BadFilename, UnsupportedType)):
        _policy(tmp_path).validate(bad, 10)


def test_non_string_filename_rejected(tmp_path):
    with pytest.raises(BadFilename):
        _policy(tmp_path).validate(1234, 10)


# --------------------------------------------------------------------------- #
# error / security — type allowlist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["malware.exe", "legacy.hwp", "README", "archive.zip",
                                 "script.sh", "x.txt.exe", "photo.png", "a.docx.bin"])
def test_unsupported_types_rejected(tmp_path, bad):
    with pytest.raises(UnsupportedType):
        _policy(tmp_path).validate(bad, 10)


# --------------------------------------------------------------------------- #
# error / security — size & caps
# --------------------------------------------------------------------------- #
def test_oversized_rejected(tmp_path):
    pol = _policy(tmp_path, max_file_bytes=1000)
    with pytest.raises(TooLarge):
        pol.validate("big.pdf", 1001)


@pytest.mark.parametrize("size", [-1, -1000])
def test_negative_size_rejected(tmp_path, size):
    with pytest.raises(TooLarge):
        _policy(tmp_path).validate("a.txt", size)


def test_count_cap(tmp_path):
    pol = _policy(tmp_path, max_files=2)
    pol.check_admission(current_count=1, current_total_bytes=0, new_size=1) # ok (→2)
    with pytest.raises(TooMany):
        pol.check_admission(current_count=2, current_total_bytes=0, new_size=1)


def test_total_byte_cap(tmp_path):
    pol = _policy(tmp_path, max_total_bytes=1000)
    pol.check_admission(current_count=0, current_total_bytes=900, new_size=100) # ok (=1000)
    with pytest.raises(QuotaExceeded):
        pol.check_admission(current_count=0, current_total_bytes=901, new_size=100)


# --------------------------------------------------------------------------- #
# error / security — overwrite & containment
# --------------------------------------------------------------------------- #
def test_same_name_twice_gets_distinct_dirs(tmp_path):
    pol = _policy(tmp_path)
    a = pol.validate("dup.txt", 10)
    b = pol.validate("dup.txt", 10)
    assert a.ingest_id != b.ingest_id
    assert pol.resolve_destination(a) != pol.resolve_destination(b)


def test_no_overwrite_existing_file(tmp_path):
    pol = _policy(tmp_path, id_factory=lambda: "fixedid")
    prep = pol.validate("doc.txt", 10)
    dest = pol.resolve_destination(prep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("already here")
    # a second resolve to the SAME (id,name) destination must refuse to overwrite
    prep2 = pol.validate("doc.txt", 10)
    with pytest.raises(ContainmentError):
        pol.resolve_destination(prep2)


def test_misbehaving_id_factory_rejected(tmp_path):
    pol = _policy(tmp_path, id_factory=lambda: "../escape")
    with pytest.raises(ContainmentError):
        pol.validate("doc.txt", 10)


def test_resolve_destination_always_contained(tmp_path):
    pol = _policy(tmp_path)
    for name in ["a.txt", "weird....txt", " spaced .pdf", "ünïcødé.docx", "CON.txt"]:
        try:
            prep = pol.validate(name, 10)
        except (BadFilename, UnsupportedType):
            continue
        assert pol.resolve_destination(prep).is_relative_to(pol.ingest_root)


# --------------------------------------------------------------------------- #
# store — unknown id & rebuild
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["nope", "", "../x", 123, None])
def test_resolve_unknown_id(tmp_path, bad):
    store = IngestStore(_policy(tmp_path))
    with pytest.raises(UnknownIngestId):
        store.resolve(bad)


def test_rebuild_from_disk(tmp_path):
    pol = _policy(tmp_path)
    # lay down a persistent _ingest/<id>/<file> + a malformed sibling that must be skipped
    (pol.ingest_root / "abc123").mkdir(parents=True)
    (pol.ingest_root / "abc123" / "kept.txt").write_text("hi")
    (pol.ingest_root / "two_files").mkdir()
    (pol.ingest_root / "two_files" / "a.txt").write_text("x")
    (pol.ingest_root / "two_files" / "b.txt").write_text("y") # 2 files → skipped
    store = IngestStore(pol)
    store.rebuild_from_disk()
    assert store.resolve("abc123") == "_ingest/abc123/kept.txt"
    with pytest.raises(UnknownIngestId):
        store.resolve("two_files")


def test_rebuild_skips_symlinked_dirs_and_files(tmp_path):
    # parity with the write path: a symlinked id-dir (pointing outside) or a symlinked file
    # must NOT be registered (so no relpath that could resolve outside docs_root is produced).
    pol = _policy(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "leak.txt").write_text("SECRET")
    pol.ingest_root.mkdir(parents=True, exist_ok=True)
    # (a) a symlinked id-dir → skipped
    (pol.ingest_root / "evil1").symlink_to(outside, target_is_directory=True)
    # (b) a real id-dir whose single file is a symlink to outside → skipped
    real = pol.ingest_root / "real01"
    real.mkdir()
    (real / "doc.txt").symlink_to(outside / "leak.txt")
    store = IngestStore(pol)
    store.rebuild_from_disk()
    assert store.count() == 0
    for bad in ("evil1", "real01"):
        with pytest.raises(UnknownIngestId):
            store.resolve(bad)


def test_rebuild_missing_root_is_noop(tmp_path):
    pol = _policy(tmp_path) # _ingest/ does not exist yet
    store = IngestStore(pol)
    store.rebuild_from_disk() # no raise
    assert store.count() == 0
