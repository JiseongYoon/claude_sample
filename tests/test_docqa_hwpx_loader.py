"""— HWPX loader (stdlib zip+XML, bounded, XXE-off).

The only NEW document format of . HWPX is the open OWPML zip; body text lives in
`Contents/sectionN.xml`. These tests build tiny HWPX zips in-memory (no committed binary),
and cover: extraction + ordering, extension registration, `.hwp`→UnsupportedFormat, and the
adversarial bounds (zip-bomb member-count + decompressed-size, XXE, not-a-zip, no-sections,
corrupt XML). No network, no real model.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from local_ai_agent.modules.docqa import loaders
from local_ai_agent.modules.docqa.loaders import (
    DocError,
    DocLoadError,
    DocTooLarge,
    UnsupportedFormat,
    is_supported,
    load_bytes,
    load_document,
)

_HP = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph"'


def _section(*paragraphs: str) -> str:
    body = "".join(
        f"<hp:p><hp:run><hp:t>{p}</hp:t></hp:run></hp:p>" for p in paragraphs
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><hp:sec {_HP}>{body}</hp:sec>'


def _make_hwpx(sections: list[str] | dict[str, str], extra: dict[str, str] | None = None) -> bytes:
    if isinstance(sections, list):
        sections = {f"Contents/section{i}.xml": s for i, s in enumerate(sections)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/hwp+zip")
        for name, content in sections.items():
            zf.writestr(name, content)
        for name, content in (extra or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# normal
# --------------------------------------------------------------------------- #
def test_hwpx_extracts_body_text():
    data = _make_hwpx([_section("Hello", "World")])
    text = load_bytes("doc.hwpx", data)
    assert "Hello" in text and "World" in text


def test_hwpx_multi_run_joined_within_paragraph():
    # two runs in one paragraph → joined without an internal break
    para = "<hp:p><hp:run><hp:t>Foo</hp:t></hp:run><hp:run><hp:t>Bar</hp:t></hp:run></hp:p>"
    sec = f'<?xml version="1.0"?><hp:sec {_HP}>{para}</hp:sec>'
    text = load_bytes("doc.hwpx", _make_hwpx([sec]))
    assert "FooBar" in text


def test_hwpx_section_order_preserved():
    data = _make_hwpx([_section("FIRST"), _section("SECOND")])
    # section10 must NOT sort before section2 (numeric, not lexical)
    text = load_bytes("doc.hwpx", data)
    assert text.index("FIRST") < text.index("SECOND")


def test_hwpx_numeric_section_ordering():
    data = _make_hwpx(
        {
            "Contents/section0.xml": _section("ZERO"),
            "Contents/section2.xml": _section("TWO"),
            "Contents/section10.xml": _section("TEN"),
        }
    )
    text = load_bytes("doc.hwpx", data)
    assert text.index("ZERO") < text.index("TWO") < text.index("TEN")


def test_hwpx_extension_registered():
    assert is_supported("anything.hwpx") is True
    assert ".hwpx" in loaders._SUPPORTED
    assert ".hwpx" in loaders._LOADERS
    assert ".hwpx" in loaders._BYTES_LOADERS


def test_hwpx_via_load_document(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "report.hwpx").write_bytes(_make_hwpx([_section("Contained")]))
    assert "Contained" in load_document(root, "report.hwpx")


# --------------------------------------------------------------------------- #
# error / security
# --------------------------------------------------------------------------- #
def test_legacy_hwp_is_unsupported():
    # legacy binary .hwp must NOT route to the HWPX loader
    with pytest.raises(UnsupportedFormat):
        load_bytes("legacy.hwp", b"\xd0\xcf\x11\xe0arbitrary")


def test_legacy_hwp_unsupported_via_path(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "legacy.hwp").write_bytes(b"\xd0\xcf\x11\xe0")
    with pytest.raises(UnsupportedFormat):
        load_document(root, "legacy.hwp")


def test_hwpx_member_count_bomb():
    extra = {f"Contents/junk{i}.bin": b"x" for i in range(loaders._HWPX_MAX_MEMBERS + 5)}
    data = _make_hwpx([_section("hi")], extra={k: "x" for k in extra})
    with pytest.raises(DocTooLarge):
        load_bytes("bomb.hwpx", data)


def test_hwpx_decompressed_size_bound(monkeypatch):
    # the bounded read must reject content larger than the budget WITHOUT trusting headers
    monkeypatch.setattr(loaders, "_DEFAULT_MAX_BYTES", 64)
    big_para = "A" * 5000
    data = _make_hwpx([_section(big_para)])
    with pytest.raises(DocTooLarge):
        load_bytes("big.hwpx", data)


def test_hwpx_xxe_no_entity_resolution(tmp_path):
    # an external-entity reference must NOT be resolved (no file/network fetch) — ElementTree
    # raises on the undefined entity → typed DocLoadError, and the secret never appears.
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-XXE")
    section = (
        '<?xml version="1.0"?>'
        f'<!DOCTYPE sec [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>'
        f'<hp:sec {_HP}><hp:p><hp:run><hp:t>&xxe;</hp:t></hp:run></hp:p></hp:sec>'
    )
    with pytest.raises(DocError) as ei:
        load_bytes("xxe.hwpx", _make_hwpx([section]))
    assert "TOP-SECRET-XXE" not in str(ei.value)


def test_hwpx_not_a_zip():
    with pytest.raises(DocLoadError):
        load_bytes("fake.hwpx", b"this is not a zip at all")


def test_hwpx_no_sections():
    data = _make_hwpx({"Contents/header.xml": "<x/>"})
    with pytest.raises(DocLoadError):
        load_bytes("empty.hwpx", data)


def test_hwpx_corrupt_section_xml():
    data = _make_hwpx({"Contents/section0.xml": "<hp:sec><unclosed>"})
    with pytest.raises(DocLoadError):
        load_bytes("corrupt.hwpx", data)


def test_hwpx_empty_bytes():
    with pytest.raises(DocLoadError):
        load_bytes("empty.hwpx", b"")


def test_other_formats_unaffected():
    # quick regression that the dispatch tables still serve the existing formats
    assert "plain" in load_bytes("a.txt", b"plain text")
    assert "Title" in load_bytes("a.html", b"<html><body><h1>Title</h1></body></html>")
