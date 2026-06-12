"""File-ingestion containment core — the SECURITY CRUX.

A gated upload accepts an **UNTRUSTED** browser file and must land it safely inside a
dedicated ingest area under `docs_root` (DF1: `docs_root/_ingest/<id>/<name>`), referenced
later by an opaque id (DF3). This module is the pure, fail-closed core that decides what is
allowed and where it goes; the FastAPI endpoint only drives it. No network, no model.

Containment is layered (mirrors `docs_root` + `ExecPolicy`):
1. **validate** the upload BEFORE any byte is written — type allowlist · size cap · filename
   **canonicalization** (a brand-new safe name derived from the basename; never reflect
   attacker path structure) · reject `..`/absolute/NUL/control/dotfile/empty.
2. **resolve + confine** — the destination is `_ingest/<server-generated id>/<safe-name>`;
   a final `resolve()` + `is_relative_to(ingest_root)` re-check catches any traversal that
   somehow survived canonicalization, and a **no-overwrite** guard never clobbers a file.
3. aggregate **caps** — count + total-byte (DoS bound), checked against the live store.

Every rejection is a typed `IngestError` with a generic message (no absolute path, no raw
stack, no attacker-filename echo). The id↔contained-path map lives in `IngestStore` (DF1
persistent: rebuildable by scanning `_ingest/` on start; never hands a caller an absolute path).
"""
from __future__ import annotations

import asyncio
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import loaders

# The default ingest area lives under docs_root; the leading underscore keeps it visually
# distinct and `list_documents` already skips dot-prefixed entries (this is not dot-prefixed,
# so ingested docs remain listable by DocQA — intentional).
DEFAULT_INGEST_SUBDIR = "_ingest"

_MAX_NAME_LEN = 255
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}") # server-generated ids only ever match this


# --------------------------------------------------------------------------- #
# errors — every rejection is a typed IngestError (generic message, no leak)
# --------------------------------------------------------------------------- #
class IngestError(Exception):
    """Base for all ingestion failures."""


class UnsupportedType(IngestError):
    """The file extension is not in the allowlist."""


class TooLarge(IngestError):
    """The file exceeds the per-file byte cap."""


class BadFilename(IngestError):
    """The filename is unsafe (traversal / absolute / NUL / control / dotfile / empty / too long)."""


class TooMany(IngestError):
    """Storing this file would exceed the file-count cap."""


class QuotaExceeded(IngestError):
    """Storing this file would exceed the total-byte cap."""


class ContainmentError(IngestError):
    """The resolved destination escapes the ingest root, or already exists (no overwrite)."""


class UnknownIngestId(IngestError):
    """No stored file is registered under this id (unknown / expired / malformed)."""


# --------------------------------------------------------------------------- #
# the prepared (validated) upload + a stored record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreparedIngest:
    ingest_id: str # opaque, server-generated
    stored_name: str # safe basename (canonicalized, lowercase ext)
    relpath: str # posix path relative to docs_root: "<subdir>/<id>/<name>"
    declared_size: int
    ext: str


@dataclass(frozen=True)
class IngestRecord:
    ingest_id: str
    relpath: str
    stored_name: str
    size: int
    ext: str


# --------------------------------------------------------------------------- #
# the policy — pure validation + confinement (fail-closed)
# --------------------------------------------------------------------------- #
class IngestPolicy:
    """Decides what may be ingested and where it is stored. Pure: no I/O except the final
    `resolve()`/`exists()` containment re-check (and that only reads). Inject `id_factory` for
    deterministic tests."""

    def __init__(
        self,
        docs_root: Path | str,
        *,
        ingest_subdir: str = DEFAULT_INGEST_SUBDIR,
        max_file_bytes: int = 10_000_000,
        max_files: int = 50,
        max_total_bytes: int = 200_000_000,
        allowed_extensions: frozenset[str] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_file_bytes <= 0 or max_files <= 0 or max_total_bytes <= 0:
            raise ValueError("ingest caps must be > 0")
        self._docs_root = Path(docs_root).resolve()
        self.ingest_subdir = ingest_subdir
        self._ingest_root = (self._docs_root / ingest_subdir).resolve()
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        # `.hwpx` is already in loaders._SUPPORTED; reuse the loader-supported set.
        self._allowed = frozenset(allowed_extensions or loaders._SUPPORTED)
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    @property
    def docs_root(self) -> Path:
        return self._docs_root

    @property
    def ingest_root(self) -> Path:
        return self._ingest_root

    # -- filename canonicalization (derive a brand-new safe name) ------------- #
    def _safe_stored_name(self, filename: object) -> tuple[str, str]:
        if not isinstance(filename, str):
            raise BadFilename("filename must be a string")
        if "\x00" in filename:
            raise BadFilename("filename contains NUL")
        name = unicodedata.normalize("NFC", filename)
        # strip EVERY directory component for both separator styles, then take the basename
        name = name.replace("\\", "/").split("/")[-1].strip()
        if name in ("", ".", ".."):
            raise BadFilename("empty or relative filename")
        if name.startswith("."):
            raise BadFilename("hidden/dotfile name not allowed")
        if any(ord(c) < 32 for c in name):
            raise BadFilename("filename contains control characters")
        if len(name) > _MAX_NAME_LEN:
            raise BadFilename("filename too long")
        suffix = Path(name).suffix # the LAST extension (so x.txt.exe → .exe)
        ext = suffix.lower()
        if ext not in self._allowed:
            raise UnsupportedType(f"unsupported type: {ext!r}")
        stem = name[: len(name) - len(suffix)]
        stored = f"{stem}{ext}" # normalize the extension to lowercase
        return stored, ext

    # -- per-file validation -------------------------------------------------- #
    def validate(self, filename: object, declared_size: int) -> PreparedIngest:
        """Validate a single upload (type · size · filename). Returns a `PreparedIngest`
        (safe name + fresh opaque id + contained relpath) or raises a typed `IngestError`."""
        if not isinstance(declared_size, int) or declared_size < 0:
            raise TooLarge("invalid size")
        if declared_size > self.max_file_bytes:
            raise TooLarge(f"file exceeds {self.max_file_bytes} bytes")
        stored_name, ext = self._safe_stored_name(filename)
        ingest_id = self._id_factory()
        if not isinstance(ingest_id, str) or not _ID_RE.fullmatch(ingest_id):
            raise ContainmentError("invalid ingest id") # guards a misbehaving id_factory
        relpath = f"{self.ingest_subdir}/{ingest_id}/{stored_name}"
        return PreparedIngest(
            ingest_id=ingest_id, stored_name=stored_name, relpath=relpath,
            declared_size=declared_size, ext=ext,
        )

    # -- destination resolution + confinement (defense-in-depth) -------------- #
    def resolve_destination(self, prepared: PreparedIngest) -> Path:
        """Absolute write path under `_ingest/<id>/<name>`, re-checked for containment and
        no-overwrite. A traversal that survived canonicalization is still caught here."""
        dest = self._ingest_root / prepared.ingest_id / prepared.stored_name
        resolved = dest.resolve()
        if not resolved.is_relative_to(self._ingest_root):
            raise ContainmentError("destination escapes the ingest root")
        if resolved.exists():
            raise ContainmentError("destination already exists") # never overwrite
        return resolved

    # -- aggregate caps ------------------------------------------------------- #
    def check_admission(self, current_count: int, current_total_bytes: int, new_size: int) -> None:
        """Raise if storing a `new_size`-byte file would breach the count or total-byte cap."""
        if current_count + 1 > self.max_files:
            raise TooMany(f"too many ingested files (> {self.max_files})")
        if current_total_bytes + new_size > self.max_total_bytes:
            raise QuotaExceeded(f"total ingest bytes exceed {self.max_total_bytes}")


# --------------------------------------------------------------------------- #
# the store — id ↔ contained relpath (DF1 persistent; never leaks an abs path)
# --------------------------------------------------------------------------- #
class IngestStore:
    """In-process id→record map over the on-disk `_ingest/<id>/` layout. `lock` serializes
    the endpoint's admit→write→register critical section. `resolve` returns only a
    `docs_root`-relative path — never an absolute filesystem path."""

    def __init__(self, policy: IngestPolicy) -> None:
        self._policy = policy
        self._records: dict[str, IngestRecord] = {}
        self.lock = asyncio.Lock()

    @property
    def policy(self) -> IngestPolicy:
        return self._policy

    def count(self) -> int:
        return len(self._records)

    def total_bytes(self) -> int:
        return sum(r.size for r in self._records.values())

    def register(self, prepared: PreparedIngest, actual_size: int) -> IngestRecord:
        rec = IngestRecord(
            ingest_id=prepared.ingest_id, relpath=prepared.relpath,
            stored_name=prepared.stored_name, size=actual_size, ext=prepared.ext,
        )
        self._records[prepared.ingest_id] = rec
        return rec

    def resolve(self, ingest_id: object) -> str:
        """The contained `docs_root`-relative path for `ingest_id`, or `UnknownIngestId`."""
        if not isinstance(ingest_id, str) or ingest_id not in self._records:
            raise UnknownIngestId("unknown attachment id")
        return self._records[ingest_id].relpath

    def get(self, ingest_id: object) -> IngestRecord:
        if not isinstance(ingest_id, str) or ingest_id not in self._records:
            raise UnknownIngestId("unknown attachment id")
        return self._records[ingest_id]

    def load_text(self, ingest_id: object, *, max_chars: int | None = None) -> str:
        """Plain text of an ingested file, read through `load_document` (which `resolve`s
        + rejects symlink/`..` escapes — the canonical contained read path). Bounded to the policy's
        per-file byte cap and optionally truncated to `max_chars`. `UnknownIngestId` for a bad id;
        a `DocError` for an unreadable/oversized/unsupported file (the caller maps it to a status)."""
        rec = self.get(ingest_id) # raises UnknownIngestId
        text = loaders.load_document(
            self._policy.docs_root, rec.relpath, max_bytes=self._policy.max_file_bytes,
        )
        return text[:max_chars] if max_chars else text

    def read_bytes(self, ingest_id: object) -> bytes:
        """Raw bytes of an ingested file (for the multimodal path — images/PDFs passed to the model,
        not text-extracted). Read through the `resolve_within` containment (rejects symlink/`..`
        escapes), bounded by the policy's per-file cap. `UnknownIngestId` for a bad id; `DocError` on
        an oversized/unreadable file."""
        rec = self.get(ingest_id) # raises UnknownIngestId
        path = loaders.resolve_within(self._policy.docs_root, rec.relpath) # contained
        cap = self._policy.max_file_bytes
        try:
            with open(path, "rb") as fh:
                data = fh.read(cap + 1) # bounded read (the stored file is ≤ cap, but never trust it)
        except OSError as exc:
            raise loaders.DocLoadError("could not read ingested file") from exc
        if len(data) > cap:
            raise loaders.DocTooLarge(f"ingested file exceeds {cap} bytes")
        return data

    def list_records(self) -> list[IngestRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def rebuild_from_disk(self) -> None:
        """Repopulate from the persistent `_ingest/<id>/<file>` layout (DF1 — ids survive a
        restart). Tolerant: a malformed id dir / multi-file dir / unreadable file is skipped,
        never raising."""
        self._records.clear()
        root = self._policy.ingest_root
        try:
            if not root.is_dir():
                return
            children = sorted(root.iterdir())
        except OSError:
            return
        for child in children:
            try:
                # parity with resolve_destination's symlink discipline: a symlinked id-dir or
                # file is skipped (never registered), so a relpath that could resolve outside
                # docs_root is never produced — defense-in-depth atop the read-path resolve_within.
                if child.is_symlink() or not child.is_dir() or not _ID_RE.fullmatch(child.name):
                    continue
                files = [p for p in child.iterdir() if p.is_file() and not p.is_symlink()]
                if len(files) != 1:
                    continue
                f = files[0]
                size = f.stat().st_size
            except OSError:
                continue
            relpath = f"{self._policy.ingest_subdir}/{child.name}/{f.name}"
            self._records[child.name] = IngestRecord(
                ingest_id=child.name, relpath=relpath, stored_name=f.name,
                size=size, ext=Path(f.name).suffix.lower(),
            )
