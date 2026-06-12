"""Storage connector seam + the GuardedConnector policy layer.

Two layers, so security policy can never be forgotten per connector:

  * **`RemoteTransport`** — the narrow *raw* interface a concrete connector implements (SSH/SFTP
    in , Synology in 5.1). It operates on already-validated ABSOLUTE remote paths and is
    expected to raise the typed `StorageError`s below (never a raw library exception that could
    embed a credential).
  * **`GuardedConnector`** — wraps a transport and enforces policy on every call BEFORE delegating:
    remote-path containment to a configured `allowed_root`, read-only mode, and a read size cap.
    Tools talk only to this; they never hold a raw transport, so there is no ungated path.

Remote containment is **lexical** (`posixpath.normpath` + prefix check) because a remote symlink
cannot be resolved without a network round-trip. An optional `realpath` hook lets a concrete
transport ( SFTP `realpath`) resolve server-side and re-check — closing the symlink-escape
gap that lexical containment alone leaves. Read-only-by-default further bounds the blast radius.
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# value type + typed errors (never leak a credential in a message)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RemoteEntry:
    """One remote filesystem entry. `path` is allowed-root-RELATIVE (POSIX) at the
    `GuardedConnector` boundary; transports return it root-ABSOLUTE (relativized by the guard)."""

    path: str
    is_dir: bool
    size: int = 0


class StorageError(Exception):
    """Base for all storage failures."""


class StorageAccessError(StorageError):
    """Path escapes `allowed_root` (`..`/absolute/symlink), or a non-permitted operation."""


class StorageNotFound(StorageError):
    """No such remote path."""


class StorageReadOnly(StorageError):
    """A mutating op was attempted on a read-only connector."""


class StorageTooLarge(StorageError):
    """A read exceeds the configured byte cap."""


class StorageAuthError(StorageError):
    """Authentication failed (message must NOT contain the credential)."""


class StorageUnavailable(StorageError):
    """The connector/transport is unreachable or errored."""


# --------------------------------------------------------------------------- #
# the raw transport seam (implemented by concrete connectors)
# --------------------------------------------------------------------------- #
@runtime_checkable
class RemoteTransport(Protocol):
    """Raw remote ops on ABSOLUTE, already-contained paths. Implementations raise the typed
    `StorageError`s (e.g. `StorageNotFound`, `StorageAuthError`, `StorageUnavailable`) — never a
    raw library exception that could embed host/credential detail."""

    async def list_dir(self, path: str) -> list[RemoteEntry]: ... # entries with ABSOLUTE paths
    async def stat(self, path: str) -> RemoteEntry: ...
    async def read_bytes(self, path: str) -> bytes: ...
    async def write_bytes(self, path: str, data: bytes) -> None: ...
    async def delete(self, path: str) -> None: ...
    async def move(self, src: str, dst: str) -> None: ...
    async def realpath(self, path: str) -> str: ... # server-side symlink resolve
    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# lexical containment
# --------------------------------------------------------------------------- #
def _norm_abs(p: str) -> str:
    if not isinstance(p, str) or not p.strip():
        raise StorageAccessError("remote path must be a non-empty string")
    n = posixpath.normpath(p)
    if not n.startswith("/"):
        raise StorageAccessError(f"remote path must be absolute: {p!r}")
    return n


def _within(root: str, path: str) -> bool:
    if path == root:
        return True
    prefix = root if root.endswith("/") else root + "/"
    return path.startswith(prefix)


def contain_remote(allowed_root: str, relpath: str) -> str:
    """Join `relpath` under `allowed_root` and require lexical containment. An absolute `relpath`
    (which resets the join) or a `..` escape resolves outside → `StorageAccessError`. Returns the
    contained ABSOLUTE remote path."""
    root = _norm_abs(allowed_root)
    if not isinstance(relpath, str):
        raise StorageAccessError("relpath must be a string")
    target = posixpath.normpath(posixpath.join(root, relpath))
    if not _within(root, target):
        raise StorageAccessError(f"path escapes allowed_root: {relpath!r}")
    return target


def _relativize(root: str, abspath: str) -> str:
    """ABSOLUTE remote path → allowed-root-relative (root itself → '.'). Defensive: an entry
    outside root (shouldn't occur) → `StorageAccessError`."""
    ap = _norm_abs(abspath)
    if ap == root:
        return "."
    prefix = root if root.endswith("/") else root + "/"
    if not ap.startswith(prefix):
        raise StorageAccessError(f"transport returned a path outside allowed_root: {abspath!r}")
    return ap[len(prefix):]


# --------------------------------------------------------------------------- #
# the policy layer — the single gated path to a transport
# --------------------------------------------------------------------------- #
RealpathHook = Callable[[str], Awaitable[str]]


class GuardedConnector:
    """Enforces containment + read-only + size cap around a `RemoteTransport`. All public methods
    take allowed-root-RELATIVE paths and return root-relative results. Mutations on a read-only
    connector raise `StorageReadOnly` before any delegation."""

    def __init__(
        self,
        transport: RemoteTransport,
        *,
        allowed_root: str,
        read_only: bool = True,
        max_bytes: int = 10_000_000,
        realpath: RealpathHook | None = None,
        name: str = "storage",
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.name = name
        self._t = transport
        self._root = _norm_abs(allowed_root)
        self._read_only = bool(read_only)
        self._max_bytes = int(max_bytes)
        self._realpath = realpath

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def allowed_root(self) -> str:
        return self._root

    # -- internal --------------------------------------------------------- #
    async def _contain(self, relpath: str) -> str:
        """Lexical containment, then (if a realpath hook is wired) server-side resolve + re-check
        so a remote symlink escaping `allowed_root` is refused."""
        target = contain_remote(self._root, relpath)
        if self._realpath is not None:
            real = await self._guarded(self._realpath, target)
            if not _within(self._root, _norm_abs(real)):
                raise StorageAccessError(f"path escapes allowed_root via symlink: {relpath!r}")
        return target

    async def _guarded(self, fn, *args):
        """Delegate to the transport, mapping any non-typed exception to `StorageUnavailable`
        (type name only — never echo a transport message that could carry host/credential text)."""
        try:
            return await fn(*args)
        except StorageError:
            raise
        except Exception as exc: # noqa: BLE001
            raise StorageUnavailable(f"transport error: {type(exc).__name__}") from exc

    def _require_writable(self) -> None:
        if self._read_only:
            raise StorageReadOnly(f"connector {self.name!r} is read-only")

    # -- public (read) ---------------------------------------------------- #
    async def list(self, path: str = ".") -> list[RemoteEntry]:
        target = await self._contain(path)
        entries = await self._guarded(self._t.list_dir, target)
        return [RemoteEntry(_relativize(self._root, e.path), e.is_dir, e.size) for e in entries]

    async def stat(self, path: str) -> RemoteEntry:
        target = await self._contain(path)
        e = await self._guarded(self._t.stat, target)
        return RemoteEntry(_relativize(self._root, e.path), e.is_dir, e.size)

    async def read_bytes(self, path: str) -> bytes:
        target = await self._contain(path)
        entry = await self._guarded(self._t.stat, target) # size check is policy → here
        if entry.size > self._max_bytes:
            raise StorageTooLarge(f"{path!r} exceeds {self._max_bytes} bytes")
        return await self._guarded(self._t.read_bytes, target)

    # -- public (mutating — read_only gated) ------------------------------ #
    async def write_bytes(self, path: str, data: bytes) -> None:
        self._require_writable()
        target = await self._contain(path)
        await self._guarded(self._t.write_bytes, target, data)

    async def delete(self, path: str) -> None:
        self._require_writable()
        target = await self._contain(path)
        await self._guarded(self._t.delete, target)

    async def move(self, src: str, dst: str) -> None:
        self._require_writable()
        src_t = await self._contain(src)
        dst_t = await self._contain(dst) # both endpoints contained
        await self._guarded(self._t.move, src_t, dst_t)

    async def close(self) -> None:
        await self._guarded(self._t.close)
