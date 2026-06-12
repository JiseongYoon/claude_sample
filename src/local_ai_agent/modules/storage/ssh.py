"""SSH/SFTP `RemoteTransport` over asyncssh.

Implements the raw `RemoteTransport` seam against an SSH/SFTP server. Policy (containment,
read-only, size) lives in `GuardedConnector`; this layer maps the seam to SFTP ops,
manages one lazy connection (single-flighted + reconnect-once), and maps asyncssh faults to typed
`StorageError`s with **scrubbed messages** (host/username/key/password never appear — only a fixed
string + the exception type name).

Host-key verification is **mandatory**: the connector requires a `known_hosts_path` and never
passes `known_hosts=None` to asyncssh (which would disable verification → MITM). The SFTP
`realpath` is exposed so the `GuardedConnector` realpath hook can resolve server-side symlinks and
re-check containment.

asyncssh is **lazy-imported** so a missing dependency disables only this connector (typed error),
never an import crash. For tests the connection factory is injectable (a fake SFTP client), and an
in-process asyncssh loopback server exercises the real protocol.
"""
from __future__ import annotations

import asyncio
import posixpath
import stat as _stat
from typing import Any, Awaitable, Callable, Protocol

from .config import StorageConnectorConfig, resolve_password
from .connector import (
    RemoteEntry,
    StorageAccessError,
    StorageAuthError,
    StorageError,
    StorageNotFound,
    StorageTooLarge,
    StorageUnavailable,
)

_FILEXFER_TYPE_DIRECTORY = 2


class SFTPLike(Protocol):
    """The narrow SFTP surface this transport uses (asyncssh.SFTPClient satisfies it; tests
    inject an in-memory fake)."""

    async def readdir(self, path: str) -> list: ...
    async def stat(self, path: str) -> Any: ...
    async def open(self, path: str, mode: str): ...
    async def remove(self, path: str) -> None: ...
    async def posix_rename(self, src: str, dst: str) -> None: ...
    async def realpath(self, path: str) -> str: ...


# (connection, sftp-client) factory — injectable for tests
ConnectFn = Callable[[], Awaitable[tuple[Any, SFTPLike]]]


# --------------------------------------------------------------------------- #
# asyncssh fault mapping (scrubbed — never echo a transport message)
# --------------------------------------------------------------------------- #
def _to_storage_error(exc: Exception) -> StorageError | None:
    try:
        import asyncssh
    except ImportError:
        return None
    if isinstance(exc, (asyncssh.SFTPNoSuchFile, asyncssh.SFTPNoSuchPath)):
        return StorageNotFound("no such remote path")
    if isinstance(exc, asyncssh.SFTPPermissionDenied):
        return StorageAccessError("remote permission denied")
    if isinstance(exc, (asyncssh.PermissionDenied, asyncssh.HostKeyNotVerifiable)):
        return StorageAuthError("authentication or host-key verification failed")
    if isinstance(exc, (asyncssh.ConnectionLost, asyncssh.DisconnectError, asyncssh.ChannelOpenError)):
        return StorageUnavailable(f"connection error: {type(exc).__name__}")
    if isinstance(exc, asyncssh.SFTPError):
        return StorageUnavailable(f"sftp error: {type(exc).__name__}")
    if isinstance(exc, OSError):
        return StorageUnavailable(f"io error: {type(exc).__name__}")
    return None


def _is_conn_lost(exc: Exception) -> bool:
    try:
        import asyncssh
    except ImportError:
        return False
    lost = (asyncssh.ConnectionLost, asyncssh.DisconnectError)
    extra = getattr(asyncssh, "SFTPConnectionLost", None)
    if extra is not None:
        lost = lost + (extra,)
    return isinstance(exc, lost)


def _is_dir(attrs: Any) -> bool:
    perm = getattr(attrs, "permissions", None)
    if perm is not None:
        return _stat.S_ISDIR(perm)
    return getattr(attrs, "type", None) == _FILEXFER_TYPE_DIRECTORY


def _size(attrs: Any) -> int:
    return int(getattr(attrs, "size", 0) or 0)


def _filename(name: Any) -> str:
    fn = getattr(name, "filename", name)
    return fn.decode("utf-8", "replace") if isinstance(fn, bytes) else str(fn)


# --------------------------------------------------------------------------- #
# the real connect factory (lazy asyncssh; host-key verification mandatory)
# --------------------------------------------------------------------------- #
async def _default_connect(config: StorageConnectorConfig) -> tuple[Any, SFTPLike]:
    try:
        import asyncssh
    except ImportError as exc: # missing optional dep → typed, not an import crash
        raise StorageUnavailable("asyncssh not installed") from exc
    if not config.known_hosts_path:
        raise StorageAuthError("known_hosts not configured — refusing to connect unverified")
    client_keys = [str(config.auth.key_path)] if config.auth.key_path else None
    password = resolve_password(config.auth)
    try:
        conn = await asyncssh.connect(
            config.host, port=config.port, username=config.username,
            client_keys=client_keys, password=password,
            known_hosts=str(config.known_hosts_path), # NEVER None
        )
        sftp = await conn.start_sftp_client()
    except (asyncssh.PermissionDenied, asyncssh.HostKeyNotVerifiable) as exc:
        raise StorageAuthError("authentication or host-key verification failed") from exc
    except (OSError, asyncssh.Error) as exc:
        raise StorageUnavailable(f"connection error: {type(exc).__name__}") from exc
    return conn, sftp


# --------------------------------------------------------------------------- #
# the transport
# --------------------------------------------------------------------------- #
class SSHTransport:
    """`RemoteTransport` over SSH/SFTP. Operates on ABSOLUTE, already-contained paths (the
    `GuardedConnector` contains first). One lazy connection, single-flighted, reconnect-once."""

    def __init__(
        self, config: StorageConnectorConfig, *, max_bytes: int, connect: ConnectFn | None = None
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._config = config
        self._max_bytes = int(max_bytes)
        self._connect: ConnectFn = connect or (lambda: _default_connect(config))
        self._conn: Any = None
        self._sftp: SFTPLike | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> SFTPLike:
        if self._sftp is not None:
            return self._sftp
        async with self._lock:
            if self._sftp is None:
                self._conn, self._sftp = await self._connect()
            return self._sftp

    async def _reset(self) -> None:
        conn, self._conn, self._sftp = self._conn, None, None
        if conn is not None:
            try:
                conn.close()
            except Exception: # noqa: BLE001 — best-effort teardown
                pass

    def _reraise(self, exc: Exception):
        mapped = _to_storage_error(exc)
        if mapped is not None:
            raise mapped from exc
        raise exc # unknown → GuardedConnector wraps as StorageUnavailable (type name only)

    async def _call(self, make_coro: Callable[[SFTPLike], Awaitable]):
        for attempt in (1, 2):
            sftp = await self._ensure()
            try:
                return await make_coro(sftp)
            except StorageError:
                raise
            except Exception as exc: # noqa: BLE001
                if attempt == 1 and _is_conn_lost(exc):
                    await self._reset()
                    continue
                self._reraise(exc)

    # -- RemoteTransport ops (absolute contained paths in) ---------------- #
    async def list_dir(self, path: str) -> list[RemoteEntry]:
        names = await self._call(lambda s: s.readdir(path))
        out: list[RemoteEntry] = []
        for n in names:
            fn = _filename(n)
            if fn in (".", ".."):
                continue
            out.append(RemoteEntry(posixpath.join(path, fn), _is_dir(n.attrs), _size(n.attrs)))
        return out

    async def stat(self, path: str) -> RemoteEntry:
        attrs = await self._call(lambda s: s.stat(path))
        return RemoteEntry(path, _is_dir(attrs), _size(attrs))

    async def read_bytes(self, path: str) -> bytes:
        async def _rd(s: SFTPLike) -> bytes:
            f = await s.open(path, "rb")
            try:
                return await f.read(self._max_bytes + 1) # +1 to detect overflow (TOCTOU defense)
            finally:
                await f.close()

        data = await self._call(_rd)
        if len(data) > self._max_bytes:
            raise StorageTooLarge(f"remote read exceeds {self._max_bytes} bytes")
        return data

    async def write_bytes(self, path: str, data: bytes) -> None:
        async def _wr(s: SFTPLike) -> None:
            f = await s.open(path, "wb")
            try:
                await f.write(data)
            finally:
                await f.close()

        await self._call(_wr)

    async def delete(self, path: str) -> None:
        await self._call(lambda s: s.remove(path))

    async def move(self, src: str, dst: str) -> None:
        await self._call(lambda s: s.posix_rename(src, dst))

    async def realpath(self, path: str) -> str:
        result = await self._call(lambda s: s.realpath(path))
        return result.decode("utf-8", "replace") if isinstance(result, bytes) else str(result)

    async def close(self) -> None:
        conn, self._conn, self._sftp = self._conn, None, None
        if conn is not None:
            conn.close()
            wait = getattr(conn, "wait_closed", None)
            if wait is not None:
                try:
                    await wait()
                except Exception: # noqa: BLE001
                    pass
