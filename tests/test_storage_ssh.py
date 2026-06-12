"""Tests for the SSH/SFTP transport.

Two layers, per the design:
  * **fake `SFTPLike` unit tests** — op mapping, read cap, error mapping (real asyncssh exception
    types), reconnect-once, and the mandatory-known_hosts connect guard. Fast, no server.
  * **loopback integration** — an in-process asyncssh SFTP server on 127.0.0.1 (generated keys)
    exercises the REAL client/protocol: host-key verification (known_hosts), pubkey auth, SFTP
    read/stat/list, and the realpath-hook containment over a real symlink escape.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import os
import posixpath
import stat as _stat

import asyncssh
import pytest

from local_ai_agent.modules.storage.config import SSHAuth, StorageConnectorConfig
from local_ai_agent.modules.storage.connector import (
    GuardedConnector,
    StorageAccessError,
    StorageAuthError,
    StorageNotFound,
    StorageTooLarge,
)
from local_ai_agent.modules.storage.ssh import SSHTransport, _default_connect

_ROOT = "/srv"


# --------------------------------------------------------------------------- #
# fake SFTP
# --------------------------------------------------------------------------- #
class _Attrs:
    def __init__(self, size=0, permissions=None):
        self.size = size
        self.permissions = permissions


class _Name:
    def __init__(self, filename, attrs):
        self.filename = filename
        self.attrs = attrs


class _File:
    def __init__(self, store, path, mode):
        self._s, self._p, self._m, self._buf = store, path, mode, b""

    async def read(self, n=-1):
        data = self._s.files.get(self._p, b"")
        return data if (n is None or n < 0) else data[:n]

    async def write(self, data):
        self._buf += data

    async def close(self):
        if "w" in self._m:
            self._s.files[self._p] = self._buf


class FakeSFTP:
    def __init__(self):
        self.files = {} # abspath -> bytes
        self.dirs = set() # abspath dirs
        self.symlinks = {} # abspath -> resolved abspath
        self.fail_once = None # exception raised once on the next op

    def _maybe_fail(self):
        if self.fail_once is not None:
            e, self.fail_once = self.fail_once, None
            raise e

    async def readdir(self, path):
        self._maybe_fail()
        out = [_Name(".", _Attrs(permissions=_stat.S_IFDIR | 0o755))]
        for p, b in self.files.items():
            if posixpath.dirname(p) == path:
                out.append(_Name(posixpath.basename(p), _Attrs(len(b), _stat.S_IFREG | 0o644)))
        for d in self.dirs:
            if d != path and posixpath.dirname(d) == path:
                out.append(_Name(posixpath.basename(d), _Attrs(permissions=_stat.S_IFDIR | 0o755)))
        return out

    async def stat(self, path):
        self._maybe_fail()
        if path in self.files:
            return _Attrs(len(self.files[path]), _stat.S_IFREG | 0o644)
        if path in self.dirs:
            return _Attrs(permissions=_stat.S_IFDIR | 0o755)
        raise asyncssh.SFTPNoSuchFile("missing")

    async def open(self, path, mode):
        self._maybe_fail()
        if "r" in mode and path not in self.files:
            raise asyncssh.SFTPNoSuchFile("missing")
        return _File(self, path, mode)

    async def remove(self, path):
        self._maybe_fail()
        self.files.pop(path, None)

    async def posix_rename(self, src, dst):
        self._maybe_fail()
        self.files[dst] = self.files.pop(src, b"")

    async def realpath(self, path):
        self._maybe_fail()
        return self.symlinks.get(path, path)


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


def _cfg(**over):
    base = dict(name="t", host="h", username="u", auth=SSHAuth(key_path="/k"), allowed_root=_ROOT)
    base.update(over)
    return StorageConnectorConfig(**base)


def _xport(sftp, *, max_bytes=10_000_000, connects=None):
    async def connect():
        if connects is not None:
            connects.append(1)
        return FakeConn(), sftp
    return SSHTransport(_cfg(), max_bytes=max_bytes, connect=connect)


def _populated():
    s = FakeSFTP()
    s.files = {f"{_ROOT}/a.txt": b"hello", f"{_ROOT}/sub/b.txt": b"x" * 100}
    s.dirs = {_ROOT, f"{_ROOT}/sub"}
    return s


# --------------------------------------------------------------------------- #
# fake unit — op mapping
# --------------------------------------------------------------------------- #
async def test_list_dir_maps_entries():
    t = _xport(_populated())
    out = {e.path: e for e in await t.list_dir(_ROOT)}
    assert out[f"{_ROOT}/a.txt"].is_dir is False and out[f"{_ROOT}/a.txt"].size == 5
    assert out[f"{_ROOT}/sub"].is_dir is True
    assert "." not in [posixpath.basename(p) for p in out] # '.' filtered


async def test_stat_and_read():
    t = _xport(_populated())
    e = await t.stat(f"{_ROOT}/a.txt")
    assert e.size == 5 and e.is_dir is False
    assert await t.read_bytes(f"{_ROOT}/a.txt") == b"hello"


async def test_read_cap():
    t = _xport(_populated(), max_bytes=10)
    with pytest.raises(StorageTooLarge):
        await t.read_bytes(f"{_ROOT}/sub/b.txt") # 100 bytes > 10


async def test_write_delete_move():
    s = _populated()
    t = _xport(s)
    await t.write_bytes(f"{_ROOT}/c.txt", b"zzz")
    assert s.files[f"{_ROOT}/c.txt"] == b"zzz"
    await t.move(f"{_ROOT}/c.txt", f"{_ROOT}/d.txt")
    assert f"{_ROOT}/d.txt" in s.files and f"{_ROOT}/c.txt" not in s.files
    await t.delete(f"{_ROOT}/d.txt")
    assert f"{_ROOT}/d.txt" not in s.files


async def test_realpath_passthrough():
    s = _populated()
    s.symlinks = {f"{_ROOT}/link": f"{_ROOT}/a.txt"}
    t = _xport(s)
    assert await t.realpath(f"{_ROOT}/link") == f"{_ROOT}/a.txt"


# --------------------------------------------------------------------------- #
# fake unit — error mapping + reconnect
# --------------------------------------------------------------------------- #
async def test_not_found_mapped():
    t = _xport(_populated())
    with pytest.raises(StorageNotFound):
        await t.stat(f"{_ROOT}/missing.txt")


async def test_unknown_exception_propagates():
    s = _populated()
    s.fail_once = RuntimeError("boom")
    t = _xport(s)
    with pytest.raises(RuntimeError): # not ours → propagates (Guard wraps type-only)
        await t.stat(f"{_ROOT}/a.txt")


async def test_reconnect_once_on_connection_lost():
    s = _populated()
    s.fail_once = asyncssh.ConnectionLost("lost")
    connects = []
    t = _xport(s, connects=connects)
    e = await t.stat(f"{_ROOT}/a.txt") # 1st op drops → reset → reconnect → succeeds
    assert e.size == 5
    assert len(connects) == 2


async def test_default_connect_requires_known_hosts():
    # the security gate: no known_hosts → refuse to connect (never disables verification)
    with pytest.raises(StorageAuthError):
        await _default_connect(_cfg()) # known_hosts_path is None


def test_bad_max_bytes():
    with pytest.raises(ValueError):
        SSHTransport(_cfg(), max_bytes=0, connect=lambda: None)


# --------------------------------------------------------------------------- #
# loopback integration — real asyncssh server on 127.0.0.1
# --------------------------------------------------------------------------- #
class _SSHServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True # require client auth

    def public_key_auth_supported(self):
        return True


async def _start_server(tmp_path):
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    client_key = asyncssh.generate_private_key("ssh-ed25519")
    ckey = tmp_path / "client_key"
    ckey.write_bytes(client_key.export_private_key())
    cpub = tmp_path / "client_key.pub"
    cpub.write_bytes(client_key.export_public_key())
    server = await asyncssh.create_server(
        _SSHServer, "127.0.0.1", 0,
        server_host_keys=[host_key],
        authorized_client_keys=str(cpub),
        sftp_factory=asyncssh.SFTPServer,
    )
    port = server.get_port()
    host_pub = host_key.export_public_key().decode().strip()
    return server, port, str(ckey), host_pub


def _conn(tmp_path, work_root, *, known_hosts, port, key_path, read_only=False):
    cfg = StorageConnectorConfig(
        name="lo", host="127.0.0.1", port=port, username="tester",
        auth=SSHAuth(key_path=key_path), allowed_root=str(work_root),
        known_hosts_path=str(known_hosts), read_only=read_only, max_bytes=10_000_000,
    )
    t = SSHTransport(cfg, max_bytes=cfg.max_bytes) # real _default_connect
    return GuardedConnector(t, allowed_root=cfg.allowed_root, read_only=read_only,
                            max_bytes=cfg.max_bytes, realpath=t.realpath), t


async def test_loopback_real_sftp_read_list_stat(tmp_path):
    work = tmp_path / "share"
    work.mkdir()
    (work / "doc.txt").write_text("hello remote", encoding="utf-8")
    server, port, key_path, host_pub = await _start_server(tmp_path)
    kh = tmp_path / "known_hosts"
    kh.write_text(f"[127.0.0.1]:{port} {host_pub}\n", encoding="utf-8")
    gc, t = _conn(tmp_path, work, known_hosts=kh, port=port, key_path=key_path)
    try:
        assert await gc.read_bytes("doc.txt") == b"hello remote"
        assert (await gc.stat("doc.txt")).size == len("hello remote")
        assert "doc.txt" in {e.path for e in await gc.list(".")}
    finally:
        await t.close()
        server.close()
        await server.wait_closed()


async def test_loopback_realpath_blocks_symlink_escape(tmp_path):
    work = tmp_path / "share"
    work.mkdir()
    # a symlink inside the allowed root pointing OUTSIDE it
    outside = tmp_path / "secret.txt"
    outside.write_text("TOPSECRET", encoding="utf-8")
    os.symlink(str(outside), str(work / "escape"))
    server, port, key_path, host_pub = await _start_server(tmp_path)
    kh = tmp_path / "known_hosts"
    kh.write_text(f"[127.0.0.1]:{port} {host_pub}\n", encoding="utf-8")
    gc, t = _conn(tmp_path, work, known_hosts=kh, port=port, key_path=key_path)
    try:
        with pytest.raises(StorageAccessError): # realpath hook resolves outside → refused
            await gc.read_bytes("escape")
    finally:
        await t.close()
        server.close()
        await server.wait_closed()


async def test_loopback_host_key_mismatch_rejected(tmp_path):
    work = tmp_path / "share"
    work.mkdir()
    server, port, key_path, host_pub = await _start_server(tmp_path)
    # known_hosts pins a DIFFERENT host key → verification must fail
    wrong_pub = asyncssh.generate_private_key("ssh-ed25519").export_public_key().decode().strip()
    kh = tmp_path / "known_hosts"
    kh.write_text(f"[127.0.0.1]:{port} {wrong_pub}\n", encoding="utf-8")
    gc, t = _conn(tmp_path, work, known_hosts=kh, port=port, key_path=key_path)
    try:
        with pytest.raises(StorageAuthError):
            await gc.stat(".")
    finally:
        await t.close()
        server.close()
        await server.wait_closed()
