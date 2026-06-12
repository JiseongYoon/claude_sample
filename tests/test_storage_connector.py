"""Tests for the storage connector seam + GuardedConnector + config/credential model
(/ ).

Network-free: an in-memory fake `RemoteTransport` stands in for SSH/Synology. Focus: remote
path containment (`..`/absolute/symlink-escape), read-only enforcement, read size cap, typed
errors (never a raw exception / credential leak), and the connector config/credential model
(JSON load, exactly-one auth, password-from-env, no inline secret). Run in conda
`local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import json
import posixpath

import pytest

from local_ai_agent.modules.storage.config import (
    SSHAuth,
    StorageConnectorConfig,
    load_connector_configs,
    resolve_password,
)
from local_ai_agent.modules.storage.connector import (
    GuardedConnector,
    RemoteEntry,
    StorageAccessError,
    StorageNotFound,
    StorageReadOnly,
    StorageTooLarge,
    StorageUnavailable,
    contain_remote,
)

_ROOT = "/srv/share"


class FakeTransport:
    def __init__(self, *, files=None, dirs=None, symlinks=None):
        self.files = dict(files or {}) # abspath -> bytes
        self.dirs = set(dirs or []) # abspath dirs
        self.symlinks = dict(symlinks or {}) # abspath -> resolved abspath
        self.calls = []
        self.closed = False

    async def list_dir(self, path):
        self.calls.append(("list_dir", path))
        out = []
        for p, b in self.files.items():
            if posixpath.dirname(p) == path:
                out.append(RemoteEntry(p, False, len(b)))
        for d in self.dirs:
            if d != path and posixpath.dirname(d) == path:
                out.append(RemoteEntry(d, True, 0))
        return out

    async def stat(self, path):
        self.calls.append(("stat", path))
        if path in self.files:
            return RemoteEntry(path, False, len(self.files[path]))
        if path in self.dirs:
            return RemoteEntry(path, True, 0)
        raise StorageNotFound(path)

    async def read_bytes(self, path):
        self.calls.append(("read_bytes", path))
        if path not in self.files:
            raise StorageNotFound(path)
        return self.files[path]

    async def write_bytes(self, path, data):
        self.calls.append(("write_bytes", path))
        self.files[path] = data

    async def delete(self, path):
        self.calls.append(("delete", path))
        self.files.pop(path, None)

    async def move(self, src, dst):
        self.calls.append(("move", src, dst))
        self.files[dst] = self.files.pop(src, b"")

    async def realpath(self, path):
        self.calls.append(("realpath", path))
        return self.symlinks.get(path, path)

    async def close(self):
        self.closed = True


def _transport():
    return FakeTransport(
        files={f"{_ROOT}/a.txt": b"hello", f"{_ROOT}/sub/b.txt": b"x" * 100},
        dirs={_ROOT, f"{_ROOT}/sub"},
    )


def _conn(transport=None, **kw):
    return GuardedConnector(transport or _transport(), allowed_root=_ROOT, **kw)


# --------------------------------------------------------------------------- #
# contain_remote (pure)
# --------------------------------------------------------------------------- #
def test_contain_within():
    assert contain_remote(_ROOT, "a.txt") == f"{_ROOT}/a.txt"
    assert contain_remote(_ROOT, "sub/b.txt") == f"{_ROOT}/sub/b.txt"
    assert contain_remote(_ROOT, ".") == _ROOT
    assert contain_remote(_ROOT, "x//y/../z") == f"{_ROOT}/x/z"


@pytest.mark.parametrize("rel", ["/etc/passwd", "../escape", "../../etc", "sub/../../out", "a/../../../x"])
def test_contain_escape(rel):
    with pytest.raises(StorageAccessError):
        contain_remote(_ROOT, rel)


def test_contain_root_slash_allows_all():
    assert contain_remote("/", "etc/passwd") == "/etc/passwd"


def test_contain_non_abs_root_and_bad_relpath():
    with pytest.raises(StorageAccessError):
        contain_remote("relative/root", "a")
    with pytest.raises(StorageAccessError):
        contain_remote(_ROOT, 123) # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# GuardedConnector — reads
# --------------------------------------------------------------------------- #
async def test_list_relativized():
    out = await _conn().list(".")
    by = {e.path: e for e in out}
    assert by["a.txt"].is_dir is False and by["a.txt"].size == 5
    assert by["sub"].is_dir is True


async def test_stat_and_read():
    c = _conn()
    assert (await c.stat("a.txt")).path == "a.txt"
    assert await c.read_bytes("a.txt") == b"hello"


async def test_read_over_cap_rejected_before_read():
    t = _transport()
    c = _conn(t, max_bytes=10)
    with pytest.raises(StorageTooLarge):
        await c.read_bytes("sub/b.txt") # size 100 > 10
    assert ("read_bytes", f"{_ROOT}/sub/b.txt") not in t.calls # never read


async def test_read_escape_not_delegated():
    t = _transport()
    with pytest.raises(StorageAccessError):
        await _conn(t).read_bytes("../../etc/shadow")
    assert all(call[0] not in ("read_bytes", "stat") for call in t.calls)


# --------------------------------------------------------------------------- #
# GuardedConnector — read-only enforcement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", ["write", "delete", "move"])
async def test_read_only_blocks_mutations(op):
    t = _transport()
    c = _conn(t, read_only=True)
    with pytest.raises(StorageReadOnly):
        if op == "write":
            await c.write_bytes("a.txt", b"z")
        elif op == "delete":
            await c.delete("a.txt")
        else:
            await c.move("a.txt", "c.txt")
    assert all(call[0] not in ("write_bytes", "delete", "move") for call in t.calls)


async def test_writable_mutations_delegate():
    t = _transport()
    c = _conn(t, read_only=False)
    await c.write_bytes("c.txt", b"zzz")
    assert t.files[f"{_ROOT}/c.txt"] == b"zzz"
    await c.move("c.txt", "d.txt")
    assert f"{_ROOT}/d.txt" in t.files and f"{_ROOT}/c.txt" not in t.files
    await c.delete("d.txt")
    assert f"{_ROOT}/d.txt" not in t.files


async def test_move_contains_both_endpoints():
    t = _transport()
    c = _conn(t, read_only=False)
    with pytest.raises(StorageAccessError):
        await c.move("a.txt", "../evil") # dst escapes
    assert f"{_ROOT}/a.txt" in t.files # nothing moved


# --------------------------------------------------------------------------- #
# realpath hook — closes the lexical symlink-escape gap
# --------------------------------------------------------------------------- #
async def test_realpath_hook_rejects_symlink_escape():
    t = FakeTransport(
        files={f"{_ROOT}/a.txt": b"ok"},
        dirs={_ROOT},
        symlinks={f"{_ROOT}/link": "/etc/shadow"}, # symlink inside root → points outside
    )
    c = GuardedConnector(t, allowed_root=_ROOT, realpath=t.realpath)
    with pytest.raises(StorageAccessError):
        await c.read_bytes("link") # lexical OK, realpath catches the escape
    assert ("read_bytes", f"{_ROOT}/link") not in t.calls


async def test_realpath_hook_allows_within():
    t = FakeTransport(
        files={f"{_ROOT}/link2": b"data", f"{_ROOT}/a.txt": b"data"},
        dirs={_ROOT},
        symlinks={f"{_ROOT}/link2": f"{_ROOT}/a.txt"},
    )
    c = GuardedConnector(t, allowed_root=_ROOT, realpath=t.realpath)
    assert await c.read_bytes("link2") == b"data"


# --------------------------------------------------------------------------- #
# error mapping + no leak
# --------------------------------------------------------------------------- #
async def test_transport_error_wrapped_no_leak():
    class Boom:
        async def stat(self, path):
            raise RuntimeError("connect host=10.0.0.5 password=topsecret failed")
        async def list_dir(self, path):
            raise RuntimeError("password=topsecret")
        async def read_bytes(self, p): ...
        async def write_bytes(self, p, d): ...
        async def delete(self, p): ...
        async def move(self, s, d): ...
        async def realpath(self, p): return p
        async def close(self): ...

    c = GuardedConnector(Boom(), allowed_root=_ROOT)
    with pytest.raises(StorageUnavailable) as ei:
        await c.list(".")
    assert "topsecret" not in str(ei.value) and "password" not in str(ei.value)


async def test_not_found_passes_through():
    with pytest.raises(StorageNotFound):
        await _conn().stat("missing.txt")


def test_bad_max_bytes_init():
    with pytest.raises(ValueError):
        _conn(max_bytes=0)


def test_non_abs_allowed_root_init():
    with pytest.raises(StorageAccessError):
        GuardedConnector(_transport(), allowed_root="not/absolute")


# --------------------------------------------------------------------------- #
# config / credential model
# --------------------------------------------------------------------------- #
def _cfg_doc(**over):
    base = {"name": "nas", "kind": "ssh", "host": "10.0.0.5", "username": "u",
            "auth": {"key_path": "/home/u/.ssh/id_ed25519"}, "allowed_root": "/srv/share"}
    base.update(over)
    return {"connectors": [base]}


def test_load_valid(tmp_path):
    f = tmp_path / "storage.json"
    f.write_text(json.dumps(_cfg_doc()), encoding="utf-8")
    cfgs = load_connector_configs(f)
    assert len(cfgs) == 1 and cfgs[0].name == "nas" and cfgs[0].read_only is True


def test_load_duplicate_name(tmp_path):
    f = tmp_path / "s.json"
    doc = _cfg_doc()
    doc["connectors"].append(dict(doc["connectors"][0]))
    f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_connector_configs(f)


def test_load_malformed_json(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_connector_configs(f)


def test_load_missing_file(tmp_path):
    with pytest.raises(ValueError):
        load_connector_configs(tmp_path / "nope.json")


@pytest.mark.parametrize("over", [
    {"allowed_root": "relative/path"}, # not absolute
    {"port": 0}, # out of range
    {"auth": {"key_path": "/k", "password_env": "PW"}}, # both auth
    {"auth": {}}, # neither auth
    {"host": ""}, # empty
])
def test_load_schema_violations(tmp_path, over):
    f = tmp_path / "s.json"
    f.write_text(json.dumps(_cfg_doc(**over)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_connector_configs(f)


def test_resolve_password_from_env(monkeypatch):
    monkeypatch.setenv("MY_PW", "supersecret")
    auth = SSHAuth(password_env="MY_PW")
    assert resolve_password(auth) == "supersecret"


def test_resolve_password_missing_env(monkeypatch):
    monkeypatch.delenv("MY_PW", raising=False)
    with pytest.raises(ValueError):
        resolve_password(SSHAuth(password_env="MY_PW"))


def test_resolve_password_keybased_is_none():
    assert resolve_password(SSHAuth(key_path="/home/u/.ssh/id_ed25519")) is None


def test_credential_never_inline_in_config(monkeypatch):
    # the config object stores only the env var NAME, never the secret value
    monkeypatch.setenv("MY_PW", "supersecret")
    cfg = StorageConnectorConfig(name="n", host="h", username="u",
                                 auth=SSHAuth(password_env="MY_PW"), allowed_root="/srv")
    blob = repr(cfg) + str(cfg) + repr(cfg.auth)
    assert "supersecret" not in blob
    assert "MY_PW" in blob # the NAME is fine to hold
