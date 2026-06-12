"""— exec executor seam + ExecPolicy + workspace containment (network-free, daemon-free).

The security crux of . A fake `Executor` records every (spec, argv/path) it is asked to run and
asserts it is NEVER reached for a blocked request — proving containment is fail-closed *before* any side
effect. No Docker, no network: `ExecPolicy` is pure, so the real security control is fully hermetic
(the F3 requirement). Covers the mandatory-hardening of every computed `SandboxSpec`, the
broker-forbidden options being unrepresentable, command/workspace-path containment, env-scrub, the
output cap + no-leak error mapping, config validators, and the universal invariant.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_ai_agent.config import Settings
from local_ai_agent.modules.exec import (
    ExecBlocked,
    ExecConfig,
    ExecPolicy,
    ExecTimeout,
    ExecUnavailable,
    GuardedExecutor,
    Mount,
    RawExecResult,
    SandboxSpec,
)

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeExecutor:
    """Records calls; asserts (via the test) that a blocked request never reaches it. `run` returns a
    canned result, or raises `raise_exc` to exercise the policy layer's error mapping."""

    def __init__(self, *, result: RawExecResult | None = None, raise_exc: Exception | None = None,
                 read_bytes: bytes = b""):
        self.calls: list[tuple] = []
        self.closed = False
        self._result = result or RawExecResult(exit_code=0, stdout=b"ok")
        self._raise = raise_exc
        self._read = read_bytes

    async def run(self, spec, argv, *, timeout, max_output_bytes):
        self.calls.append(("run", spec, tuple(argv), timeout, max_output_bytes))
        if self._raise is not None:
            raise self._raise
        return self._result

    async def read_file(self, spec, path, *, max_bytes):
        self.calls.append(("read", spec, path, max_bytes))
        if self._raise is not None:
            raise self._raise
        return self._read

    async def write_file(self, spec, path, data):
        self.calls.append(("write", spec, path, bytes(data)))
        if self._raise is not None:
            raise self._raise

    async def close(self):
        self.closed = True


def _policy(environ=None, **overrides) -> ExecPolicy:
    cfg = ExecConfig.from_settings(Settings(**_DIRS, **overrides))
    return ExecPolicy(cfg, environ=environ)


def _guarded(executor, environ=None, **overrides) -> GuardedExecutor:
    return GuardedExecutor(
        executor, policy=_policy(environ=environ, **overrides), workspace_volume="ws-1",
        timeout=5.0, max_output_bytes=1000,
    )


# --------------------------------------------------------------------------- #
# normal — spec hardening + containment + plumbing
# --------------------------------------------------------------------------- #
def test_build_spec_sets_all_mandatory_hardening():
    spec = _policy().build_spec(workspace_volume="ws-1")
    assert isinstance(spec, SandboxSpec)
    assert spec.network == "none"
    assert spec.read_only is True
    assert spec.user not in ("root", "0", "0:0")
    assert spec.cap_drop == ("ALL",)
    assert spec.no_new_privileges is True
    assert spec.init is True
    assert spec.seccomp == "default"
    assert spec.memory_swap == spec.mem_bytes # no swap escape
    assert spec.tmpfs == ("/tmp",)
    assert spec.mounts == (Mount(volume="ws-1", target="/workspace", read_only=False),)
    # resource caps come from config (all > 0)
    assert spec.mem_bytes > 0 and spec.cpus > 0 and spec.pids_limit > 0
    assert spec.ulimit_nofile > 0 and spec.ulimit_fsize > 0


@pytest.mark.parametrize("rel,expected", [
    ("x.txt", "/workspace/x.txt"),
    ("a/b.txt", "/workspace/a/b.txt"),
    ("a/../b.txt", "/workspace/b.txt"), # stays under root after normalization
    ("./c.txt", "/workspace/c.txt"),
])
def test_contain_workspace_path_admits_in_workspace(rel, expected):
    assert _policy().contain_workspace_path(rel) == expected


async def test_run_command_runs_contained_argv_on_hardened_spec():
    fx = FakeExecutor(result=RawExecResult(exit_code=0, stdout=b"hello"))
    g = _guarded(fx)
    res = await g.run_command(["echo", "hello"])
    assert res.stdout == b"hello" and res.exit_code == 0
    kind, spec, argv, timeout, cap = fx.calls[0]
    assert kind == "run" and argv == ("echo", "hello")
    assert spec.network == "none" and spec.cap_drop == ("ALL",) # hardened spec was used
    assert timeout == 5.0 and cap == 1000 # caps applied uniformly


async def test_read_and_write_workspace_file_contained():
    fx = FakeExecutor(read_bytes=b"data")
    g = _guarded(fx)
    assert await g.read_workspace_file("notes/a.txt") == b"data"
    await g.write_workspace_file("out/b.txt", b"payload")
    assert fx.calls[0][:3] == ("read", fx.calls[0][1], "/workspace/notes/a.txt")
    assert fx.calls[1][:4] == ("write", fx.calls[1][1], "/workspace/out/b.txt", b"payload")


def test_config_from_settings_maps_every_field():
    c = ExecConfig.from_settings(Settings(**_DIRS, exec_env_whitelist=["LANG"]))
    assert c.image and c.workspace_root == "/workspace" and c.network == "none"
    assert c.user == "1000:1000" and c.mem_bytes > 0 and c.cpus > 0 and c.pids_limit > 0
    assert c.timeout > 0 and c.max_output_bytes > 0 and c.tmpfs_size_bytes > 0
    assert c.ulimit_nofile > 0 and c.ulimit_fsize > 0 and c.env_whitelist == ("LANG",)


# --------------------------------------------------------------------------- #
# error / security — fail-closed, never reaches the raw executor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    None, "", " ", "echo hi", # bare string rejected (argv only)
    b"echo", [], (), [""], ["ok", ""], ["ok", None], ["ok", 1], ["a\x00b"], 42, {"a": 1},
])
async def test_run_command_rejects_bad_command_without_calling_executor(bad):
    fx = FakeExecutor()
    g = _guarded(fx)
    with pytest.raises(ExecBlocked):
        await g.run_command(bad)
    assert fx.calls == [] # raw executor never reached (fail-closed)


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "/workspace/../etc/x", "../etc/passwd", "../../x", "a/../../x",
    ".", "", " ", "a\x00b", None, 5, ["x"], b"x",
])
async def test_workspace_path_escape_blocked_without_calling_executor(bad):
    fx = FakeExecutor()
    g = _guarded(fx)
    with pytest.raises(ExecBlocked):
        await g.read_workspace_file(bad)
    with pytest.raises(ExecBlocked):
        await g.write_workspace_file(bad, b"x")
    assert fx.calls == []


def test_spec_never_exposes_broker_forbidden_options():
    # over several configs the spec is always hardened and never carries a host bind / docker.sock /
    # privileged / cap-add (these are structurally unrepresentable in SandboxSpec).
    for ov in ({}, {"exec_mem_bytes": 512_000_000}, {"exec_cpus": 1.0}, {"exec_pids_limit": 64}):
        spec = _policy(**ov).build_spec(workspace_volume="ws-x")
        assert spec.network == "none" and spec.read_only is True and spec.no_new_privileges is True
        assert spec.cap_drop == ("ALL",) and spec.user != "root"
        for m in spec.mounts: # only the workspace volume; no host path field exists
            assert isinstance(m, Mount) and m.volume == "ws-x" and m.target == "/workspace"
            assert "docker.sock" not in m.target
        assert not hasattr(spec, "privileged") and not hasattr(spec, "cap_add")


def test_env_scrub_drops_secrets_even_if_whitelisted():
    environ = {
        "LANG": "C.UTF-8", "PATH": "/usr/bin",
        "AWS_SECRET_ACCESS_KEY": "x", "GITHUB_TOKEN": "y", "MY_PASSWORD": "z", "SSH_AUTH_SOCK": "/s",
    }
    # whitelist includes a benign pair AND a secret-looking var — the secret is still denied.
    spec = _policy(environ=environ, exec_env_whitelist=["LANG", "PATH", "GITHUB_TOKEN", "NOT_SET"]).build_spec(
        workspace_volume="ws-1"
    )
    env = dict(spec.env)
    assert env == {"LANG": "C.UTF-8", "PATH": "/usr/bin"} # only benign whitelisted vars
    for leaked in ("AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "MY_PASSWORD", "SSH_AUTH_SOCK", "NOT_SET"):
        assert leaked not in env


async def test_non_typed_executor_error_mapped_to_unavailable_no_leak():
    secret = "host=/home/user/.ssh/id_rsa token=abc123"
    fx = FakeExecutor(raise_exc=ValueError(secret))
    g = _guarded(fx)
    with pytest.raises(ExecUnavailable) as ei:
        await g.run_command(["echo", "hi"])
    msg = str(ei.value)
    assert "ValueError" in msg
    assert "id_rsa" not in msg and "abc123" not in msg and "user" not in msg


async def test_typed_exec_error_passes_through():
    fx = FakeExecutor(raise_exc=ExecTimeout("timed out"))
    g = _guarded(fx)
    with pytest.raises(ExecTimeout):
        await g.run_command(["sleep", "999"])


async def test_write_rejects_non_bytes_data():
    fx = FakeExecutor()
    g = _guarded(fx)
    with pytest.raises(ExecBlocked):
        await g.write_workspace_file("a.txt", "not-bytes") # type: ignore[arg-type]
    assert fx.calls == []


@pytest.mark.parametrize("kw", [
    {"exec_mem_bytes": 0}, {"exec_cpus": 0}, {"exec_cpus": -1.0}, {"exec_pids_limit": 0},
    {"exec_timeout": 0}, {"exec_max_output_bytes": 0}, {"exec_ulimit_nofile": -1},
    {"exec_ulimit_fsize": 0}, {"exec_tmpfs_size_bytes": 0},
    {"exec_image": ""}, {"exec_user": " "}, {"exec_workspace_root": "workspace"},
    {"exec_network": "bridge"}, {"exec_network": "host"},
])
def test_config_validators_reject_bad_values(kw):
    with pytest.raises(ValidationError):
        Settings(**_DIRS, **kw)


@pytest.mark.parametrize("junk", [None, 5, b"x", ["a"], {"k": "v"}, 3.14])
def test_build_spec_rejects_bad_workspace_volume(junk):
    with pytest.raises(ExecBlocked):
        _policy().build_spec(workspace_volume=junk)


def test_guarded_executor_ctor_validates():
    fx = FakeExecutor()
    p = _policy()
    with pytest.raises(ValueError):
        GuardedExecutor(fx, policy=p, workspace_volume="ws", timeout=0, max_output_bytes=10)
    with pytest.raises(ValueError):
        GuardedExecutor(fx, policy=p, workspace_volume="ws", timeout=1, max_output_bytes=0)
    with pytest.raises(ValueError):
        GuardedExecutor(fx, policy=p, workspace_volume=" ", timeout=1, max_output_bytes=10)


async def test_close_delegates():
    fx = FakeExecutor()
    g = _guarded(fx)
    await g.close()
    assert fx.closed is True
