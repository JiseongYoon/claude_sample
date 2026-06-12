"""— exec-broker argv builders + DockerSandbox (network-free, daemon-free).

A fake `ProcRunner` records every docker argv it is asked to run and returns canned `ProcResult`s, so
tests assert the EXACT hardening flags emitted (and that no forbidden flag ever appears), plus the
container lifecycle, timeout→kill, output bounding, no-leak error mapping, and best-effort cleanup. No
real Docker — the daemon round-trip is exercised only by `scripts/smoke_exec.py`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.modules.exec import (
    DockerSandbox,
    ExecBlocked,
    ExecConfig,
    ExecError,
    ExecPolicy,
    ExecTimeout,
    ExecTooLarge,
    ExecUnavailable,
    Mount,
    ProcResult,
    SandboxSpec,
    build_run_argv,
    gc_orphans,
    probe,
)
from local_ai_agent.modules.exec.sandbox import FORBIDDEN_ARGV_TOKENS

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _spec(**overrides) -> SandboxSpec:
    cfg = ExecConfig.from_settings(Settings(**_DIRS, exec_env_whitelist=["LANG"]))
    spec = ExecPolicy(cfg, environ={"LANG": "C.UTF-8"}).build_spec(workspace_volume="local_ai_agent_ws_t1")
    if overrides:
        from dataclasses import replace
        spec = replace(spec, **overrides)
    return spec


# --------------------------------------------------------------------------- #
# fake runner
# --------------------------------------------------------------------------- #
class FakeRunner:
    """Records (argv, stdin, timeout, max_output_bytes); returns a queue of canned results (or a default).
    `raise_exc` simulates a runner-level crash for no-leak testing."""

    def __init__(self, results=None, default=None, raise_exc=None):
        self.calls: list[dict] = []
        self._results = list(results or [])
        self._default = default or ProcResult(returncode=0, stdout=b"")
        self._raise = raise_exc

    async def __call__(self, argv, *, stdin=None, timeout=None, max_output_bytes=None):
        self.calls.append(dict(argv=list(argv), stdin=stdin, timeout=timeout, cap=max_output_bytes))
        if self._raise is not None:
            raise self._raise
        return self._results.pop(0) if self._results else self._default

    @property
    def argvs(self):
        return [c["argv"] for c in self.calls]


# --------------------------------------------------------------------------- #
# normal — argv builders
# --------------------------------------------------------------------------- #
def test_build_run_argv_emits_full_hardening():
    a = build_run_argv(_spec(), name="local_ai_agent_exec_t1")
    s = " ".join(a)
    assert a[:5] == ["docker", "run", "-d", "--name", "local_ai_agent_exec_t1"]
    assert "--network none" in s and "--read-only" in s
    assert "--tmpfs" in a and any("noexec,nosuid" in x for x in a)
    assert any(x.startswith("type=volume,src=local_ai_agent_ws_t1,dst=/workspace") for x in a)
    assert "--workdir" in a and "/workspace" in a
    assert ["--user", "1000:1000"] == [a[a.index("--user")], a[a.index("--user") + 1]]
    assert "--cap-drop" in a and a[a.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in a and "--init" in a
    assert "--memory" in a and "--memory-swap" in a
    assert a[a.index("--memory") + 1] == a[a.index("--memory-swap") + 1] # no swap escape
    assert "--cpus" in a and "--pids-limit" in a
    assert any(x.startswith("nofile=") for x in a) and any(x.startswith("fsize=") for x in a)
    assert "--env" in a and "LANG=C.UTF-8" in a
    assert a[-2:] == ["sleep", "infinity"]


def test_build_run_argv_emits_no_forbidden_flag():
    a = build_run_argv(_spec(), name="local_ai_agent_exec_t1")
    for tok in FORBIDDEN_ARGV_TOKENS:
        assert tok not in a
    assert not any("docker.sock" in x for x in a)
    assert "host" not in a[a.index("--network") + 1:a.index("--network") + 2] # --network none, not host


async def test_run_creates_once_then_execs():
    fr = FakeRunner(results=[
        ProcResult(returncode=0, stdout=b"cid"), # docker run -d
        ProcResult(returncode=0, stdout=b"hello"), # docker exec
        ProcResult(returncode=0, stdout=b"world"), # docker exec again (no re-create)
    ])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    spec = _spec()
    r1 = await sb.run(spec, ["echo", "hello"], timeout=5, max_output_bytes=1000)
    r2 = await sb.run(spec, ["echo", "world"], timeout=5, max_output_bytes=1000)
    assert r1.stdout == b"hello" and r2.stdout == b"world"
    assert fr.argvs[0][:3] == ["docker", "run", "-d"] # created once
    assert fr.argvs[1][:2] == ["docker", "exec"] and fr.argvs[2][:2] == ["docker", "exec"]
    assert sum(1 for av in fr.argvs if av[:3] == ["docker", "run", "-d"]) == 1


async def test_read_and_write_file_argv():
    fr = FakeRunner(results=[
        ProcResult(returncode=0), # create
        ProcResult(returncode=0, stdout=b"file-bytes"), # read (cat)
        ProcResult(returncode=0), # create cached → next is write
    ])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    spec = _spec()
    data = await sb.read_file(spec, "/workspace/a.txt", max_bytes=1000)
    assert data == b"file-bytes"
    assert fr.argvs[1] == ["docker", "exec", "local_ai_agent_exec_t1", "cat", "--", "/workspace/a.txt"]
    await sb.write_file(spec, "/workspace/b.txt", b"payload")
    wcall = fr.calls[-1]
    assert wcall["argv"][:4] == ["docker", "exec", "-i", "local_ai_agent_exec_t1"]
    assert wcall["stdin"] == b"payload"


async def test_close_removes_container_and_volume():
    fr = FakeRunner()
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    await sb.run(_spec(), ["true"], timeout=5, max_output_bytes=100) # _ensure records the volume
    await sb.close()
    assert ["docker", "rm", "-f", "local_ai_agent_exec_t1"] in fr.argvs
    assert ["docker", "volume", "rm", "local_ai_agent_ws_t1"] in fr.argvs


async def test_gc_orphans_lists_and_removes():
    fr = FakeRunner(results=[ProcResult(returncode=0, stdout=b"id1\nid2\n")])
    removed = await gc_orphans(fr)
    assert removed == ["id1", "id2"]
    assert fr.argvs[0][:3] == ["docker", "ps", "-aq"]
    assert ["docker", "rm", "-f", "id1"] in fr.argvs and ["docker", "rm", "-f", "id2"] in fr.argvs


async def test_probe_true_on_zero_exit():
    assert await probe(FakeRunner(default=ProcResult(returncode=0))) is True
    assert await probe(FakeRunner(default=ProcResult(returncode=1))) is False
    assert await probe(FakeRunner(raise_exc=OSError("no daemon"))) is False


# --------------------------------------------------------------------------- #
# error / security
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {"network": "host"}, {"network": "bridge"}, {"read_only": False}, {"cap_drop": ()},
    {"cap_drop": ("NET_ADMIN",)}, {"no_new_privileges": False}, {"user": "root"}, {"user": "0:0"},
    {"user": ""}, {"image": ""}, {"mounts": (Mount(volume="v", target="/var/run/docker.sock"),)},
    {"mounts": (Mount(volume="v", target="/"),)},
])
def test_build_run_argv_refuses_non_hardened_spec(bad):
    with pytest.raises(ExecBlocked):
        build_run_argv(_spec(**bad), name="local_ai_agent_exec_t1")


def test_build_run_argv_rejects_bad_name():
    for nm in ("", " ", None, 5):
        with pytest.raises(ExecBlocked):
            build_run_argv(_spec(), name=nm)


@pytest.mark.parametrize("bad", [
    {"image": "--privileged"}, {"image": "-v"}, {"image": "--cap-add"}, {"image": "--network"},
    {"image": "--pid=host"}, {"user": "--privileged"}, {"workdir": "--volume"},
])
def test_build_run_argv_refuses_option_shaped_fields(bad):
    # an option-shaped image/user/workdir would be parsed by docker as a flag → refused.
    with pytest.raises(ExecBlocked):
        build_run_argv(_spec(**bad), name="local_ai_agent_exec_t1")


def test_build_run_argv_separates_image_with_double_dash():
    a = build_run_argv(_spec(), name="local_ai_agent_exec_t1")
    # the tail is exactly: -- <image> sleep infinity (the `--` terminates docker option parsing)
    assert a[-4:] == ["--", "local_ai_agent_exec:base", "sleep", "infinity"]


async def test_timeout_kills_container_and_raises():
    fr = FakeRunner(results=[
        ProcResult(returncode=0), # create
        ProcResult(returncode=-1, timed_out=True), # exec times out
    ])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    with pytest.raises(ExecTimeout):
        await sb.run(_spec(), ["sleep", "999"], timeout=1, max_output_bytes=100)
    assert ["docker", "kill", "local_ai_agent_exec_t1"] in fr.argvs


async def test_create_failure_maps_to_unavailable_no_leak():
    secret = "Error: /home/user/.ssh/id_rsa token=ABC123 10.0.0.5"
    fr = FakeRunner(results=[ProcResult(returncode=125, stderr=secret.encode())])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    with pytest.raises(ExecUnavailable) as ei:
        await sb.run(_spec(), ["echo", "x"], timeout=5, max_output_bytes=100)
    msg = str(ei.value)
    assert "id_rsa" not in msg and "ABC123" not in msg and "user" not in msg and "10.0.0.5" not in msg


async def test_runner_crash_maps_to_unavailable_no_leak():
    fr = FakeRunner(raise_exc=RuntimeError("/home/user/.ssh/id_rsa leaked"))
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    with pytest.raises(ExecUnavailable) as ei:
        await sb.run(_spec(), ["echo", "x"], timeout=5, max_output_bytes=100)
    assert "RuntimeError" in str(ei.value) and "id_rsa" not in str(ei.value)


async def test_run_nonzero_exit_is_a_result_not_an_error():
    # a command failing inside the sandbox (rc != 0) is a normal RawExecResult, not an exception.
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=2, stderr=b"boom")])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    res = await sb.run(_spec(), ["false"], timeout=5, max_output_bytes=100)
    assert res.exit_code == 2 and res.stderr == b"boom"


async def test_output_bounding():
    big = b"x" * 5000
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0, stdout=big)])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    res = await sb.run(_spec(), ["yes"], timeout=5, max_output_bytes=1000)
    assert len(res.stdout) == 1000 and res.truncated is True


async def test_read_too_large_raises():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0, stdout=b"y" * 2000)])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    with pytest.raises(ExecTooLarge):
        await sb.read_file(_spec(), "/workspace/big.bin", max_bytes=1000)


async def test_read_write_nonzero_exit_raises_exec_error():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=1, stderr=b"no such file")])
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    with pytest.raises(ExecError):
        await sb.read_file(_spec(), "/workspace/missing", max_bytes=100)


async def test_cleanup_never_raises_even_if_runner_errors():
    fr = FakeRunner(raise_exc=RuntimeError("daemon gone"))
    sb = DockerSandbox(name="local_ai_agent_exec_t1", runner=fr)
    sb._created = True # pretend created so close attempts rm
    sb._volume = "local_ai_agent_ws_t1"
    await sb.close() # must not raise
    assert await gc_orphans(fr) == [] # gc swallows the crash too


def test_ctor_rejects_bad_name():
    for nm in ("", " ", None):
        with pytest.raises(ValueError):
            DockerSandbox(name=nm)
