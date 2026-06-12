"""— execution contract + single-session workspace model + time-bomb guard.

Daemon-free (fake `ProcRunner`). Verifies the time-bomb guard (a fresh workspace per session: `start()`
removes the stale volume after gc'ing orphan containers, in that order), the contract (non-interactive
`docker exec`, one-shot result), intra-session persistence, and best-effort robustness of `start()`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import HealthStatus
from local_ai_agent.modules.exec import ProcResult
from local_ai_agent.modules.exec.module import ExecModule
from tests.test_exec_sandbox import FakeRunner

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


def _mod(runner):
    return ExecModule(Settings(**_DIRS, enable_exec=True), runner=runner)


async def test_start_removes_stale_volume_after_gc_before_create():
    fr = FakeRunner(default=ProcResult(returncode=0))
    mod = _mod(fr)
    await mod.start()
    argvs = fr.argvs
    # the time-bomb guard: a fresh workspace each session.
    assert ["docker", "volume", "rm", "local_ai_agent_ws_main"] in argvs
    # ordering: gc (ps) → volume rm → probe (info); and the volume rm precedes any container create.
    gc_i = next(i for i, a in enumerate(argvs) if a[:3] == ["docker", "ps", "-aq"])
    vol_i = argvs.index(["docker", "volume", "rm", "local_ai_agent_ws_main"])
    assert gc_i < vol_i
    assert not any(a[:3] == ["docker", "run", "-d"] for a in argvs) # no container created during start
    assert mod.health().status is HealthStatus.ok


async def test_start_is_best_effort_even_if_cleanup_errors():
    # a runner that raises on every call: start must not crash; health resolves to down (probe failed).
    fr = FakeRunner(raise_exc=RuntimeError("daemon flaky"))
    mod = _mod(fr)
    await mod.start() # must not raise
    assert mod.health().status is HealthStatus.down


async def test_run_command_exec_is_non_interactive_and_one_shot():
    # create, then exec → assert the exec argv has no -i/-t (non-interactive); result is one-shot.
    fr = FakeRunner(results=[
        ProcResult(returncode=0), # gc ps
        ProcResult(returncode=0), # volume rm
        ProcResult(returncode=0), # probe info
        ProcResult(returncode=0), # container create
        ProcResult(returncode=3, stdout=b"out", stderr=b"err"), # docker exec
    ])
    mod = _mod(fr)
    await mod.start()
    run_tool = next(t for t in mod.tools if t.name == "run_command")
    res = await run_tool.run({"command": ["sh", "-c", "exit 3"]})
    assert res["ok"] is True and res["exit_code"] == 3 # non-zero exit = normal result
    assert res["stdout"] == "out" and res["stderr"] == "err"
    exec_argv = fr.argvs[-1]
    assert exec_argv[:3] == ["docker", "exec", "local_ai_agent_exec_main"]
    assert "-i" not in exec_argv and "-t" not in exec_argv # non-interactive


async def test_intra_session_workspace_persists_across_calls():
    # within one session the same container is reused (created once) → workspace persists.
    fr = FakeRunner(default=ProcResult(returncode=0))
    mod = _mod(fr)
    await mod.start()
    write_tool = next(t for t in mod.tools if t.name == "write_workspace_file")
    read_tool = next(t for t in mod.tools if t.name == "read_workspace_file")
    await write_tool.run({"path": "a.txt", "content": "x"})
    await read_tool.run({"path": "a.txt"})
    creates = [a for a in fr.argvs if a[:3] == ["docker", "run", "-d"]]
    assert len(creates) == 1 # one session container, reused


async def test_stop_tears_down_session():
    fr = FakeRunner(default=ProcResult(returncode=0))
    mod = _mod(fr)
    await mod.start()
    run_tool = next(t for t in mod.tools if t.name == "run_command")
    await run_tool.run({"command": ["true"]}) # creates the container
    await mod.stop()
    assert ["docker", "rm", "-f", "local_ai_agent_exec_main"] in fr.argvs
    assert mod.health().status is HealthStatus.absent
