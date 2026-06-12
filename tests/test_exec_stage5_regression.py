"""— exec capability regression against the user-chosen behavioral mechanism.

Verifies the steps work as an INTEGRATED WHOLE through the composed, gated dispatcher (network-
free, daemon-free, fake `ProcRunner`). The user-chosen "권장 기본 세트": gate routing · double
containment end-to-end · fail-closed (Docker down) · time-bomb fresh-workspace · output cap + timeout
→kill · confirm→approve→execute HITL · flag isolation. The real-Docker integrated path is separately
proven by `scripts/smoke_exec.py` (operator-run, PASS 12/12).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import HealthStatus
from local_ai_agent.main import build_application
from local_ai_agent.modules.exec import ProcResult
from local_ai_agent.modules.exec.module import ExecModule
from local_ai_agent.modules.exec.tools import EXEC_SAFE_TOOL_NAMES
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from tests.test_exec_sandbox import FakeRunner

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


async def _composed(runner, **settings_kw):
    """An ExecModule (fake runner) on a real gate+dispatcher — the composed gated path; started."""
    mod = ExecModule(Settings(**_DIRS, enable_exec=True, **settings_kw), runner=runner)
    await mod.start()
    gate = SafetyGate(safe_tools=frozenset(set(DEFAULT_SAFE_TOOLS) | EXEC_SAFE_TOOL_NAMES))
    disp = ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=[])
    for t in mod.tools:
        disp.register(t)
    return mod, disp


async def _run_gated(disp, action):
    d = await disp.dispatch(action, "t")
    if d.outcome is Outcome.pending:
        disp.approvals.approve(d.approval_id, action, _Approver())
        return await disp.execute_approved(d.approval_id, action)
    return d


# ① gate routing -------------------------------------------------------------- #
def test_gate_routing():
    gate = build_application(Settings(**_DIRS, enable_agent=True, enable_exec=True)).agent_runtime.gate
    assert gate.classify(Action("run_command", {"command": ["ls"]})).verdict is Verdict.needs_confirmation
    assert gate.classify(Action("write_workspace_file", {"path": "a"})).verdict is Verdict.needs_confirmation
    assert gate.classify(Action("read_workspace_file", {"path": "a"})).verdict is Verdict.safe
    # a plain absolute / escaping path is never auto-safe (nonlocal-path rule) → confirm …
    assert gate.classify(Action("read_workspace_file", {"path": "/data/x"})).verdict is Verdict.needs_confirmation
    # … and a secret path is outright blocked (stronger — the gate's secret-read rule)
    assert gate.classify(Action("read_workspace_file", {"path": "/etc/shadow"})).verdict is Verdict.blocked


# ② double containment end-to-end --------------------------------------------- #
async def test_double_containment_escape_refused_after_approval():
    fr = FakeRunner(default=ProcResult(returncode=0))
    _, disp = await _composed(fr)
    before = len(fr.calls)
    done = await _run_gated(disp, Action("read_workspace_file", {"path": "../../etc/passwd"}))
    assert done.outcome is Outcome.executed and done.result["ok"] is False # containment refused
    assert len(fr.calls) == before # executor never reached


# ③ fail-closed --------------------------------------------------------------- #
async def test_fail_closed_health_and_graceful_tool():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0), # gc, volume rm
                             ProcResult(returncode=1)]) # probe (info) fails
    mod, disp = await _composed(fr)
    assert mod.health().status is HealthStatus.down # capability not served
    # a daemon-error command still degrades gracefully with no leak
    fr2 = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0), ProcResult(returncode=0),
                              ProcResult(returncode=125, stderr=b"connect /var/run/docker.sock denied")])
    _, disp2 = await _composed(fr2)
    done = await _run_gated(disp2, Action("run_command", {"command": ["echo", "x"]}))
    assert done.result["ok"] is False and "docker.sock" not in done.result["error"]


# ④ time-bomb guard ----------------------------------------------------------- #
async def test_time_bomb_fresh_workspace_on_start():
    fr = FakeRunner(default=ProcResult(returncode=0))
    await _composed(fr)
    assert ["docker", "volume", "rm", "local_ai_agent_ws_main"] in fr.argvs


# ⑤ output cap + timeout → kill ----------------------------------------------- #
async def test_output_capped():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0), ProcResult(returncode=0),
                             ProcResult(returncode=0), # create
                             ProcResult(returncode=0, stdout=b"y" * 5000)]) # exec flood
    _, disp = await _composed(fr, exec_max_output_bytes=100)
    done = await _run_gated(disp, Action("run_command", {"command": ["yes"]}))
    assert done.result["ok"] is True and len(done.result["stdout"]) == 100 and done.result["truncated"]


async def test_timeout_kills_and_degrades_gracefully():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0), ProcResult(returncode=0),
                             ProcResult(returncode=0), # create
                             ProcResult(returncode=-1, timed_out=True)]) # exec times out
    _, disp = await _composed(fr)
    done = await _run_gated(disp, Action("run_command", {"command": ["sleep", "999"]}))
    assert done.result["ok"] is False and "ExecTimeout" in done.result["error"]
    assert ["docker", "kill", "local_ai_agent_exec_main"] in fr.argvs


# ⑥ confirm→approve→execute HITL --------------------------------------------- #
async def test_hitl_round_trip_run_and_read():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0), ProcResult(returncode=0),
                             ProcResult(returncode=0), # create
                             ProcResult(returncode=0, stdout=b"done"), # run exec
                             ProcResult(returncode=0, stdout=b"contents")]) # read cat
    _, disp = await _composed(fr)
    run = await _run_gated(disp, Action("run_command", {"command": ["echo", "done"]}))
    assert run.outcome is Outcome.executed and run.result["stdout"] == "done"
    # read is safe-listed → executes WITHOUT an approval
    rd = await disp.dispatch(Action("read_workspace_file", {"path": "out.txt"}), "t")
    assert rd.outcome is Outcome.executed and rd.result["content"] == "contents"


# ⑦ flag isolation ------------------------------------------------------------ #
def test_flag_isolation_off_by_default():
    app = build_application(Settings(**_DIRS, enable_agent=True)) # enable_exec False
    assert "exec" not in {m.spec.name for m in app.modules}
    disp = app.agent_runtime.dispatcher
    for name in ("run_command", "read_workspace_file", "write_workspace_file"):
        assert name not in disp._tools
    # safe-list unchanged: read_workspace_file is not auto-safe when exec is off
    assert app.agent_runtime.gate.classify(
        Action("read_workspace_file", {"path": "a"})).verdict is Verdict.needs_confirmation
