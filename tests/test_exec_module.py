"""— ExecModule + gated tools + enable_exec wiring (network-free, daemon-free).

Two groups:
  * **A — real `build_application` wiring**: the exec tools are registered on the dispatcher;
    gate classification is correct (`run_command` → confirm via `_SHELL_TOOLS`, `write_workspace_file`
    → confirm via `_MUTATING_FILE_TOOLS`, `read_workspace_file` safe-listed); `enable_exec` builds the
    module + safety chain. No Docker (build doesn't start the module).
  * **B — composed execution through the dispatcher with an injected fake runner**: safe read runs
    directly; gated run/write need confirm→approve→execute; **double containment** (a workspace-escape
    path is refused by containment even after the gate approves); fail-closed (Docker unreachable →
    health `down` + graceful tool error). No real Docker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.core.module import HealthStatus
from local_ai_agent.main import build_application
from local_ai_agent.modules.exec import ExecConfig, ProcResult
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


def _app(**kw):
    return build_application(Settings(**_DIRS, enable_agent=True, enable_exec=True, **kw))


def _disp(runner):
    """A dispatcher+gate with an ExecModule backed by a fake runner (composed exec, no Docker)."""
    mod = ExecModule(Settings(**_DIRS, enable_exec=True), runner=runner)
    gate = SafetyGate(safe_tools=frozenset(set(DEFAULT_SAFE_TOOLS) | EXEC_SAFE_TOOL_NAMES))
    disp = ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300), tools=[])
    for t in mod.tools:
        disp.register(t)
    return mod, disp


async def _confirm_run(disp, action):
    """dispatch → (if gated) approve → execute; returns the final DispatchResult."""
    d = await disp.dispatch(action, "t")
    if d.outcome is Outcome.pending:
        disp.approvals.approve(d.approval_id, action, _Approver())
        return await disp.execute_approved(d.approval_id, action)
    return d


# --------------------------------------------------------------------------- #
# Group A — real build_application wiring
# --------------------------------------------------------------------------- #
def test_exec_module_present_and_spec():
    app = _app()
    mods = {m.spec.name: m for m in app.modules}
    assert "exec" in mods
    spec = mods["exec"].spec
    assert spec.capabilities == ("exec",) and spec.depends_on == ()


def test_exec_tools_registered_on_dispatcher():
    disp = _app().agent_runtime.dispatcher
    for name in ("run_command", "read_workspace_file", "write_workspace_file"):
        assert name in disp._tools


def test_gate_classification():
    gate = _app().agent_runtime.gate
    assert gate.classify(Action("run_command", {"command": ["echo", "hi"]})).verdict is Verdict.needs_confirmation
    assert gate.classify(Action("write_workspace_file", {"path": "a.txt"})).verdict is Verdict.needs_confirmation
    assert gate.classify(Action("read_workspace_file", {"path": "a.txt"})).verdict is Verdict.safe
    # an absolute / escaping read path is NOT auto-safe (nonlocal-path rule) → confirm
    assert gate.classify(Action("read_workspace_file", {"path": "/etc/passwd"})).verdict is Verdict.needs_confirmation


def test_module_disabled_by_default():
    app = build_application(Settings(**_DIRS)) # enable_exec defaults False
    assert "exec" not in {m.spec.name for m in app.modules}


# --------------------------------------------------------------------------- #
# Group B — composed execution with a fake runner
# --------------------------------------------------------------------------- #
async def test_module_health_lifecycle_ok():
    mod = ExecModule(Settings(**_DIRS, enable_exec=True), runner=FakeRunner(default=ProcResult(returncode=0)))
    assert mod.health().status is HealthStatus.absent # not started
    await mod.start()
    assert mod.health().status is HealthStatus.ok # docker info rc 0 → reachable
    await mod.stop()
    assert mod.health().status is HealthStatus.absent


async def test_module_health_fail_closed_when_docker_down():
    mod = ExecModule(Settings(**_DIRS, enable_exec=True), runner=FakeRunner(default=ProcResult(returncode=1)))
    await mod.start()
    assert mod.health().status is HealthStatus.down # docker unreachable → fail-closed


async def test_module_tools_names():
    mod = ExecModule(Settings(**_DIRS, enable_exec=True), runner=FakeRunner())
    assert {t.name for t in mod.tools} == {"run_command", "read_workspace_file", "write_workspace_file"}


async def test_read_workspace_file_runs_safe_listed():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0, stdout=b"file-content")])
    _, disp = _disp(fr)
    res = await disp.dispatch(Action("read_workspace_file", {"path": "notes/a.txt"}), "t")
    assert res.outcome is Outcome.executed # safe-listed → no approval
    assert res.result["ok"] is True and res.result["content"] == "file-content"


async def test_run_command_gated_then_executes():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0, stdout=b"hello")])
    _, disp = _disp(fr)
    action = Action("run_command", {"command": ["echo", "hello"]})
    d = await disp.dispatch(action, "t")
    assert d.outcome is Outcome.pending # gated (confirm) — INV: entry permission
    disp.approvals.approve(d.approval_id, action, _Approver())
    done = await disp.execute_approved(d.approval_id, action)
    assert done.outcome is Outcome.executed
    assert done.result["ok"] is True and done.result["exit_code"] == 0 and done.result["stdout"] == "hello"


async def test_write_workspace_file_gated_then_executes():
    fr = FakeRunner(results=[ProcResult(returncode=0), ProcResult(returncode=0)])
    _, disp = _disp(fr)
    action = Action("write_workspace_file", {"path": "out/b.txt", "content": "data"})
    d = await disp.dispatch(action, "t")
    assert d.outcome is Outcome.pending
    disp.approvals.approve(d.approval_id, action, _Approver())
    done = await disp.execute_approved(d.approval_id, action)
    assert done.outcome is Outcome.executed and done.result["ok"] is True
    assert fr.calls[-1]["stdin"] == b"data" # data piped to the sandbox


async def test_double_containment_escape_refused_even_after_approval():
    fr = FakeRunner(results=[ProcResult(returncode=0)])
    _, disp = _disp(fr)
    action = Action("read_workspace_file", {"path": "/etc/passwd"}) # absolute → gate confirms
    d = await disp.dispatch(action, "t")
    assert d.outcome is Outcome.pending
    disp.approvals.approve(d.approval_id, action, _Approver())
    done = await disp.execute_approved(d.approval_id, action)
    # approved at the gate, but CONTAINMENT still refuses → graceful {ok: False}; executor never reached
    assert done.outcome is Outcome.executed and done.result["ok"] is False
    assert fr.calls == [] # contain_workspace_path raised before any docker call


async def test_run_command_bad_argv_graceful():
    _, disp = _disp(FakeRunner())
    action = Action("run_command", {"command": "echo hi"}) # a bare string is not an argv list
    done = await _confirm_run(disp, action)
    assert done.outcome is Outcome.executed and done.result["ok"] is False


async def test_run_command_docker_down_graceful():
    # create fails (daemon error) → typed ExecUnavailable → graceful tool error, loop survives.
    fr = FakeRunner(results=[ProcResult(returncode=125, stderr=b"Cannot connect to the Docker daemon")])
    _, disp = _disp(fr)
    action = Action("run_command", {"command": ["echo", "x"]})
    done = await _confirm_run(disp, action)
    assert done.outcome is Outcome.executed and done.result["ok"] is False
    assert "Cannot connect" not in done.result["error"] # no daemon-detail leak
