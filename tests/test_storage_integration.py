"""Integration: storage wired into the composition root.

Group A — real `build_application` wiring under `enable_storage`: tool registration, gate
safe/confirm/blocked classification over storage paths, fail-fast on bad config, flag
independence, and fault isolation (a real lazy SSH connector to an unreachable host → graceful).
Group B — composed execution through the dispatcher with a fake transport: read→safe→runs,
write→confirm, read-only refusal, unknown connector, secret→blocked.

Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.main import build_application
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import Outcome, ToolDispatcher
from local_ai_agent.modules.safety.gate import DEFAULT_SAFE_TOOLS, Action, SafetyGate, Verdict
from local_ai_agent.modules.storage.connector import GuardedConnector
from local_ai_agent.modules.storage.tools import STORAGE_SAFE_TOOL_NAMES, build_tools
from tests.test_storage_connector import FakeTransport

_DIRS = dict(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g")
_ROOT = "/srv/share"


@dataclass
class _Approver:
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


def _connectors_file(tmp_path, *, host="127.0.0.1", port=1, read_only=True):
    f = tmp_path / "storage.json"
    kh = tmp_path / "known_hosts"
    kh.write_text("", encoding="utf-8")
    key = tmp_path / "id"
    key.write_text("dummy", encoding="utf-8")
    doc = {"connectors": [{
        "name": "nas", "kind": "ssh", "host": host, "port": port, "username": "u",
        "auth": {"key_path": str(key)}, "allowed_root": "/srv/share",
        "read_only": read_only, "known_hosts_path": str(kh),
    }]}
    f.write_text(json.dumps(doc), encoding="utf-8")
    return f


# --------------------------------------------------------------------------- #
# Group A — real build_application wiring
# --------------------------------------------------------------------------- #
def test_storage_read_tools_safe_listed(tmp_path):
    app = build_application(Settings(**_DIRS, enable_agent=True, enable_storage=True,
                                     storage_connectors_file=_connectors_file(tmp_path)))
    gate = app.agent_runtime.gate
    for name in ("storage_list", "storage_stat", "storage_read"):
        assert gate.classify(Action(name, {"connector": "nas", "path": "report.txt"})).verdict is Verdict.safe, name


def test_storage_mutations_need_confirmation(tmp_path):
    app = build_application(Settings(**_DIRS, enable_agent=True, enable_storage=True,
                                     storage_connectors_file=_connectors_file(tmp_path)))
    gate = app.agent_runtime.gate
    for name in ("storage_write", "storage_delete", "storage_move"):
        assert gate.classify(Action(name, {"connector": "nas", "path": "x"})).verdict is Verdict.needs_confirmation, name


def test_secret_path_blocked(tmp_path):
    app = build_application(Settings(**_DIRS, enable_agent=True, enable_storage=True,
                                     storage_connectors_file=_connectors_file(tmp_path)))
    d = app.agent_runtime.gate.classify(Action("storage_read", {"connector": "nas", "path": "/etc/shadow"}))
    assert d.verdict is Verdict.blocked


async def test_fault_isolation_unreachable_host_graceful(tmp_path):
    # a real lazy SSH connector to a refused port → tool returns graceful unavailable, no crash
    app = build_application(Settings(**_DIRS, enable_agent=True, enable_storage=True,
                                     storage_connectors_file=_connectors_file(tmp_path)))
    res = await app.agent_runtime.dispatcher.dispatch(
        Action("storage_list", {"connector": "nas", "path": "."}), "t")
    assert res.outcome is Outcome.executed
    assert res.result["ok"] is False # connection failed → StorageUnavailable/StorageAuthError


def test_fail_fast_missing_connectors_file():
    with pytest.raises(ValueError):
        build_application(Settings(**_DIRS, enable_storage=True, storage_connectors_file=None))


def test_fail_fast_invalid_connectors_file(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        build_application(Settings(**_DIRS, enable_storage=True, storage_connectors_file=f))


def test_storage_without_agent_registers_module_no_runtime(tmp_path):
    app = build_application(Settings(**_DIRS, enable_agent=False, enable_storage=True,
                                     storage_connectors_file=_connectors_file(tmp_path)))
    assert app.agent_runtime is None
    known = {c for m in app.registry.modules for c in m.spec.capabilities}
    assert "storage" in known


def test_default_safe_tools_unchanged(tmp_path):
    gate = build_application(Settings(**_DIRS, enable_agent=True, enable_storage=True,
                                      storage_connectors_file=_connectors_file(tmp_path))).agent_runtime.gate
    assert gate.classify(Action("read_file", {"path": "x"})).verdict is Verdict.safe
    assert gate.classify(Action("write_file", {"path": "x"})).verdict is Verdict.needs_confirmation


# --------------------------------------------------------------------------- #
# Group B — composed execution through the dispatcher with a fake transport
# --------------------------------------------------------------------------- #
def _composed(*, read_only=True):
    t = FakeTransport(files={f"{_ROOT}/a.txt": b"hello"}, dirs={_ROOT})
    connectors = {"nas": GuardedConnector(t, allowed_root=_ROOT, read_only=read_only, max_bytes=10_000_000)}
    gate = SafetyGate(safe_tools=DEFAULT_SAFE_TOOLS | STORAGE_SAFE_TOOL_NAMES)
    return ToolDispatcher(gate=gate, approvals=PendingApprovals(timeout_seconds=300),
                          tools=build_tools(connectors))


async def test_read_executes_via_dispatcher():
    disp = _composed()
    listed = await disp.dispatch(Action("storage_list", {"connector": "nas", "path": "."}), "t")
    assert listed.outcome is Outcome.executed and listed.result["ok"] is True
    assert "a.txt" in {e["path"] for e in listed.result["entries"]}
    read = await disp.dispatch(Action("storage_read", {"connector": "nas", "path": "a.txt"}), "t")
    assert read.outcome is Outcome.executed and read.result["text"] == "hello" and read.result["bytes"] == 5


async def test_unknown_connector_graceful():
    disp = _composed()
    res = await disp.dispatch(Action("storage_read", {"connector": "ghost", "path": "a.txt"}), "t")
    assert res.outcome is Outcome.executed and res.result["ok"] is False
    assert "unknown connector" in res.result["error"]


async def test_write_to_read_only_refused_after_approval():
    disp = _composed(read_only=True)
    action = Action("storage_write", {"connector": "nas", "path": "b.txt", "content": "x"})
    pending = await disp.dispatch(action, "t")
    assert pending.outcome is Outcome.pending # mutation → confirm
    disp.approvals.approve(pending.approval_id, action, _Approver())
    done = await disp.execute_approved(pending.approval_id, action)
    assert done.outcome is Outcome.executed
    assert done.result["ok"] is False and "StorageReadOnly" in done.result["error"]


async def test_write_to_writable_connector():
    disp = _composed(read_only=False)
    action = Action("storage_write", {"connector": "nas", "path": "b.txt", "content": "data"})
    pending = await disp.dispatch(action, "t")
    assert pending.outcome is Outcome.pending
    disp.approvals.approve(pending.approval_id, action, _Approver())
    done = await disp.execute_approved(pending.approval_id, action)
    assert done.outcome is Outcome.executed and done.result["ok"] is True


async def test_secret_path_refused_via_dispatcher():
    disp = _composed()
    res = await disp.dispatch(Action("storage_read", {"connector": "nas", "path": "/etc/shadow"}), "t")
    assert res.outcome is Outcome.refused and res.result is None


async def test_relative_secret_escape_refused_via_dispatcher():
    # a relative escape that canonicalizes onto a secret path is still blocked end-to-end
    disp = _composed()
    res = await disp.dispatch(Action("storage_read", {"connector": "nas", "path": "../../etc/shadow"}), "t")
    assert res.outcome is Outcome.refused and res.result is None


@pytest.mark.parametrize("tool", ["storage_list", "storage_stat", "storage_read"])
async def test_unknown_connector_graceful_all_read_tools(tool):
    disp = _composed()
    res = await disp.dispatch(Action(tool, {"connector": "ghost", "path": "a.txt"}), "t")
    assert res.outcome is Outcome.executed and res.result["ok"] is False
    assert "unknown connector" in res.result["error"]
