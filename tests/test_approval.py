"""Tests for the HITL PendingApprovals state machine.

Pure unit tests — deterministic via an injected counter-clock and a sequential id
factory. Run in conda `local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from local_ai_agent.modules.safety.approval import (
    ActionMismatch,
    ApprovalNotFound,
    ApprovalState,
    ApprovalStateError,
    Approver,
    NotApprovable,
    NotAuthorized,
    PendingApprovals,
)
from local_ai_agent.modules.safety.gate import Action, Decision, Verdict


# --------------------------------------------------------------------------- #
# test doubles
# --------------------------------------------------------------------------- #
class FakeClock:
    """A controllable monotonic-ish clock: `t` advances only when told to."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class SeqIds:
    """Deterministic, unique id factory."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"appr-{self.n:04d}"


@dataclass
class FakePrincipal:
    """Structurally satisfies the Approver protocol."""

    subject: str = "operator"
    scopes: frozenset[str] = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


# convenience verdicts
def _confirm(reason: str = "needs human ok") -> Decision:
    return Decision(Verdict.needs_confirmation, reason, "confirm.rule")


def _blocked() -> Decision:
    return Decision(Verdict.blocked, "catastrophic", "block.rule")


def _safe() -> Decision:
    return Decision(Verdict.safe, "read-only", "safe.allowlist")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def reg(clock: FakeClock) -> PendingApprovals:
    return PendingApprovals(timeout_seconds=300.0, now=clock, id_factory=SeqIds())


@pytest.fixture
def approver() -> FakePrincipal:
    return FakePrincipal(subject="alice")


DELETE = Action("delete_file", {"path": "workspace/report.txt"})


# --------------------------------------------------------------------------- #
# NORMAL class
# --------------------------------------------------------------------------- #
def test_request_creates_pending_with_expiry(reg, clock):
    rec = reg.request(DELETE, _confirm(), requested_by="agent")
    assert rec.status is ApprovalState.pending
    assert rec.approval_id == "appr-0001"
    assert rec.created_at == clock.t
    assert rec.expires_at == clock.t + 300.0
    assert rec.requested_by == "agent"
    assert rec.verdict is Verdict.needs_confirmation


def test_happy_path_request_approve_consume(reg, approver):
    rec = reg.request(DELETE, _confirm(), requested_by="agent")
    appr = reg.approve(rec.approval_id, DELETE, approver)
    assert appr.status is ApprovalState.approved
    assert appr.decided_by == "alice"
    consumed = reg.consume(rec.approval_id, DELETE)
    assert consumed.status is ApprovalState.consumed


def test_deny_on_pending(reg, approver):
    rec = reg.request(DELETE, _confirm(), requested_by="agent")
    denied = reg.deny(rec.approval_id, approver)
    assert denied.status is ApprovalState.denied
    assert denied.decided_by == "alice"


def test_deny_on_approved_revokes_before_run(reg, approver):
    rec = reg.request(DELETE, _confirm(), requested_by="agent")
    reg.approve(rec.approval_id, DELETE, approver)
    denied = reg.deny(rec.approval_id)
    assert denied.status is ApprovalState.denied
    # and it can no longer be consumed
    with pytest.raises(ApprovalStateError):
        reg.consume(rec.approval_id, DELETE)


def test_pending_lists_only_live(reg, approver, clock):
    a = reg.request(Action("delete_file", {"path": "a"}), _confirm(), "agent")
    b = reg.request(Action("delete_file", {"path": "b"}), _confirm(), "agent")
    reg.approve(b.approval_id, Action("delete_file", {"path": "b"}), approver)
    ids = {r.approval_id for r in reg.pending()}
    assert ids == {a.approval_id} # b is approved, not pending


def test_sweep_expires_and_counts(reg, clock):
    reg.request(Action("delete_file", {"path": "a"}), _confirm(), "agent")
    reg.request(Action("delete_file", {"path": "b"}), _confirm(), "agent")
    clock.advance(301)
    assert reg.sweep() == 2
    assert reg.pending() == []


def test_various_timeouts_construct(clock):
    for t in (0.001, 1.0, 300.0, 86400.0):
        PendingApprovals(timeout_seconds=t, now=clock, id_factory=SeqIds())


# --------------------------------------------------------------------------- #
# ERROR class — every path must be non-runnable (fail-closed)
# --------------------------------------------------------------------------- #
def test_replay_consume_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    reg.consume(rec.approval_id, DELETE)
    with pytest.raises(ApprovalStateError):
        reg.consume(rec.approval_id, DELETE)


def test_double_approve_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    with pytest.raises(ApprovalStateError):
        reg.approve(rec.approval_id, DELETE, approver)


def test_approve_after_consume_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    reg.consume(rec.approval_id, DELETE)
    with pytest.raises(ApprovalStateError):
        reg.approve(rec.approval_id, DELETE, approver)


def test_approve_after_deny_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.deny(rec.approval_id)
    with pytest.raises(ApprovalStateError):
        reg.approve(rec.approval_id, DELETE, approver)


def test_consume_before_approve_rejected(reg):
    rec = reg.request(DELETE, _confirm(), "agent")
    with pytest.raises(ApprovalStateError):
        reg.consume(rec.approval_id, DELETE)


def test_action_substitution_at_consume_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    other = Action("delete_file", {"path": "workspace/OTHER.txt"})
    with pytest.raises(ActionMismatch):
        reg.consume(rec.approval_id, other)


def test_action_substitution_at_approve_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    other = Action("delete_file", {"path": "/etc/passwd"})
    with pytest.raises(ActionMismatch):
        reg.approve(rec.approval_id, other, approver)


def test_tool_substitution_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    # same args, different tool → different fingerprint
    with pytest.raises(ActionMismatch):
        reg.consume(rec.approval_id, Action("shell", {"path": "workspace/report.txt"}))


def test_blocked_never_enrollable(reg):
    with pytest.raises(NotApprovable):
        reg.request(DELETE, _blocked(), "agent")


def test_safe_never_enrollable(reg):
    with pytest.raises(NotApprovable):
        reg.request(DELETE, _safe(), "agent")


def test_timeout_then_approve_rejected(reg, approver, clock):
    rec = reg.request(DELETE, _confirm(), "agent")
    clock.advance(300) # exactly at expires_at → boundary favors expiry
    with pytest.raises(ApprovalStateError):
        reg.approve(rec.approval_id, DELETE, approver)
    assert reg.get(rec.approval_id).status is ApprovalState.expired


def test_timeout_boundary_is_inclusive(reg, approver, clock):
    rec = reg.request(DELETE, _confirm(), "agent")
    clock.advance(299.999) # just before → still approvable
    reg.approve(rec.approval_id, DELETE, approver)
    assert reg.get(rec.approval_id).status is ApprovalState.approved


def test_approved_then_expire_does_not_block_consume(reg, approver, clock):
    # expiry only acts on pending; once approved, the human decision stands
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    clock.advance(10_000)
    consumed = reg.consume(rec.approval_id, DELETE)
    assert consumed.status is ApprovalState.consumed


def test_missing_scope_rejected(reg):
    rec = reg.request(DELETE, _confirm(), "agent")
    weak = FakePrincipal(subject="bob", scopes=frozenset({"read", "invoke"}))
    with pytest.raises(NotAuthorized):
        reg.approve(rec.approval_id, DELETE, weak)


def test_wildcard_scope_allowed(reg):
    rec = reg.request(DELETE, _confirm(), "agent")
    admin = FakePrincipal(subject="root", scopes=frozenset({"*"}))
    appr = reg.approve(rec.approval_id, DELETE, admin)
    assert appr.status is ApprovalState.approved


def test_unknown_id_rejected(reg, approver):
    with pytest.raises(ApprovalNotFound):
        reg.approve("appr-9999", DELETE, approver)
    with pytest.raises(ApprovalNotFound):
        reg.consume("forged", DELETE)
    assert reg.get("nope") is None


def test_nonpositive_timeout_rejected(clock):
    with pytest.raises(ValueError):
        PendingApprovals(timeout_seconds=0, now=clock)
    with pytest.raises(ValueError):
        PendingApprovals(timeout_seconds=-5, now=clock)


def test_unfingerprintable_action_rejected(reg):
    # a set is not JSON-serializable → cannot be bound → fail-closed
    bad = Action("delete_file", {"path": {1, 2, 3}})
    with pytest.raises(NotApprovable):
        reg.request(bad, _confirm(), "agent")


def test_arg_order_independent_but_value_bound(reg, approver):
    # dict key order must not matter (canonical sort), but values must match exactly
    a1 = Action("storage_write", {"path": "x", "data": "hello"})
    a2 = Action("storage_write", {"data": "hello", "path": "x"}) # same, reordered
    a3 = Action("storage_write", {"path": "x", "data": "HELLO"}) # different value
    rec = reg.request(a1, _confirm(), "agent")
    reg.approve(rec.approval_id, a2, approver) # reordered → matches
    with pytest.raises(ActionMismatch):
        reg.consume(rec.approval_id, a3)


def test_deny_on_consumed_rejected(reg, approver):
    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    reg.consume(rec.approval_id, DELETE)
    with pytest.raises(ApprovalStateError):
        reg.deny(rec.approval_id)


# --- hardening regressions (verifying-code round-1 findings) --------------- #
def test_record_is_frozen_no_replay_via_mutation(reg, approver):
    """F1: a returned record is immutable, so a caller cannot flip a consumed grant
    back to approved and replay it."""
    import dataclasses as _dc

    rec = reg.request(DELETE, _confirm(), "agent")
    reg.approve(rec.approval_id, DELETE, approver)
    consumed = reg.consume(rec.approval_id, DELETE)
    with pytest.raises(_dc.FrozenInstanceError):
        consumed.status = ApprovalState.approved
    # registry state is untouched → replay still fails
    with pytest.raises(ApprovalStateError):
        reg.consume(rec.approval_id, DELETE)


def test_record_fingerprint_cannot_be_rebound(reg, approver):
    """F2: a holder of a pending record cannot swap its fingerprint to an evil action."""
    import dataclasses as _dc

    from local_ai_agent.modules.safety.approval import _fingerprint

    rec = reg.request(DELETE, _confirm(), "agent")
    evil = Action("shell", {"cmd": "rm -rf /"})
    with pytest.raises(_dc.FrozenInstanceError):
        rec.fingerprint = _fingerprint(evil)
    # the stored binding still rejects the evil action
    with pytest.raises(ActionMismatch):
        reg.approve(rec.approval_id, evil, approver)


def test_non_dict_args_refused_at_request(reg):
    """F3: args must be a dict — non-dict args are refused (defense-in-depth)."""
    with pytest.raises(NotApprovable):
        reg.request(Action("write_file", "not-a-dict"), _confirm(), "agent")
