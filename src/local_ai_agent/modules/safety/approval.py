"""HITL approval protocol — the pending-approval state machine.

A pure, in-memory state machine that sits between `SafetyGate.classify` and
the gated dispatcher. When a proposed action is `needs_confirmation`, it is
*enrolled* as a pending approval; a human carrying the `approve` scope *approves* or
*denies* it; the dispatcher then *consumes* the approval exactly once to run the action.

This module owns **INV-4 (approval integrity)**:
- **single-use** — `consume` is a one-shot `approved → consumed` transition; replays fail.
- **bound to the exact action** — a sha256 fingerprint of the literal `(tool, args)` is
  checked at *both* `approve` and `consume`, so an approval for action A can never
  authorize action B even with a valid id.
- **authorized** — `approve` requires an approver with the `approve` scope.
- **timeout → deny (fail-closed)** — a pending approval expires; expiry is evaluated
  lazily on every access with a `now() >= expires_at` boundary (favors expiry).
- **`blocked` is never approvable** — only `needs_confirmation` actions can be enrolled
  (re-checked at `approve` for defense-in-depth); `blocked`/`safe` are refused.

No transport here — the WS request/approve/deny routes and the audit timestamps are
. No I/O, no LLM.

Concurrency: every method is **synchronous and `await`-free**, so under the
single-threaded asyncio event loop no two calls can interleave mid-method — the registry
is inherently atomic and needs no lock. (If any method ever gains an `await`, it must
take a lock first — cf. the model-manager's single-flight lock, which exists precisely
because it awaits subprocess I/O.)
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

from .gate import Action, Decision, Verdict

_APPROVE_SCOPE = "approve"


class ApprovalState(str, Enum):
    pending = "pending"
    approved = "approved"
    consumed = "consumed"
    denied = "denied"
    expired = "expired"


# --------------------------------------------------------------------------- #
# errors — every one means "do not run" (fail-closed). The dispatcher/gateway
# maps these to refusals; tests assert the precise failure mode.
# --------------------------------------------------------------------------- #
class ApprovalError(Exception):
    """Base: any failure to obtain a valid, single-use, action-bound grant."""


class ApprovalNotFound(ApprovalError):
    """No approval with that id (unknown or forged)."""


class ApprovalStateError(ApprovalError):
    """Wrong state for the requested transition (replay, expired, denied,
    double-approve, consume-before-approve)."""


class ActionMismatch(ApprovalError):
    """The presented action does not match the action the approval was bound to."""


class NotApprovable(ApprovalError):
    """The verdict is not `needs_confirmation` (`blocked`/`safe` cannot be enrolled),
    or the action cannot be fingerprinted."""


class NotAuthorized(ApprovalError):
    """The approver lacks the required scope."""


@runtime_checkable
class Approver(Protocol):
    """Minimal authorization seam. `core.auth.Principal` satisfies it structurally —
    this module does not import `core.auth` (same decoupling discipline as gate.py)."""

    subject: str

    def has_scope(self, scope: str) -> bool: ...


@dataclass(frozen=True)
class ApprovalRecord:
    """One pending/decided approval. **Immutable** — the binding (`fingerprint`,
    `verdict`, …) is fixed at request time and `status` advances only by the registry
    replacing the stored record with a new frozen copy (`dataclasses.replace`). Because
    it is frozen, a caller holding a returned record cannot tamper with the stored
    binding or replay-enable a consumed grant (closes the mutable-reference escape)."""

    approval_id: str
    fingerprint: str
    verdict: Verdict
    reason: str
    requested_by: str
    created_at: float
    expires_at: float
    status: ApprovalState = ApprovalState.pending
    decided_by: str | None = None
    decided_at: float | None = None


def _fingerprint(action: Action) -> str:
    """Stable identity of the literal action (what will actually run). Binds tool+args
    exactly; raises NotApprovable if the action cannot be canonically serialized."""
    try:
        canon = json.dumps(
            {"tool": action.tool, "args": action.args},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise NotApprovable(f"action is not fingerprintable: {exc}") from exc
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class PendingApprovals:
    """In-memory registry + state machine for HITL approvals (INV-4).

    Construct with a timeout (validated > 0) and, for deterministic tests, injectable
    `now` (default `time.monotonic` — monotonic so expiry is immune to wall-clock jumps)
    and `id_factory` (default `uuid4().hex` — 122 bits, unguessable). The approve-scope
    name is fixed by construction; there is no setter to loosen any rule at runtime."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 300.0,
        now: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] | None = None,
        approve_scope: str = _APPROVE_SCOPE,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._timeout = float(timeout_seconds)
        self._now = now
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._approve_scope = approve_scope
        self._records: dict[str, ApprovalRecord] = {}

    # -- internal helpers ---------------------------------------------------- #
    def _store(self, record: ApprovalRecord, **changes) -> ApprovalRecord:
        """Advance a frozen record by storing a replaced copy; return the new record."""
        updated = dataclasses.replace(record, **changes)
        self._records[updated.approval_id] = updated
        return updated

    def _get_live(self, approval_id: str) -> ApprovalRecord:
        """Fetch a record, applying lazy expiry. Raises ApprovalNotFound if unknown."""
        record = self._records.get(approval_id)
        if record is None:
            raise ApprovalNotFound(approval_id)
        if record.status is ApprovalState.pending and self._now() >= record.expires_at:
            record = self._store(record, status=ApprovalState.expired, decided_at=self._now())
        return record

    # -- lifecycle ----------------------------------------------------------- #
    def request(self, action: Action, decision: Decision, requested_by: str) -> ApprovalRecord:
        """Enrol a `needs_confirmation` action as a pending approval.

        Refuses any other verdict (`blocked`/`safe`) — so a blocked action can never
        enter the registry — and any un-fingerprintable action."""
        if decision.verdict is not Verdict.needs_confirmation:
            raise NotApprovable(
                f"only needs_confirmation is approvable, got {decision.verdict.value!r}"
            )
        if not isinstance(action.args, dict): # defense-in-depth: gate also blocks this
            raise NotApprovable("action.args must be a dict")
        fingerprint = _fingerprint(action)
        now = self._now()
        approval_id = self._id_factory()
        record = ApprovalRecord(
            approval_id=approval_id,
            fingerprint=fingerprint,
            verdict=decision.verdict,
            reason=decision.reason,
            requested_by=requested_by,
            created_at=now,
            expires_at=now + self._timeout,
        )
        self._records[approval_id] = record
        return record

    def approve(self, approval_id: str, action: Action, approver: Approver) -> ApprovalRecord:
        """pending → approved. Guarded (in order): exists → not expired → is pending →
        authorized → still needs_confirmation → action matches."""
        record = self._get_live(approval_id)
        if record.status is not ApprovalState.pending:
            raise ApprovalStateError(
                f"cannot approve from state {record.status.value!r}"
            )
        if not approver.has_scope(self._approve_scope):
            raise NotAuthorized(f"approver lacks {self._approve_scope!r} scope")
        if record.verdict is not Verdict.needs_confirmation: # defense-in-depth
            raise NotApprovable("stored verdict is not needs_confirmation")
        if _fingerprint(action) != record.fingerprint:
            raise ActionMismatch("presented action does not match the approval")
        return self._store(
            record,
            status=ApprovalState.approved,
            decided_by=getattr(approver, "subject", None),
            decided_at=self._now(),
        )

    def deny(self, approval_id: str, approver: Approver | None = None) -> ApprovalRecord:
        """pending|approved → denied (revoke-before-run allowed). Always the fail-safe
        direction, so no scope is required; records the decider when supplied."""
        record = self._get_live(approval_id)
        if record.status not in (ApprovalState.pending, ApprovalState.approved):
            raise ApprovalStateError(f"cannot deny from state {record.status.value!r}")
        return self._store(
            record,
            status=ApprovalState.denied,
            decided_by=getattr(approver, "subject", None),
            decided_at=self._now(),
        )

    def consume(self, approval_id: str, action: Action) -> ApprovalRecord:
        """The one-shot execution gate, called by the dispatcher (system, not a user).
        approved → consumed, exactly once. A second consume (replay) fails; pending,
        denied, and expired all fail-closed."""
        record = self._get_live(approval_id)
        if record.status is not ApprovalState.approved:
            raise ApprovalStateError(
                f"cannot consume from state {record.status.value!r}"
            )
        if _fingerprint(action) != record.fingerprint:
            raise ActionMismatch("presented action does not match the approval")
        return self._store(record, status=ApprovalState.consumed)

    # -- reads / housekeeping ------------------------------------------------ #
    def get(self, approval_id: str) -> ApprovalRecord | None:
        """Read a record (applies lazy expiry). None if unknown."""
        try:
            return self._get_live(approval_id)
        except ApprovalNotFound:
            return None

    def pending(self) -> list[ApprovalRecord]:
        """Live pending approvals (after expiring stale ones)."""
        self.sweep()
        return [r for r in self._records.values() if r.status is ApprovalState.pending]

    def purge_terminal(self) -> int:
        """Drop terminal records (consumed/denied/expired), keeping pending/approved.
        Returns the count removed. Bounds registry growth across many tasks; never
        touches a live (pending/approved) approval, so it cannot affect a grant."""
        terminal = {ApprovalState.consumed, ApprovalState.denied, ApprovalState.expired}
        stale = [aid for aid, r in self._records.items() if r.status in terminal]
        for aid in stale:
            del self._records[aid]
        return len(stale)

    def sweep(self) -> int:
        """Expire all stale pendings; return how many were expired (housekeeping)."""
        now = self._now()
        stale = [
            r.approval_id for r in self._records.values()
            if r.status is ApprovalState.pending and now >= r.expires_at
        ]
        for approval_id in stale:
            self._store(self._records[approval_id], status=ApprovalState.expired, decided_at=now)
        return len(stale)
