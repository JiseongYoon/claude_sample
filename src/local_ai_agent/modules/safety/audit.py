"""Audit sink — the INV-6 record of safety decisions.

Every classification + execution outcome (from the orchestrator), every approval /
denial, and every task start/end (from the session) is recorded here with the principal
and a wall-clock timestamp. Audit is **observation-only**: it never alters control flow,
so a sink that raises must not break a run — callers use `safe_record`.

This is deliberately tiny: a `record(event: dict)` seam with a logging implementation, a
null default, and (in tests) a list recorder. A real deployment can swap in a sink that
writes to a file / DB / SIEM without touching the safety components.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

logger = logging.getLogger("local_ai_agent.audit")


@runtime_checkable
class AuditSink(Protocol):
    def record(self, event: dict) -> None: ...


class NullAuditSink:
    """Default no-op sink."""

    def record(self, event: dict) -> None: # noqa: D102
        return None


class LoggingAuditSink:
    """Emit one structured log line per event, stamped with wall-clock UTC time.

    Wall-clock (not the components' monotonic clock) is intentional — audit needs a
    human-readable, comparable timestamp; expiry/deadline math needs monotonic."""

    def __init__(self, level: int = logging.INFO) -> None:
        self._level = level

    def record(self, event: dict) -> None:
        stamped = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        logger.log(self._level, "audit %s", stamped)


def safe_record(sink: AuditSink, event: dict) -> None:
    """Record without letting a faulty sink break the caller (audit is side-channel)."""
    try:
        sink.record(event)
    except Exception: # noqa: BLE001 — audit must never affect control flow
        logger.warning("audit sink raised; dropping event", exc_info=True)
