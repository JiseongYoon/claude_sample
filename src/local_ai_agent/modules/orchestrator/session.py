"""AgentSession — the interactive WS approval round-trip.

Runs one agent task over a bidirectional `Channel` (the FastAPI `WebSocket` satisfies it
structurally). The orchestrator runs as a task; a concurrent reader loop consumes the
client's approve/deny messages and resolves the orchestrator's `obtain_approval` seam.

Security + robustness (from the pre-build codebase cross-check):
- **server-side action binding** — the wire carries only `approval_id` + decision; the
  action approved/consumed is the one the orchestrator proposed (stored here), so a client
  cannot substitute or forge the action behind an approval.
- **scope** — `run_task` needs `agent:run`; an approve is authorized by `approvals.approve`
  (`approve` scope, via the `Principal`).
- **transport-agnostic** — the session imports no `starlette`; any `receive_json` exception
  is treated as connection-closed (B3). The gateway hands off by calling
  `AgentRuntime.run_ws_session` so it never imports this module (B1).
- **clean lifecycle** — orchestrator-task ⟷ reader-task race with cancellation in both
  directions; terminal approvals purged in `finally` (B4). A decision timeout **denies** the
  approval so a late approve can't leave an orphan `approved` record (B5).
- sends are serialized by a lock; the session never crashes on a junk message or a dead channel.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from ..safety.approval import ApprovalError, ApprovalState
from ..safety.audit import safe_record
from ..safety.gate import Action, Decision
from .runtime import SCOPE_AGENT_RUN

logger = logging.getLogger(__name__)

_MAX_ATTACHMENTS = 16 # bound the attachment list (DoS / prompt-bloat)


def _attachment_note(items: list[tuple[str, str]]) -> str:
    """An instruction telling the agent which uploaded docs are available and how to read them via the
    EXISTING gated tools (INV-1 unchanged). Each item is `(id, relpath)`. A TEXT document is read with
    the DocQA tools using its PATH; an IMAGE or scanned PDF is read with `answer_about_image` using its
    ID (that vision tool exists only when the model is multimodal — if it's absent, the path tools apply)."""
    lines = "\n".join(f"- id={aid} path={rel}" for aid, rel in items)
    return ("The user attached the following document(s):\n" + lines + "\n"
            "To use one: for a TEXT document call a DocQA tool (e.g. summarize_document / answer_question) "
            "with its path; for an IMAGE or scanned PDF call answer_about_image with its id.")


@runtime_checkable
class Channel(Protocol):
    async def send_json(self, data: dict) -> None: ...
    async def receive_json(self) -> Any: ...


class AgentSession:
    """One interactive agent run over a `Channel`. Construct per connection/task."""

    def __init__(self, channel: Channel, principal, runtime) -> None:
        self._ch = channel
        self._principal = principal
        self._rt = runtime
        self._pending: dict[str, asyncio.Future] = {}
        self._actions: dict[str, Action] = {} # server-side action binding (never client-supplied)
        self._send_lock = asyncio.Lock()

    # -- transport helpers --------------------------------------------------- #
    async def _safe_send(self, data: dict) -> None:
        """Serialized, failure-tolerant send (channel may close mid-send)."""
        async with self._send_lock:
            try:
                await self._ch.send_json(data)
            except Exception: # noqa: BLE001 — a dead channel must not crash the session
                logger.debug("send failed (channel likely closed)", exc_info=True)

    def _audit(self, event: dict) -> None:
        safe_record(self._rt.audit, {**event, "principal": self._principal.subject})

    # -- the streaming-observation seam -- #
    async def emit_event(self, event: dict) -> None:
        """Relay a loop streaming event (tool_call/tool_result) to the client. Uses the same
        serialized, failure-tolerant `_safe_send` as approval prompts — a dead channel is swallowed,
        never crashing the run. Write-only side-channel: it carries no authority (the wire UI can
        still only relay approve/deny by id)."""
        await self._safe_send(event)

    def _deny_if_pending(self, approval_id: str) -> None:
        """Deny an approval that is still `pending` (never granted) — used in cleanup so a
        timeout/cancel doesn't leave it lingering until expiry. A no-op for any non-pending
        (approved/consumed/denied/expired/absent) record. Sync + await-free → safe in a
        `finally` even during task cancellation."""
        rec = self._rt.approvals.get(approval_id)
        if rec is not None and rec.status is ApprovalState.pending:
            try:
                self._rt.approvals.deny(approval_id, self._principal)
            except ApprovalError:
                pass

    # -- the approval seam (runs inside the orchestrator task) --------------- #
    async def obtain_approval(self, action: Action, decision: Decision, approval_id: str) -> None:
        self._actions[approval_id] = action # bind server-side
        await self._safe_send({"event": "approval_request", "approval_id": approval_id,
                               "tool": action.tool, "args": action.args, "reason": decision.reason})
        fut = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = fut
        try:
            await asyncio.wait_for(asyncio.shield(fut), self._rt.decision_timeout)
        except asyncio.TimeoutError:
            self._audit({"kind": "approval", "approval_id": approval_id, "decision": "timeout"})
            await self._safe_send({"event": "approval_timeout", "approval_id": approval_id})
        finally:
            self._pending.pop(approval_id, None)
            self._actions.pop(approval_id, None)
            # deny-if-live: if the approval was never granted — a timeout OR a cancel
            # (disconnect cancels the orchestrator task → CancelledError here) — deny it so it
            # can't linger `pending` until expiry and a late approve can't run it. An already
            # `approved` record (happy path) is left intact for execute_approved to consume.
            self._deny_if_pending(approval_id)

    # -- decision handling (runs in the reader loop) ------------------------- #
    async def _handle_decision(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            await self._safe_send({"event": "approval_error", "reason": "malformed message"})
            return
        act = msg.get("action")
        approval_id = msg.get("approval_id")
        if act not in ("approve", "deny") or not isinstance(approval_id, str):
            await self._safe_send({"event": "approval_error",
                                   "approval_id": approval_id if isinstance(approval_id, str) else None,
                                   "reason": "unknown decision"})
            return
        action = self._actions.get(approval_id) # the SERVER-stored action, never the client's
        if action is None:
            await self._safe_send({"event": "approval_error", "approval_id": approval_id,
                                   "reason": "unknown or stale approval"})
            return
        try:
            if act == "approve":
                self._rt.approvals.approve(approval_id, action, self._principal) # scope+binding+state
            else:
                self._rt.approvals.deny(approval_id, self._principal)
        except ApprovalError as exc:
            self._audit({"kind": "approval", "approval_id": approval_id,
                         "decision": act, "error": type(exc).__name__})
            await self._safe_send({"event": "approval_error", "approval_id": approval_id,
                                   "reason": type(exc).__name__})
            return
        self._audit({"kind": "approval", "approval_id": approval_id, "decision": act})
        fut = self._pending.get(approval_id)
        if fut is not None and not fut.done():
            fut.set_result(True)

    async def _reader_loop(self) -> None:
        """Consume decision messages until the channel dies (receive_json raises)."""
        while True:
            msg = await self._ch.receive_json() # raises on disconnect → ends this task
            await self._handle_decision(msg)

    # -- entry point --------------------------------------------------------- #
    async def run(self, first_message: Any) -> None:
        """Run one task. `first_message` is the already-read `run_task` envelope."""
        if not isinstance(first_message, dict) or first_message.get("action") != "run_task":
            await self._safe_send({"event": "error", "reason": "expected run_task"})
            return
        if not self._principal.has_scope(SCOPE_AGENT_RUN):
            await self._safe_send({"event": "error", "reason": "forbidden"})
            return
        task = first_message.get("task")
        if not isinstance(task, str) or not task:
            await self._safe_send({"event": "error", "reason": "missing task"})
            return
        raw_system = first_message.get("system")
        system_prompt = raw_system if isinstance(raw_system, str) else None

        # optional `attachments` (opaque ids). The SERVER resolves each id → its
        # contained `docs_root`-relative path (the wire never carries a path → no client forgery)
        # and folds an instruction into the system prompt; the agent reads the docs through the
        # EXISTING gated DocQA tools (INV-1 unchanged). Unknown id → typed error, no run.
        raw_attachments = first_message.get("attachments")
        if raw_attachments is not None:
            if (not isinstance(raw_attachments, list) or len(raw_attachments) > _MAX_ATTACHMENTS
                    or not all(isinstance(a, str) for a in raw_attachments)):
                await self._safe_send({"event": "error", "reason": "invalid attachments"})
                return
            if raw_attachments:
                store = getattr(self._rt, "ingest_store", None)
                if store is None:
                    await self._safe_send({"event": "error", "reason": "attachments unavailable"})
                    return
                items: list[tuple[str, str]] = []
                for aid in raw_attachments:
                    try:
                        items.append((aid, store.resolve(aid))) # (id, contained relpath)
                    except Exception: # noqa: BLE001 — UnknownIngestId (duck-typed) → no leak, no run
                        await self._safe_send({"event": "error", "reason": "unknown attachment"})
                        return
                note = _attachment_note(items)
                system_prompt = f"{system_prompt}\n\n{note}" if system_prompt else note

        # optional `history` — the prior conversation the CLIENT replays (same trust
        # tier as `task`/`system`). TEXT-ONLY `{role, content}` with role ∈ {user, assistant}; we
        # REJECT system/tool/approval frames + malformed shapes (no authority injection — history
        # never carries a tool call or an approved action). Keep only the most-recent N (drop oldest
        # beyond the cap) before validating; the loop further caps each message's content length.
        history: list[dict] | None = None
        raw_history = first_message.get("history")
        if raw_history is not None:
            if not isinstance(raw_history, list):
                await self._safe_send({"event": "error", "reason": "invalid history"})
                return
            kept = raw_history[-self._rt.max_history_messages:] # drop oldest beyond the count cap
            validated: list[dict] = []
            for item in kept:
                if (not isinstance(item, dict) or item.get("role") not in ("user", "assistant")
                        or not isinstance(item.get("content"), str)):
                    await self._safe_send({"event": "error", "reason": "invalid history"})
                    return
                validated.append({"role": item["role"], "content": item["content"]})
            history = validated

        # F3a: surface the registered tools' OpenAI schemas to the model — without
        # this the orchestrator is built with an EMPTY tool list and the model can never propose a
        # tool call (the gate still governs execution; this only tells the model what exists).
        orch = self._rt.build_orchestrator(
            self.obtain_approval, system_prompt=system_prompt,
            tools=self._rt.dispatcher.tool_schemas(), history=history,
            emit=self.emit_event) # live tool-activity events
        self._audit({"kind": "task", "phase": "start", "via": "ws"})
        orch_task = asyncio.create_task(orch.run(task, self._principal.subject))
        reader_task = asyncio.create_task(self._reader_loop())
        try:
            done, _ = await asyncio.wait({orch_task, reader_task},
                                         return_when=asyncio.FIRST_COMPLETED)
            if orch_task in done:
                try:
                    result = orch_task.result()
                    await self._safe_send({"event": "task_result", "status": result.status.value,
                                           "answer": result.answer, "steps": result.steps,
                                           "tool_calls_made": result.tool_calls_made})
                except Exception: # noqa: BLE001 — run() shouldn't raise, but never crash the session
                    logger.warning("orchestrator task raised unexpectedly", exc_info=True)
                    await self._safe_send({"event": "task_result", "status": "error",
                                           "answer": None, "reason": "internal error"})
            # else: reader finished first (disconnect) → orch_task is cancelled below
        finally:
            for t in (orch_task, reader_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(orch_task, reader_task, return_exceptions=True)
            self._rt.approvals.purge_terminal()
