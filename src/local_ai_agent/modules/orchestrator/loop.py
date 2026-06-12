"""Bounded ReAct orchestrator loop.

The agent loop: the model *proposes* actions (via tool-calling); the loop dispatches
each through the gated `ToolDispatcher`, feeds the observation back, and loops
— until the model produces a final answer or a bound is hit. It is the prompt-injection
target, so it never classifies, approves, or runs a tool itself: it forwards proposals
to the dispatcher and human decisions through an injected approval seam (the
deterministic gate/dispatcher dispose — INV-1/2/4 stay where they were proven).

This module owns **INV-5 (bounded loop)**:
- terminates under `max_steps` (model turns) AND `max_tool_calls` (cumulative) AND an
  optional wall-clock `deadline`;
- every observation fed back to the model is size-capped;
- a `refused`/`aborted`/`error` dispatch outcome is an observation the loop feeds back
  and continues — never a crash; a model fault ends the run with `error`, not an
  exception out of `run`.

Hermetic by construction: the model is a `ToolCallingModel` Protocol (tests script it),
tools live only behind the dispatcher, and time comes from an injected `now`. The real
`llm_serving` adapter, the WS approval round-trip, and audit logging (INV-6) are .
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Protocol, runtime_checkable

from ..safety.audit import AuditSink, NullAuditSink, safe_record
from ..safety.dispatcher import DispatchResult, Outcome, ToolDispatcher
from ..safety.gate import Action, Decision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """One action the model proposed in a turn. `id` correlates the observation back."""

    id: str
    tool: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantTurn:
    """A model turn: a final answer (`text`, no tool_calls) or a batch of `tool_calls`."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class ToolCallingModel(Protocol):
    """The loop's view of the LLM. Wire-format-agnostic — the adapter wraps
    `llm_serving.chat` + OpenAI tool-call parsing into this. `tools` are opaque JSON
    schemas the loop passes straight through."""

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


@runtime_checkable
class StreamingToolCallingModel(Protocol):
    """A model that can stream text deltas. `complete_stream` forwards each text
    delta to `on_token` AND returns the complete `AssistantTurn` (accumulate-then-parse) so the loop
    still dispatches a full, well-formed turn. The loop uses this ONLY when a real emit seam is wired
    and the model implements it — otherwise it falls back to `complete` (hermetic/non-streaming)."""

    async def complete_stream(self, messages: list[dict], tools: list[dict],
                              on_token: Callable[[str], Awaitable[None]]) -> AssistantTurn: ...


# the human-decision seam: resolves once a decision has been recorded out-of-band
# (the gateway calls approvals.approve/deny with the principal, ) or on timeout.
ApprovalSeam = Callable[[Action, Decision, str], Awaitable[None]]

# the streaming-observation seam: the loop fires `tool_call`/`tool_result` events
# through it so a live client can show tool activity. It is ADDITIVE OBSERVATION ONLY — it never
# feeds back into the loop's control flow, so INV-1/4/6 are unaffected. Default = no-op.
EmitSeam = Callable[[dict], Awaitable[None]]


async def _no_emit(event: dict) -> None:
    """Default emit seam: drop the event (non-streaming runs / tests are unchanged)."""
    return None


class RunStatus(str, Enum):
    completed = "completed" # the model produced a final answer
    max_steps = "max_steps" # hit the turn bound
    timed_out = "timed_out" # hit the wall-clock deadline
    budget_exhausted = "budget_exhausted" # hit the cumulative tool-call bound
    error = "error" # the model call failed unrecoverably


@dataclass(frozen=True)
class RunResult:
    status: RunStatus
    answer: str | None = None
    steps: int = 0
    tool_calls_made: int = 0
    reason: str = ""
    transcript: list[dict] = field(default_factory=list)


async def _never_approve(action: Action, decision: Decision, approval_id: str) -> None:
    """Default approval seam: do nothing → the approval is never granted, so any
    needs_confirmation action aborts (fail-closed). A real seam is injected at ."""
    return None


class Orchestrator:
    """The bounded agent loop. Holds the dispatcher (never a tool) so INV-1 stands."""

    def __init__(
        self,
        *,
        model: ToolCallingModel,
        dispatcher: ToolDispatcher,
        obtain_approval: ApprovalSeam | None = None,
        max_steps: int = 12,
        max_tool_calls: int = 32,
        result_char_cap: int = 4096,
        deadline_seconds: float | None = None,
        now: Callable[[], float] = time.monotonic,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        audit: AuditSink | None = None,
        history: list[dict] | None = None,
        max_history_messages: int = 20,
        history_char_cap: int = 4096,
        emit: EmitSeam | None = None,
    ) -> None:
        if max_steps < 0 or max_tool_calls < 0 or result_char_cap < 0:
            raise ValueError("bounds must be non-negative")
        if max_history_messages <= 0 or history_char_cap <= 0:
            raise ValueError("history caps must be > 0")
        if deadline_seconds is not None and deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be > 0 when set")
        self._model = model
        self._dispatcher = dispatcher
        self._obtain_approval = obtain_approval or _never_approve
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._cap = result_char_cap
        self._deadline = deadline_seconds
        self._now = now
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._audit = audit or NullAuditSink()
        self._history = history or []
        self._max_history_messages = max_history_messages
        self._history_cap = history_char_cap
        self._emit = emit or _no_emit

    # -- multi-turn history ---------------------------------- #
    def _bounded_history(self) -> list[dict]:
        """The prior-conversation turns to seed (between the system prompt and the new user turn),
        bounded so multi-turn cannot defeat INV-5's bounded-work intent or bloat the prompt: keep
        only the most-recent `max_history_messages`, cap each content to `history_char_cap`. The
        session already shape-validates (user/assistant text only); this re-checks defensively (a
        non-{user|assistant}-text item is dropped) and never raises."""
        recent = self._history[-self._max_history_messages:]
        out: list[dict] = []
        for m in recent:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            out.append({"role": role, "content": content[: self._history_cap]})
        return out

    # -- observation rendering (size-capped, leak-free) ---------------------- #
    def _cap_str(self, s: str) -> str:
        if len(s) <= self._cap:
            return s
        return s[: self._cap] + "…[truncated]"

    def _observe(self, res: DispatchResult) -> str:
        if res.outcome is Outcome.executed:
            return self._cap_str(str(res.result))
        # refused/aborted/error/pending → our own safe strings (no tool-internal leak)
        detail = res.error or res.reason or ""
        return self._cap_str(f"[{res.outcome.value}] {detail}".rstrip())

    # -- streaming observation -- #
    async def _emit_event(self, event: dict) -> None:
        """Fire a streaming observation event best-effort + isolated: a faulty/dead/slow emit seam
        must NEVER crash or alter the loop (mirrors the approval-seam guard at `_dispatch_with_approval`).
        emit is a read-only side-channel — it never feeds back into `messages`/`tool_calls_made`/the
        dispatcher, so INV-1/4/6 stay byte-for-byte where they were proven."""
        try:
            await self._emit(event)
        except Exception: # noqa: BLE001 — a faulty emit seam must not crash the loop
            logger.warning("emit seam raised; ignoring", exc_info=True)

    async def _emit_token(self, delta: str) -> None:
        """Forward one streamed text delta as a `token` event. Goes through `_emit_event` so a
        faulty/dead channel is swallowed — token streaming never crashes the run."""
        await self._emit_event({"event": "token", "delta": delta})

    async def _model_turn(self, messages: list[dict]) -> AssistantTurn:
        """One model turn. Streams token deltas ONLY when a real emit seam is wired AND the
        model supports streaming; otherwise the unchanged buffered `complete` (hermetic/non-stream).
        Either way the loop receives a COMPLETE `AssistantTurn` to dispatch (INV-5/tool-calling
        unaffected)."""
        if self._emit is not _no_emit and isinstance(self._model, StreamingToolCallingModel):
            return await self._model.complete_stream(messages, self._tools, self._emit_token)
        return await self._model.complete(messages, self._tools)

    # -- gated dispatch + approval pause ------------------------------------- #
    async def _dispatch_with_approval(self, action: Action, requested_by: str) -> DispatchResult:
        res = await self._dispatcher.dispatch(action, requested_by)
        if res.outcome is not Outcome.pending:
            return res
        # needs_confirmation: wait for a human decision (out-of-band), then attempt to run.
        try:
            await self._obtain_approval(action, res.decision, res.approval_id or "")
        except Exception: # noqa: BLE001 — a faulty seam must not crash the loop
            logger.warning("approval seam raised; treating as no decision", exc_info=True)
        # execute_approved re-checks (re-classify + consume): runs iff approved, else aborts.
        return await self._dispatcher.execute_approved(res.approval_id or "", action)

    # -- the loop ------------------------------------------------------------ #
    async def run(self, task: str, requested_by: str) -> RunResult:
        """Audit-bracketed entry point: records task start/end around the loop (INV-6)."""
        safe_record(self._audit, {"kind": "task", "phase": "start", "principal": requested_by})
        result = await self._loop(task, requested_by)
        safe_record(self._audit, {"kind": "task", "phase": "end", "principal": requested_by,
                                  "status": result.status.value, "steps": result.steps,
                                  "tool_calls_made": result.tool_calls_made})
        return result

    async def _loop(self, task: str, requested_by: str) -> RunResult:
        messages: list[dict] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._bounded_history()) # prior turns, bounded
        messages.append({"role": "user", "content": task})

        start = self._now()
        tool_calls_made = 0
        steps = 0

        while steps < self._max_steps:
            # budget check at the top of each turn
            if self._deadline is not None and self._now() - start > self._deadline:
                return RunResult(RunStatus.timed_out, None, steps, tool_calls_made,
                                 "wall-clock deadline exceeded", messages)

            try:
                turn = await self._model_turn(messages)
            except Exception as exc: # noqa: BLE001 — model fault ends the run gracefully
                logger.warning("model.complete raised", exc_info=True)
                return RunResult(RunStatus.error, None, steps, tool_calls_made,
                                 f"model error: {type(exc).__name__}", messages)
            steps += 1

            if not turn.tool_calls:
                return RunResult(RunStatus.completed, turn.text, steps, tool_calls_made,
                                 "model produced a final answer", messages)

            messages.append({
                "role": "assistant",
                "content": turn.text or "",
                "tool_calls": [{"id": c.id, "tool": c.tool, "args": c.args}
                               for c in turn.tool_calls],
            })

            for call in turn.tool_calls:
                if tool_calls_made >= self._max_tool_calls:
                    return RunResult(RunStatus.budget_exhausted, None, steps, tool_calls_made,
                                     "max_tool_calls exceeded", messages)
                action = Action(call.tool, call.args)
                # announce the proposed call BEFORE dispatch (the gate/approval still run
                # exactly as before; for a needs_confirmation tool the approval_request follows).
                await self._emit_event({"event": "tool_call", "id": call.id,
                                        "tool": action.tool, "args": action.args})
                res = await self._dispatch_with_approval(action, requested_by)
                tool_calls_made += 1
                observation = self._observe(res)
                safe_record(self._audit, {
                    "kind": "dispatch", "principal": requested_by, "tool": action.tool,
                    "verdict": res.decision.verdict.value, "outcome": res.outcome.value,
                    "approval_id": res.approval_id,
                })
                # report the outcome AFTER the INV-6 audit record (a stalled emit can never
                # delay or drop the audit). `observation` is the same capped, leak-free string the
                # model sees — no tool-internal path/secret leaks beyond what the gate already allows.
                await self._emit_event({"event": "tool_result", "id": call.id,
                                        "tool": action.tool, "outcome": res.outcome.value,
                                        "result": observation})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": observation,
                })

        return RunResult(RunStatus.max_steps, None, steps, tool_calls_made,
                         "max_steps exhausted", messages)
