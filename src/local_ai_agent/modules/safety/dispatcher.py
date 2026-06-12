"""Tool dispatcher — the single gated chokepoint.

Every action a tool would perform flows through here and nowhere else (INV-1). The
dispatcher classifies a proposed `Action` with the deterministic `SafetyGate`
and routes it:

- `safe` → run the tool immediately.
- `needs_confirmation` → enrol a pending approval and return `pending`; the
                         gateway later approves (with the `approve` scope) and calls
                         `execute_approved`, which consumes the single-use, action-bound
                         grant and runs the tool.
- `blocked` → refuse; the tool is never invoked — not even with a valid
                         approval (both entry points re-classify and refuse `blocked`).

Tools are reachable ONLY through this dispatcher's registry; the orchestrator
is wired with the dispatcher alone and never holds a tool, so there is no structural
bypass. The dispatcher **never raises** for a policy or tool condition — it returns a
`DispatchResult` the loop turns into an observation (fail-closed, INV-3; loop-safe,
INV-5). The WS approve/deny routes, scopes, and audit are .
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol, runtime_checkable

from .approval import ApprovalError, PendingApprovals
from .gate import Action, Decision, SafetyGate, Verdict

logger = logging.getLogger(__name__)


@runtime_checkable
class Tool(Protocol):
    """A capability the agent can invoke. Real tools (phases 6–8) do I/O, so `run` is
    async. The dispatcher passes the action's `args` straight through."""

    name: str

    async def run(self, args: dict) -> Any: ...


class Outcome(str, Enum):
    executed = "executed" # ran; `result` holds the tool output
    refused = "refused" # blocked → never run
    pending = "pending" # needs_confirmation → `approval_id` set, awaiting human
    aborted = "aborted" # denied / expired / mismatch / unapproved → never run
    error = "error" # unknown tool, or the tool raised → caught, fed back


@dataclass(frozen=True)
class DispatchResult:
    """The single return type of every dispatch path. Immutable; carries enough for the
    loop to build an observation and for the audit layer to log."""

    outcome: Outcome
    action: Action
    decision: Decision
    result: Any = None
    approval_id: str | None = None
    reason: str = ""
    error: str | None = None


class ToolDispatcher:
    """The one gated path to tool execution (INV-1).

    Construct with the deterministic gate, the approvals registry, and the
    tools. `register` is composition-time wiring — it adds capability, never permission:
    a newly registered tool is still classified by the same immutable gate rules."""

    def __init__(
        self,
        *,
        gate: SafetyGate,
        approvals: PendingApprovals,
        tools: Iterable[Tool] = (),
    ) -> None:
        self._gate = gate
        self._approvals = approvals
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry (composition-time). Does not loosen any gate rule."""
        self._tools[tool.name] = tool

    @property
    def approvals(self) -> PendingApprovals:
        """The approvals registry — the gateway calls `approve`/`deny` on it with
        the authenticated principal. Exposed read-only-ish (no setter for rules)."""
        return self._approvals

    def roster(self) -> list[dict]:
        """Read-only: each registered tool's name + its BASELINE safety tier.
        Classified with empty args, so it reflects the tool's name-derived tier (safe vs
        needs_confirmation); a real call always re-classifies with the actual args (a `safe`
        tool can still escalate to confirm/blocked on hostile args). No execution, no mutation."""
        out: list[dict] = []
        for name in sorted(self._tools):
            decision = self._gate.classify(Action(tool=name, args={}))
            out.append({"name": name, "tier": decision.verdict.value})
        return out

    def tool_schemas(self) -> list[dict]:
        """OpenAI function-tool schemas for every registered tool. The agent
        loop hands these to the model so it knows which tools exist and how to call them — without
        this the model is given an EMPTY tool list and can never propose a tool call. Each tool may
        expose `description` (str) + a JSON-Schema `parameters` (dict); MCP tools expose
        `input_schema`; missing metadata falls back to an empty-object parameter schema (the tool is
        still offered by name). Capability surfacing ONLY — the gate, not this metadata, is the
        security control (a `safe` tool still re-classifies on the real args at dispatch)."""
        out: list[dict] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            fn: dict = {"name": name}
            desc = getattr(tool, "description", None)
            if isinstance(desc, str) and desc:
                fn["description"] = desc
            params = getattr(tool, "parameters", None)
            if not isinstance(params, dict):
                params = getattr(tool, "input_schema", None) # MCP tools (untrusted, policy-bounded)
            fn["parameters"] = params if isinstance(params, dict) else {"type": "object", "properties": {}}
            out.append({"type": "function", "function": fn})
        return out

    # -- internal: the ONLY place tool.run is invoked ------------------------ #
    async def _run_tool(self, action: Action, decision: Decision) -> DispatchResult:
        tool = self._tools.get(action.tool)
        if tool is None: # cannot execute an unregistered tool — fail-closed
            return DispatchResult(Outcome.error, action, decision,
                                  error=f"unknown tool: {action.tool!r}",
                                  reason="no such registered tool")
        try:
            result = await tool.run(action.args)
        except Exception as exc: # noqa: BLE001 — a tool fault must not crash the loop
            # full detail (incl. message/traceback) goes to the operator log only; the
            # returned object carries just the exception TYPE so no tool-internal string
            # (which could embed a path/secret) leaks into the loop/audit/transport.
            logger.warning("tool %r raised during dispatch", action.tool, exc_info=True)
            return DispatchResult(Outcome.error, action, decision,
                                  error=f"tool raised {type(exc).__name__}",
                                  reason="tool execution failed")
        return DispatchResult(Outcome.executed, action, decision, result=result)

    # -- entry point 1: classify + route ------------------------------------- #
    async def dispatch(self, action: Action, requested_by: str) -> DispatchResult:
        """Classify and route. Never blocks waiting for a human; `needs_confirmation`
        returns `pending` with an `approval_id` the gateway resolves out-of-band."""
        decision = self._gate.classify(action)

        if decision.verdict is Verdict.blocked:
            return DispatchResult(Outcome.refused, action, decision, reason=decision.reason)

        # block precedence is settled above; an unregistered tool can never run.
        if action.tool not in self._tools:
            return DispatchResult(Outcome.error, action, decision,
                                  error=f"unknown tool: {action.tool!r}",
                                  reason="no such registered tool")

        if decision.verdict is Verdict.safe:
            return await self._run_tool(action, decision)

        # needs_confirmation → enrol a pending approval. Does not run.
        try:
            record = self._approvals.request(action, decision, requested_by)
        except ApprovalError as exc:
            return DispatchResult(Outcome.aborted, action, decision,
                                  reason=f"could not enrol approval: {exc}")
        return DispatchResult(Outcome.pending, action, decision,
                              approval_id=record.approval_id, reason=decision.reason)

    # -- entry point 2: run a human-approved action -------------------------- #
    async def execute_approved(self, approval_id: str, action: Action) -> DispatchResult:
        """Run an action whose approval was granted (by the gateway via
        `approvals.approve`). All checks are synchronous up to the single `await run`, so
        the prefix is atomic under asyncio (concurrent calls → one run, the rest aborted).
        A re-classification to `blocked` refuses even a validly-approved action."""
        decision = self._gate.classify(action)

        # blocked is never runnable — not even with a valid approval.
        if decision.verdict is Verdict.blocked:
            return DispatchResult(Outcome.refused, action, decision, reason=decision.reason)

        # unknown tool → error WITHOUT consuming the approval (so it isn't wasted).
        if action.tool not in self._tools:
            return DispatchResult(Outcome.error, action, decision,
                                  error=f"unknown tool: {action.tool!r}",
                                  reason="no such registered tool")

        # single-use + action-binding + state gate. No await before this.
        try:
            self._approvals.consume(approval_id, action)
        except ApprovalError as exc:
            return DispatchResult(Outcome.aborted, action, decision,
                                  reason=f"{type(exc).__name__}: {exc}")

        return await self._run_tool(action, decision)
