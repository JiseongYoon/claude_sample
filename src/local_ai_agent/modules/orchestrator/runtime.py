"""AgentRuntime — the composed agent bundle.

Holds the shared, long-lived safety + model components (gate · approvals · dispatcher ·
model · audit) plus the loop bounds. `build_application` constructs one when `enable_agent`
and attaches it to the `Application`; the gateway/session reads it and builds a
per-task `Orchestrator` via `build_orchestrator(obtain_approval)`, injecting the
connection-specific approval seam while reusing the shared components.

Scope names live here so the gateway and session agree on a single source.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..safety.approval import PendingApprovals
from ..safety.audit import AuditSink, NullAuditSink
from ..safety.dispatcher import ToolDispatcher
from ..safety.gate import SafetyGate
from .loop import ApprovalSeam, EmitSeam, Orchestrator, ToolCallingModel

# scope names — the single source the gateway + session enforce
SCOPE_AGENT_RUN = "agent:run"
SCOPE_APPROVE = "approve"


@dataclass
class AgentRuntime:
    """Shared agent components + bounds. One per process (built in build_application)."""

    gate: SafetyGate
    approvals: PendingApprovals
    dispatcher: ToolDispatcher
    model: ToolCallingModel
    audit: AuditSink = NullAuditSink()
    max_steps: int = 12
    max_tool_calls: int = 32
    result_char_cap: int = 4096
    deadline_seconds: float | None = None
    decision_timeout: float = 120.0
    # multi-turn history bounds (config-driven; applied to the seeded prior turns)
    max_history_messages: int = 20
    history_char_cap: int = 4096
    # the file-ingestion store (id → contained relpath). When a `run_task` carries
    # `attachments`, the session resolves each id to its `docs_root`-relative path and tells the
    # agent to read it via the EXISTING gated DocQA tools (INV-1 unchanged). None → no ingestion.
    ingest_store: object | None = None

    def build_orchestrator(
        self,
        obtain_approval: ApprovalSeam | None = None,
        *,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
        emit: EmitSeam | None = None,
    ) -> Orchestrator:
        """Build a per-task Orchestrator reusing the shared model/dispatcher/audit and the
        configured bounds, with this task's approval seam injected. `history` is
        the client-supplied prior conversation (already shape-validated by the session); the loop
        seeds it, bounded by the configured history caps. `emit` is the per-task
        streaming-observation seam (tool_call/tool_result events) — additive, never alters control."""
        return Orchestrator(
            model=self.model,
            dispatcher=self.dispatcher,
            obtain_approval=obtain_approval,
            audit=self.audit,
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            result_char_cap=self.result_char_cap,
            deadline_seconds=self.deadline_seconds,
            system_prompt=system_prompt,
            tools=tools,
            history=history,
            max_history_messages=self.max_history_messages,
            history_char_cap=self.history_char_cap,
            emit=emit,
        )

    async def run_ws_session(self, channel, principal, first_message) -> None:
        """Drive one interactive agent task over a `Channel`. The gateway calls this on the
        opaque `app.agent_runtime`, so `core` never imports the session (preserves the
        no-core→modules-import rule). Deferred import also avoids a runtime↔session cycle."""
        from .session import AgentSession

        await AgentSession(channel, principal, self).run(first_message)
