"""Orchestrator — the bounded agent loop.

Drives the local model via tool-calling, dispatching every proposed action through the
gated `ToolDispatcher`. Bounded and crash-safe (INV-5)."""
from .loop import (
    AssistantTurn,
    Orchestrator,
    RunResult,
    RunStatus,
    ToolCall,
    ToolCallingModel,
)

__all__ = [
    "AssistantTurn",
    "Orchestrator",
    "RunResult",
    "RunStatus",
    "ToolCall",
    "ToolCallingModel",
]
