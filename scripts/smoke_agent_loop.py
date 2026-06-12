"""Real-model smoke for the AGENT LOOP — (the first end-to-end proof).

NOT a unit test — an OPERATOR smoke that spins up the real Gemma engine and drives the FULL
agent loop where THE MODEL PROPOSES TOOL CALLS (model → LLMToolModel → llm_serving.chat with
`tools` on the wire → tool_calls parsed → gated dispatcher → observation → final answer).

This is the path fixed and that NO prior smoke exercised: smoke_docqa.py drives tools by
directly calling `dispatcher.dispatch(Action(...))` (the model never proposes anything), and the
unit suite uses a scripted model that never hits `chat()`. So before this the agent had never been
shown its tools at all. Run on the GPU box, inside conda `local-ai-agent-env-1`, with the model:

    GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf \
    DOCS_ROOT=/abs/path/to/a/folder/with/docs \
    python scripts/smoke_agent_loop.py [--task "..."]

PASS criteria (printed): the model PROPOSED at least one tool call (`tool_calls_made >= 1`), each
proposal routed through the gated dispatcher, and the run reached a final answer (`completed`).
Loads a ~27 GB model (minutes) and uses the GPUs — run deliberately, not in CI.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

# allow `python scripts/smoke_agent_loop.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_ai_agent.config import Settings # noqa: E402
from local_ai_agent.main import build_application # noqa: E402
from local_ai_agent.modules.safety.gate import Action, Decision # noqa: E402


@dataclass
class _Approver:
    subject: str = "smoke-operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve", "agent:run"}))

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


async def _wait_serving(mm, timeout: float = 600.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mm.is_serving:
            return True
        await asyncio.sleep(2.0)
    return False


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="List the available documents, then summarize the first one.")
    args = ap.parse_args()

    docs_root = os.environ.get("DOCS_ROOT")
    if not docs_root:
        print("ERROR: set DOCS_ROOT to a folder containing real documents.", file=sys.stderr)
        return 2

    settings = Settings(enable_agent=True, enable_docqa=True, docs_root=docs_root,
                        gguf_file=os.environ.get("GGUF_FILE"))
    app = build_application(settings)
    await app.startup()
    rt = app.agent_runtime
    mm = app.get_module("model-manager")

    # auto-approve seam (operator smoke): approve whatever the loop pauses on, then it executes.
    approver = _Approver()

    async def obtain_approval(action: Action, decision: Decision, approval_id: str) -> None:
        try:
            rt.approvals.approve(approval_id, action, approver)
        except Exception as exc: # noqa: BLE001
            print(" (approve failed:", type(exc).__name__, exc, ")")

    try:
        _hr("loading model (this can take minutes)")
        await mm.load(settings.gguf_file)
        if not await _wait_serving(mm):
            print("ERROR: engine never became ready.", file=sys.stderr)
            return 3
        print(f"serving: {mm.is_serving} file={mm.loaded_file}")

        schemas = rt.dispatcher.tool_schemas()
        _hr(f"tools offered to the model ({len(schemas)})")
        for s in schemas:
            print(" -", s["function"]["name"])
        if not schemas:
            print("ERROR: no tool schemas — the model would have nothing to call.", file=sys.stderr)
            return 4

        # an emit seam that prints live tool activity + streamed tokens (operator sees the
        # real streaming path, not just the terminal answer).
        seen = {"token": 0, "tool_call": 0, "tool_result": 0}

        async def emit(event: dict) -> None:
            kind = event.get("event")
            seen[kind] = seen.get(kind, 0) + 1
            if kind == "token":
                print(event.get("delta", ""), end="", flush=True)
            elif kind == "tool_call":
                print(f"\n [tool_call] {event.get('tool')}({event.get('args')})")
            elif kind == "tool_result":
                print(f" [tool_result] {event.get('tool')} → {event.get('outcome')}")

        _hr(f"agent run — task={args.task!r}")
        orch = rt.build_orchestrator(obtain_approval, tools=schemas, emit=emit,
                                     system_prompt="You are a helpful agent. Use the available tools.")
        result = await orch.run(args.task, approver.subject)

        print("\nstatus:", result.status.value)
        print("steps:", result.steps, " tool_calls_made:", result.tool_calls_made)
        print("streamed events:", seen)
        print("answer:", (result.answer or "")[:1000])

        _hr("VERDICT")
        ok = result.tool_calls_made >= 1 and result.status.value == "completed"
        # the headline proof: the MODEL proposed a tool call over the real wire and it ran gated.
        print("PASS" if ok else "FAIL",
              "— the model proposed a tool call through the gated path and produced an answer."
              if ok else "— the model did NOT propose/complete a gated tool call (inspect above).")
        return 0 if ok else 5
    finally:
        await app.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
