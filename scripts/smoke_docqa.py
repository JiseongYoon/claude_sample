"""Real-model smoke for DocQA.

NOT a unit test — an OPERATOR smoke that spins up the real Gemma engine (via the model
manager) and drives the DocQA capability through the REAL stack. Run on the GPU box, inside
conda `local-ai-agent-env-1`, with the GGUF model present:

    GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf \
    DOCS_ROOT=/abs/path/to/a/folder/with/docs \
    python scripts/smoke_docqa.py [--question "your question"]

What it checks:
  * Part A — real model reachable: `llm-serving.chat` returns a real completion (engine →
    adapter path). (Full agent-loop / WS round-trip procedure: see
    
  * Part B — real-tool HITL: `list_documents` / `summarize_document` / `answer_question` run
    through the gated dispatcher (the chokepoint) over a REAL document with the REAL
    model, including a confirm→approve→execute path. Prints outcomes + citations.

It loads a ~27 GB model (several minutes) and uses the GPUs — run deliberately, not in CI.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

# allow `python scripts/smoke_docqa.py` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_ai_agent.config import Settings # noqa: E402
from local_ai_agent.main import build_application # noqa: E402
from local_ai_agent.modules.safety.dispatcher import Outcome # noqa: E402
from local_ai_agent.modules.safety.gate import Action # noqa: E402


@dataclass
class _Approver:
    subject: str = "smoke-operator"
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

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
    ap.add_argument("--question", default="What is this document about?")
    ap.add_argument("--doc", default=None, help="root-relative doc for the single-doc summary")
    args = ap.parse_args()

    docs_root = os.environ.get("DOCS_ROOT")
    if not docs_root:
        print("ERROR: set DOCS_ROOT to a folder containing real documents.", file=sys.stderr)
        return 2

    settings = Settings(
        enable_agent=True, enable_docqa=True, docs_root=docs_root,
        gguf_file=os.environ.get("GGUF_FILE"),
    )
    app = build_application(settings)
    await app.startup()
    dispatcher = app.agent_runtime.dispatcher
    serving = app.get_module("llm-serving")
    mm = app.get_module("model-manager")

    try:
        _hr("loading model (this can take minutes)")
        await mm.load(settings.gguf_file)
        if not await _wait_serving(mm):
            print("ERROR: engine never became ready.", file=sys.stderr)
            return 3
        print(f"serving: {mm.is_serving} state={mm.engine_state.value} file={mm.loaded_file}")

        _hr("Part A — real model reachable (llm-serving.chat)")
        resp = await serving.chat([{"role": "user", "content": "Reply with the single word OK."}],
                                  max_tokens=8)
        print("chat response:", resp.get("choices", [{}])[0].get("message", {}).get("content"))

        _hr("Part B.1 — list_documents (gate=safe → executes)")
        r = await dispatcher.dispatch(Action("list_documents", {"subdir": "."}), "smoke")
        print(r.outcome.value, "→", r.result)
        docs = (r.result or {}).get("documents", [])
        target = args.doc or (docs[0] if docs else None)

        if target:
            _hr(f"Part B.2 — summarize_document(path={target!r})")
            r = await dispatcher.dispatch(Action("summarize_document", {"path": target}), "smoke")
            print(r.outcome.value, "→ ok:", (r.result or {}).get("ok"))
            print("summary:", (r.result or {}).get("summary"))

        _hr(f"Part B.3 — answer_question(question={args.question!r}, subdir='.')")
        r = await dispatcher.dispatch(
            Action("answer_question", {"question": args.question, "subdir": "."}), "smoke")
        print(r.outcome.value, "→ ok:", (r.result or {}).get("ok"),
              "answer_found:", (r.result or {}).get("answer_found"))
        print("answer:", (r.result or {}).get("answer"))
        print("citations:", (r.result or {}).get("citations"))

        _hr("Part B.4 — confirm→approve→execute (absolute path requires HITL)")
        abs_doc = os.path.join(os.path.abspath(docs_root), target) if target else None
        if abs_doc:
            action = Action("summarize_document", {"path": abs_doc})
            r = await dispatcher.dispatch(action, "smoke")
            print("dispatch:", r.outcome.value, "(expect pending — absolute path → confirm)")
            if r.outcome is Outcome.pending:
                dispatcher.approvals.approve(r.approval_id, action, _Approver())
                r2 = await dispatcher.execute_approved(r.approval_id, action)
                # NOTE: an absolute path OUTSIDE docs_root would be DocAccessError; an absolute
                # path INSIDE docs_root summarizes after approval. Either way: no crash, gated.
                print("execute_approved:", r2.outcome.value, "→ ok:", (r2.result or {}).get("ok"))

        _hr("SMOKE COMPLETE")
        return 0
    finally:
        await app.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
