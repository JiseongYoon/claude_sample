"""Real-fetch (+ optional real-SearXNG + real-model) smoke for browser.

NOT a unit test — an OPERATOR smoke. Drives the browser capability through the REAL
`build_application` dispatcher, with the gated confirm→approve→execute round-trip:

  * Part 1 (always) — `open_url` against a real PUBLIC page (real httpx fetch + IP-pin + trafilatura
    extraction) through the gate, plus an SSRF check that `open_url http://127.0.0.1/` is refused
    with no real connection.
  * Part 2 (only if BROWSER_SEARXNG_URL is set AND GGUF_FILE is set) — `web_answer` end-to-end with a
    REAL SearXNG instance + the REAL Gemma model: search → fetch top-N → grounded answer with URL
    citations, through the gate. Loads a ~27 GB model (minutes) + uses the GPUs.

Requires network egress. Run on the box, inside conda `local-ai-agent-env-1`:

    PUBLIC_URL=https://example.com python scripts/smoke_browser.py # Part 1 only

    BROWSER_SEARXNG_URL=http://127.0.0.1:8888/search \
    GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf \
    MODEL_GGUF_DIR=models/gemma-4-gguf MODEL_SAFETENSORS_DIR=./models/s \
    python scripts/smoke_browser.py # Part 1 + Part 2

A self-hosted SearXNG (docker `local-ai-agent-searxng`) must expose JSON output
(`search.formats: [json]` in its settings.yml) for Part 2.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_ai_agent.config import Settings # noqa: E402
from local_ai_agent.main import build_application # noqa: E402
from local_ai_agent.modules.safety.dispatcher import Outcome # noqa: E402
from local_ai_agent.modules.safety.gate import Action # noqa: E402


@dataclass
class _Approver:
    subject: str = "smoke"
    scopes: frozenset = field(default_factory=lambda: frozenset({"approve"}))

    def has_scope(self, scope):
        return "*" in self.scopes or scope in self.scopes


def _hr(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


async def _approve_run(dispatcher, action, who="smoke"):
    """dispatch → (if pending) approve → execute_approved. Returns the final result."""
    res = await dispatcher.dispatch(action, who)
    if res.outcome is Outcome.pending:
        dispatcher.approvals.approve(res.approval_id, action, _Approver())
        res = await dispatcher.execute_approved(res.approval_id, action)
    return res


async def main() -> int:
    searxng_url = os.environ.get("BROWSER_SEARXNG_URL")
    model_mode = bool(os.environ.get("GGUF_FILE")) and bool(searxng_url)
    public_url = os.environ.get("PUBLIC_URL", "https://example.com")

    settings = Settings(
        _env_file=None,
        model_safetensors_dir=os.environ.get("MODEL_SAFETENSORS_DIR", "./models/s"),
        model_gguf_dir=os.environ.get("MODEL_GGUF_DIR", "./models/g"),
        gguf_file=os.environ.get("GGUF_FILE"),
        enable_agent=True, enable_browser=True,
        browser_searxng_url=searxng_url,
    )
    app = build_application(settings)
    await app.startup()
    dispatcher = app.agent_runtime.dispatcher

    try:
        _hr("Part 1 — open_url against a real public page (gated)")
        r = await _approve_run(dispatcher, Action("open_url", {"url": public_url}))
        res = r.result or {}
        print(f"open_url {public_url}:", r.outcome.value, "→ ok:", res.get("ok"), "status:", res.get("status"))
        print("title:", res.get("title"))
        print("text[:200]:", (res.get("text") or "")[:200])

        _hr("SSRF check — open_url http://127.0.0.1/ (expect refused, no connection)")
        r = await _approve_run(dispatcher, Action("open_url", {"url": "http://127.0.0.1/"}))
        print("internal open_url:", r.outcome.value, "→ ok:", (r.result or {}).get("ok"),
              "(False = contained)")

        if model_mode:
            _hr("loading model (minutes)")
            mm = app.get_module("model-manager")
            await mm.load(settings.gguf_file)
            deadline = time.monotonic() + 600
            while not mm.is_serving and time.monotonic() < deadline:
                await asyncio.sleep(2.0)
            print("serving:", mm.is_serving, "state:", mm.engine_state.value)

            _hr("Part 2 — web_answer with REAL SearXNG + REAL model (gated)")
            q = os.environ.get("SMOKE_QUERY", "What is the capital of France?")
            r = await _approve_run(dispatcher, Action("web_answer", {"query": q}))
            res = r.result or {}
            print("web_answer:", r.outcome.value, "→ ok:", res.get("ok"), "found:", res.get("answer_found"))
            print("answer:", res.get("answer"))
            print("citations:", res.get("citations"))
            print("pages:", res.get("pages"))
        else:
            _hr("Part 2 skipped (set BROWSER_SEARXNG_URL + GGUF_FILE to run web_answer)")

        _hr("SMOKE COMPLETE")
        return 0
    finally:
        await app.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
