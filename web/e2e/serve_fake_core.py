"""Live core for theWeb-UI real-API e2e. NOT part of the Python suite.

Serves the REAL FastAPI gateway over a real socket (uvicorn) with a hand-built AgentRuntime driven by a
*scripted* fake model (no real Gemma) + a gated `delete_file` tool — so the agent loop proposes a
needs_confirmation action and the WS approval round-trip fires exactly as in production. The frontend's
real `ApiClient` then drives it from Node over a real WebSocket (see `run.mts`). Mirrors the harness in
`tests/test_agent_ws.py`.

Run (from project root, conda env):
 conda run -n local-ai-agent-env-1 python web/e2e/serve_fake_core.py --port 8137
"""
from __future__ import annotations

import argparse

import uvicorn

from local_ai_agent.config import Settings
from local_ai_agent.core.app import Application
from local_ai_agent.core.gateway import create_gateway
from local_ai_agent.modules.model_manager.module import ModelManagerModule
from local_ai_agent.modules.orchestrator.loop import AssistantTurn, ToolCall
from local_ai_agent.modules.orchestrator.runtime import AgentRuntime
from local_ai_agent.modules.safety.approval import PendingApprovals
from local_ai_agent.modules.safety.dispatcher import ToolDispatcher
from local_ai_agent.modules.safety.gate import SafetyGate


class ScriptedModel:
 """Turn 1: propose the gated `delete_file` tool. Turn 2 (post-approval): final answer.

 also implements `complete_stream` (so the live agent path STREAMS token deltas)
 — a text turn is emitted as token fragments via `on_token`; a tool-call turn streams no text."""

 def __init__(self, turns):
 self.turns = turns
 self.i = 0

 def _next(self):
 t = self.turns[min(self.i, len(self.turns) - 1)]
 self.i += 1
 return t

 async def complete(self, messages, tools): # noqa: ARG002
 return self._next()

 async def complete_stream(self, messages, tools, on_token): # noqa: ARG002
 t = self._next()
 if t.text:
 for frag in (t.text[: len(t.text) // 2], t.text[len(t.text) // 2:]):
 if frag:
 await on_token(frag)
 return t


class RecordingTool:
 def __init__(self, name, result="ok"):
 self.name = name
 self.result = result
 self.calls = []

 async def run(self, args):
 self.calls.append(args)
 return self.result


def build_app(settings: Settings) -> Application:
 confirm = ToolCall("c1", "delete_file", {"path": "workspace/old.txt"})
 model = ScriptedModel([
 AssistantTurn(tool_calls=[confirm]),
 AssistantTurn(text="deleted the file"),
 ])
 tool = RecordingTool("delete_file", "deleted")
 # a real model-manager module (load-on-demand — never launches a server) so thediscovery
 # endpoint GET /model/files returns the GGUF names in model_gguf_dir live, and GET /model/status works.
 app = Application(modules=[ModelManagerModule(settings)])
 approvals = PendingApprovals(timeout_seconds=300)
 dispatcher = ToolDispatcher(gate=SafetyGate(), approvals=approvals, tools=[tool])
 app.agent_runtime = AgentRuntime(
 gate=SafetyGate(), approvals=approvals, dispatcher=dispatcher, model=model,
 decision_timeout=30.0,
 )
 # attach a real IngestStore over a temp docs_root so POST /ingest works live (the
 # frontend ApiClient.uploadFile drives it from Node — see run.mtschecks).
 import tempfile
 from pathlib import Path

 from local_ai_agent.modules.docqa.ingest import IngestPolicy, IngestStore

 docs_root = Path(tempfile.mkdtemp(prefix="e2e_ingest_"))
 app.ingest_store = IngestStore(IngestPolicy(docs_root))
 return app


def main() -> None:
 ap = argparse.ArgumentParser()
 ap.add_argument("--port", type=int, default=8137)
 ap.add_argument("--host", default="127.0.0.1")
 args = ap.parse_args()
 # a temp GGUF dir with a fake model file so GET /model/files () returns it live.
 import tempfile
 from pathlib import Path

 gguf_dir = Path(tempfile.mkdtemp(prefix="e2e_gguf_"))
 (gguf_dir / "e2e-model.gguf").write_bytes(b"\x00")
 # auth disabled → the WS principal is anonymous full-scope (agent:run + approve). The Node client
 # sends no Origin header, so CORS (browser-only) doesn't apply. (Real deploys configure auth + CORS.)
 # A jwt_secret is set so the mint route (/auth/token) can sign a token even with auth off.
 settings = Settings(_env_file=None, model_safetensors_dir=str(gguf_dir), model_gguf_dir=str(gguf_dir),
 auth_enabled=False, jwt_secret="e2e-only-jwt-secret-not-a-real-deploy-secret-000")
 uvicorn.run(create_gateway(build_app(settings), settings), host=args.host, port=args.port,
 log_level="warning")


if __name__ == "__main__":
 main()
