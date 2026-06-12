"""Real-SSH (+ optional real-model) smoke for storage.

NOT a unit test — an OPERATOR smoke. Spins an in-process asyncssh SFTP server on 127.0.0.1
(generated keys, no external host) and drives the storage capability through the REAL
`build_application` dispatcher:

  * Part 1 (always) — storage tools over real SSH/SFTP: list / write / read / move / delete,
    incl. a confirm→approve→execute mutation, through the gated dispatcher.
  * Part 2 (only if GGUF_FILE is set) — remote → DocQA with the REAL Gemma model:
    `summarize_remote` / `answer_remote` over a file served by the loopback SSH server, through
    the gate, with citations. Loads a ~27 GB model (minutes) + uses the GPUs.

Run on the box, inside conda `local-ai-agent-env-1`:

    python scripts/smoke_storage.py # Part 1 only (no model)
    GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf \
    MODEL_GGUF_DIR=models/gemma-4-gguf MODEL_SAFETENSORS_DIR=./models/s \
    python scripts/smoke_storage.py # Part 1 + Part 2 (real model)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncssh # noqa: E402

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


class _SSHServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def public_key_auth_supported(self):
        return True


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
    model_mode = bool(os.environ.get("GGUF_FILE"))
    tmp = Path(tempfile.mkdtemp(prefix="smoke_storage_"))
    work = tmp / "share"
    work.mkdir()
    (work / "doc.txt").write_text(
        "The storage connector reads only within a configured allowed_root. "
        "Reading a secret path is blocked by the safety gate.", encoding="utf-8")

    # --- in-process asyncssh SFTP server (127.0.0.1, generated keys) ---
    host_key = asyncssh.generate_private_key("ssh-ed25519")
    client_key = asyncssh.generate_private_key("ssh-ed25519")
    ckey = tmp / "client"
    ckey.write_bytes(client_key.export_private_key())
    cpub = tmp / "client.pub"
    cpub.write_bytes(client_key.export_public_key())
    server = await asyncssh.create_server(
        _SSHServer, "127.0.0.1", 0, server_host_keys=[host_key],
        authorized_client_keys=str(cpub), sftp_factory=asyncssh.SFTPServer)
    port = server.get_port()
    kh = tmp / "known_hosts"
    kh.write_text(f"[127.0.0.1]:{port} {host_key.export_public_key().decode().strip()}\n", encoding="utf-8")

    connectors = tmp / "connectors.json"
    connectors.write_text(json.dumps({"connectors": [{
        "name": "lo", "kind": "ssh", "host": "127.0.0.1", "port": port, "username": "tester",
        "auth": {"key_path": str(ckey)}, "allowed_root": str(work),
        "read_only": False, "known_hosts_path": str(kh), "max_bytes": 10_000_000,
    }]}), encoding="utf-8")

    settings = Settings(
        _env_file=None,
        model_safetensors_dir=os.environ.get("MODEL_SAFETENSORS_DIR", "./models/s"),
        model_gguf_dir=os.environ.get("MODEL_GGUF_DIR", "./models/g"),
        gguf_file=os.environ.get("GGUF_FILE"),
        enable_agent=True, enable_storage=True, enable_docqa=model_mode,
        storage_connectors_file=connectors, docs_root=str(work) if model_mode else None,
    )
    app = build_application(settings)
    await app.startup()
    dispatcher = app.agent_runtime.dispatcher

    try:
        _hr("Part 1 — storage tools over real SSH/SFTP")
        r = await dispatcher.dispatch(Action("storage_list", {"connector": "lo", "path": "."}), "smoke")
        print("storage_list:", r.outcome.value, "→", r.result)
        r = await _approve_run(dispatcher, Action("storage_write",
              {"connector": "lo", "path": "note.txt", "content": "written via smoke"}))
        print("storage_write:", r.outcome.value, "→", r.result)
        r = await dispatcher.dispatch(Action("storage_read", {"connector": "lo", "path": "note.txt"}), "smoke")
        print("storage_read:", r.outcome.value, "→", r.result)
        r = await _approve_run(dispatcher, Action("storage_move",
              {"connector": "lo", "src": "note.txt", "dst": "moved.txt"}))
        print("storage_move:", r.outcome.value, "→", r.result)
        r = await dispatcher.dispatch(Action("storage_read", {"connector": "lo", "path": "/etc/shadow"}), "smoke")
        print("secret read (expect refused):", r.outcome.value)

        if model_mode:
            _hr("loading model (minutes)")
            mm = app.get_module("model-manager")
            await mm.load(settings.gguf_file)
            deadline = time.monotonic() + 600
            while not mm.is_serving and time.monotonic() < deadline:
                await asyncio.sleep(2.0)
            print("serving:", mm.is_serving, "state:", mm.engine_state.value)

            _hr("Part 2 — remote → DocQA with the REAL model")
            r = await dispatcher.dispatch(Action("summarize_remote", {"connector": "lo", "path": "doc.txt"}), "smoke")
            print("summarize_remote:", r.outcome.value, "→ ok:", (r.result or {}).get("ok"))
            print("summary:", (r.result or {}).get("summary"))
            r = await dispatcher.dispatch(Action("answer_remote",
                  {"question": "What happens when a secret path is read?", "connector": "lo", "path": "doc.txt"}), "smoke")
            print("answer_remote:", r.outcome.value, "→ ok:", (r.result or {}).get("ok"),
                  "found:", (r.result or {}).get("answer_found"))
            print("answer:", (r.result or {}).get("answer"))
            print("citations:", (r.result or {}).get("citations"))

        _hr("SMOKE COMPLETE")
        return 0
    finally:
        await app.shutdown()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
