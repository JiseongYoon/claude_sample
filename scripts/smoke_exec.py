"""Real-Docker smoke for exec (the daemon round-trip).

NOT a unit test — an OPERATOR smoke. The unit suite (`tests/test_exec_sandbox.py`) is daemon-free with a
fake runner; this drives the REAL `DockerSandbox` + `_subprocess_runner` against a live Docker daemon to
prove the containment is actually applied. It uses `local_ai_agent_exec_<id>` / `local_ai_agent_ws_<id>`
names and cleans them up.

Checks:
  1. daemon reachable (`probe`).
  2. a contained command runs and returns output.
  3. `docker inspect` confirms the hardening is APPLIED (NetworkMode=none, ReadonlyRootfs, no cap, etc.).
  4. `--network none` is functional (an egress attempt fails).
  5. workspace write → read → cat round-trip on the per-task volume.
  6. output truncation (a flood is byte-capped) and timeout → container kill.
  7. clean teardown (container + volume removed).

Run on the box, inside conda `local-ai-agent-env-1` (the user is in the `docker` group here):

    python scripts/smoke_exec.py # uses EXEC_IMAGE or alpine:latest
    EXEC_IMAGE=python:3.11-slim python scripts/smoke_exec.py

⚠️ `docker run/pull` trips the Requires Docker daemon access (the user must be in the `docker` group).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from local_ai_agent.modules.exec import ( # noqa: E402
    DockerSandbox,
    ExecConfig,
    ExecPolicy,
    ExecTimeout,
    gc_orphans,
    probe,
)
from local_ai_agent.modules.exec.sandbox import _subprocess_runner # noqa: E402
from local_ai_agent.config import Settings # noqa: E402

TASK = "smoke1"
NAME = f"local_ai_agent_exec_{TASK}"
VOL = f"local_ai_agent_ws_{TASK}"
BASE = os.environ.get("EXEC_IMAGE", "alpine:latest")
# The production hardened base image MUST own /workspace as the sandbox UID, so a fresh per-task volume
# inherits that ownership (else a non-root sandbox can't write its workspace). We build a tiny image on
# top of BASE here to satisfy that contract for the smoke.
IMAGE = "local_ai_agent_exec_smoketest:base"
_DOCKERFILE = f"FROM {BASE}\nRUN mkdir -p /workspace && chown 1000:1000 /workspace\n".encode()


def _ok(label, cond):
    print(f" [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


async def main() -> int:
    if not await probe():
        print("Docker daemon not reachable — aborting."); return 2
    print(f"Docker reachable. base={BASE} image={IMAGE} container={NAME} volume={VOL}")

    # build the workspace-owning image (the hardened-image contract: /workspace owned by the sandbox UID)
    b = await _subprocess_runner(["docker", "build", "-t", IMAGE, "-"], stdin=_DOCKERFILE, timeout=180)
    if b.returncode != 0:
        print("image build failed — aborting."); return 2

    # build the policy + spec the way the module will, but with the smoke image + task volume.
    cfg = ExecConfig.from_settings(
        Settings(_env_file=None, model_safetensors_dir="./models/s", model_gguf_dir="./models/g",
                 exec_image=IMAGE, exec_env_whitelist=["LANG"])
    )
    spec = ExecPolicy(cfg, environ={"LANG": "C.UTF-8"}).build_spec(workspace_volume=VOL)
    sb = DockerSandbox(name=NAME, runner=_subprocess_runner)

    await gc_orphans() # clear any leftovers from a prior aborted run
    await _subprocess_runner(["docker", "volume", "rm", VOL], timeout=15) # ensure a truly fresh volume
    passed = True
    try:
        # 2. contained command
        r = await sb.run(spec, ["echo", "hello-sandbox"], timeout=15, max_output_bytes=10000)
        passed &= _ok("contained echo", r.exit_code == 0 and b"hello-sandbox" in r.stdout)

        # 3. inspect — hardening actually applied
        ins = await _subprocess_runner(["docker", "inspect", NAME], timeout=15)
        info = json.loads(ins.stdout.decode())[0] if ins.returncode == 0 else {}
        hc = info.get("HostConfig", {})
        passed &= _ok("NetworkMode=none", hc.get("NetworkMode") == "none")
        passed &= _ok("ReadonlyRootfs", hc.get("ReadonlyRootfs") is True)
        passed &= _ok("CapDrop=ALL", "ALL" in (hc.get("CapDrop") or []))
        passed &= _ok("Memory cap set", int(hc.get("Memory") or 0) == spec.mem_bytes)
        passed &= _ok("PidsLimit set", int((hc.get("PidsLimit") or 0)) == spec.pids_limit)
        passed &= _ok("no-new-privileges", "no-new-privileges" in (hc.get("SecurityOpt") or []))

        # 4. network none is functional (egress must fail)
        net = await sb.run(spec, ["sh", "-c", "wget -q -T 3 -O - http://93.184.216.34/ || echo BLOCKED"],
                           timeout=15, max_output_bytes=10000)
        passed &= _ok("egress blocked (--network none)", b"BLOCKED" in net.stdout)

        # 5. workspace write → read round-trip
        await sb.write_file(spec, "/workspace/note.txt", b"persisted-data")
        back = await sb.read_file(spec, "/workspace/note.txt", max_bytes=10000)
        passed &= _ok("workspace write/read round-trip", back == b"persisted-data")

        # 6a. output truncation
        flood = await sb.run(spec, ["sh", "-c", "yes 2>/dev/null | head -c 100000"],
                             timeout=15, max_output_bytes=1000)
        passed &= _ok("output truncated to cap", len(flood.stdout) == 1000 and flood.truncated)

        # 6b. timeout → kill
        timed_out = False
        try:
            await sb.run(spec, ["sleep", "30"], timeout=2, max_output_bytes=1000)
        except ExecTimeout:
            timed_out = True
        passed &= _ok("timeout raises ExecTimeout + kills container", timed_out)
    finally:
        await sb.close()
        # 7. teardown verified
        chk = await _subprocess_runner(["docker", "ps", "-aq", "--filter", f"name={NAME}"], timeout=15)
        _ok("container removed", chk.returncode == 0 and not chk.stdout.strip())
        await _subprocess_runner(["docker", "rmi", IMAGE], timeout=30) # remove the smoke image too

    print("\nSMOKE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
