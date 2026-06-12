#!/usr/bin/env bash
# Web-UI real-API e2e runner. Operator/CI-run (like the backend smokes), NOT the unit
# suite. Starts the live fake core (uvicorn + scripted model + gated delete_file), then drives the REAL
# frontend ApiClient over a REAL WebSocket through connect → run_task → approval_request → approve →
# task_result. Run from the PROJECT ROOT:
#     bash web/e2e/run_e2e.sh
# Requires the conda env (Python + uvicorn) and web deps installed (npm --prefix web install).
set -u
PORT="${E2E_PORT:-8157}"
PY="${E2E_PYTHON:-python}"

"$PY" web/e2e/serve_fake_core.py --port "$PORT" > /tmp/e2e_core.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

if ! curl -sf --retry-connrefused --retry 60 --retry-delay 1 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
  echo "[e2e] core failed to start; log:"; tail -20 /tmp/e2e_core.log; exit 2
fi
echo "[e2e] core ready on ${PORT} (pid ${SERVER_PID})"

# npm exec puts the env's node on PATH for tsx; the port is passed as argv (env vars don't propagate
# reliably through `npm exec`). NOTE: do NOT `pkill -f serve_fake_core` from here — this script's own
# command line contains that string, so a pattern kill would self-terminate. Kill the specific PID only.
npm --prefix web exec -- tsx web/e2e/run.mts "$PORT"
exit $?
