#!/usr/bin/env bash
# Deploy CareShell into a NemoClaw sandbox on GB10.
#
#     ./deploy/deploy.sh [sandbox-name]
#
# Idempotent: safe to re-run. It applies the policy delta, installs dependencies, and
# starts the nurse console. It does NOT start a timeline run -- use deploy/demo.sh.

set -euo pipefail

SANDBOX="${1:-careshell}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

step "Preparing host directories"
mkdir -p data/live workspace logs
echo "    data/live workspace logs"

step "Applying the OpenShell policy delta"
# policy-add LAYERS onto NemoClaw's baseline. Never `openshell policy set` -- that
# replaces the baseline and silently strips inference.local and the gateway WebSocket.
nemoclaw "$SANDBOX" policy-add --from-file openshell/careshell-preset.yaml
echo "    applied openshell/careshell-preset.yaml"

step "Verifying the baseline survived"
if nemoclaw "$SANDBOX" policy-show | grep -q 'inference.local'; then
  echo "    inference.local still present"
else
  echo "    ERROR: inference.local is missing from the policy." >&2
  echo "    The delta was applied as a replacement. Recreate the sandbox and retry." >&2
  exit 1
fi

step "Installing Python dependencies"
nemoclaw "$SANDBOX" exec -- pip install --quiet -r requirements.txt
echo "    done"

step "Resolving the model served by vLLM"
MODEL_ID="$(nemoclaw "$SANDBOX" exec -- curl -s --max-time 10 http://inference.local/v1/models \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
if [[ -n "$MODEL_ID" ]]; then
  echo "    $MODEL_ID"
  echo "$MODEL_ID" > .careshell-model
  echo "    written to .careshell-model"
else
  echo "    WARNING: could not reach inference.local. Run ./deploy/preflight.sh." >&2
fi

step "Starting the nurse console on :8000"
nemoclaw "$SANDBOX" exec -- pkill -f 'uvicorn dashboard.app' 2>/dev/null || true
nemoclaw "$SANDBOX" exec -- \
  sh -c 'nohup uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 \
         > logs/console.log 2>&1 & echo started'
sleep 2
if nemoclaw "$SANDBOX" exec -- curl -sf --max-time 5 http://127.0.0.1:8000/healthz >/dev/null; then
  echo "    console healthy"
else
  echo "    WARNING: console did not answer /healthz. See logs/console.log." >&2
fi

cat <<EOF

Deployed.

  Nurse console:  http://<gb10-lan-ip>:8000
  Model:          ${MODEL_ID:-<unresolved>}

Next:
  ./deploy/preflight.sh $SANDBOX     # verify the whole stack
  ./deploy/demo.sh $SANDBOX          # run the streaming demo
EOF
