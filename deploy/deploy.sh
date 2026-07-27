#!/usr/bin/env bash
# Deploy CareShell into a NemoClaw sandbox.
#
#     ./deploy/deploy.sh [sandbox-name]
#
# Idempotent: safe to re-run. Syncs the code, installs dependencies, resolves the model
# served by the attached inference, and starts the nurse console.
#
# Verified against nemoclaw v0.0.93 / openshell 0.0.85. Both are alpha; if a command
# below is rejected, check `nemoclaw --help` for a renamed subcommand.

set -euo pipefail

SANDBOX="${1:-$(nemoclaw list --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("defaultSandbox",""))' 2>/dev/null || echo careshell)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP=/sandbox/app
cd "$ROOT"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
in_sandbox() { nemoclaw "$SANDBOX" exec --timeout "${TIMEOUT:-120}" -- "$@"; }

step "Target sandbox: $SANDBOX"
if ! in_sandbox true >/dev/null 2>&1; then
  echo "    cannot exec into '$SANDBOX'. Run 'nemoclaw list' to see what exists." >&2
  exit 1
fi
echo "    reachable"

step "Checking the sandbox policy"
# CareShell needs no policy delta on a stock sandbox: the baseline already grants
# read_write on /sandbox and routes inference.local, and the console reaches the host
# through a gRPC forward rather than a policy ingress rule. Presets are network-only --
# they cannot declare filesystem or ingress rules at all. See the long comment in
# openshell/careshell-preset.yaml. Set CARESHELL_APPLY_PRESET=1 to layer one on anyway.
#
# If you do apply one, use `policy-add` (layers onto the baseline), never
# `openshell policy set` (replaces it, stripping inference.local and the gateway
# dial-back WebSocket).
if [[ "${CARESHELL_APPLY_PRESET:-0}" == "1" ]]; then
  nemoclaw "$SANDBOX" policy-add --from-file openshell/careshell-preset.yaml --yes 2>&1 | tail -2
fi

if nemoclaw "$SANDBOX" policy-get 2>/dev/null | grep -qi 'inference'; then
  echo "    inference route present"
else
  echo "    WARNING: no inference route in the policy. Check 'nemoclaw $SANDBOX policy-get'." >&2
fi
if nemoclaw "$SANDBOX" policy-get 2>/dev/null | grep -qE '^\s+- /sandbox$'; then
  echo "    /sandbox is read_write"
else
  echo "    WARNING: /sandbox is not read_write; SQLite and the live feed need it." >&2
fi

step "Syncing code to $APP"
# `nemoclaw upload <dir> <dest>` places the directory INSIDE dest as dest/<basename>,
# so upload to a staging path and move the contents into place.
STAGE="$(mktemp -d)/careshell-src"
mkdir -p "$STAGE"
tar -cf - --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
          --exclude='*.pyc' --exclude='.pytest_cache' \
    careshell alerting dashboard tools tests data workspace requirements.txt pytest.ini \
  | (cd "$STAGE" && tar -xf -)
nemoclaw "$SANDBOX" upload "$STAGE" /sandbox/_stage >/dev/null
in_sandbox sh -c "
  set -e
  mkdir -p $APP
  for d in careshell alerting dashboard tools tests data workspace; do
    rm -rf $APP/\$d && cp -r /sandbox/_stage/careshell-src/\$d $APP/\$d
  done
  cp /sandbox/_stage/careshell-src/requirements.txt /sandbox/_stage/careshell-src/pytest.ini $APP/
  rm -rf /sandbox/_stage
  find $APP -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
"
rm -rf "$STAGE"
echo "    synced"

step "Installing dependencies"
# The sandbox image is PEP 668 externally-managed, so a venv is required.
TIMEOUT=420 in_sandbox sh -c "
  set -e
  [ -x $APP/.venv/bin/python ] || python3 -m venv $APP/.venv
  $APP/.venv/bin/pip install --quiet --disable-pip-version-check -r $APP/requirements.txt
  $APP/.venv/bin/python -c 'import pydantic,yaml,requests,fastapi; print(\"    dependencies ok\")'
"

step "Resolving a chat model served by inference.local"
# https, not http: plain http is refused by the egress proxy with a 403.
#
# Do NOT just take data[0] -- on an OpenAI-backed gateway that is often an embedding
# model (text-embedding-ada-002), which cannot serve chat completions. Prefer the model
# the sandbox is configured with, then anything that is clearly not embeddings/audio.
# Note: `nemoclaw list --json` returns sandboxes as a LIST of objects, unlike
# ~/.nemoclaw/sandboxes.json which keys them by name. Don't assume a dict here.
# The sandbox name goes in as argv, not an env var: `VAR=x cmd | python3` sets VAR for
# `cmd`, not for the process on the other side of the pipe.
CONFIGURED="$(nemoclaw list --json 2>/dev/null \
  | python3 -c '
import json, sys
name = sys.argv[1]
boxes = json.load(sys.stdin).get("sandboxes", [])
if isinstance(boxes, dict):
    boxes = list(boxes.values())
print(next((b.get("model", "") for b in boxes if b.get("name") == name), ""))
' "$SANDBOX" 2>/dev/null || true)"
[[ -n "$CONFIGURED" ]] && echo "    sandbox is configured for: $CONFIGURED"

MODELS_JSON="$(TIMEOUT=90 in_sandbox sh -c \
  'curl -s --max-time 45 https://inference.local/v1/models' 2>/dev/null \
  | grep -v 'Active gateway set' || true)"
MODEL_ID="$(printf '%s' "$MODELS_JSON" | python3 deploy/pick_model.py "$CONFIGURED" 2>/dev/null || true)"
if [[ -n "$MODEL_ID" ]]; then
  echo "    $MODEL_ID"
  echo "$MODEL_ID" > .careshell-model
else
  echo "    WARNING: could not reach inference.local. Run ./deploy/preflight.sh $SANDBOX." >&2
fi

step "Running the safety tests inside the sandbox"
# Trust the exit code, not the text. pytest.ini already sets `-q`, so passing another
# `-q` here would make it `-qq`, which suppresses the "N passed" summary line entirely
# and makes any grep for it silently fail.
TEST_OUT="$(TIMEOUT=300 in_sandbox sh -c \
  "cd $APP && PYTHONPATH=$APP $APP/.venv/bin/python -m pytest -p no:cacheprovider 2>&1 | tail -25" || true)"
if printf '%s' "$TEST_OUT" | grep -qE '[0-9]+ passed' \
   && ! printf '%s' "$TEST_OUT" | grep -qE '[0-9]+ (failed|error)'; then
  echo "    $(printf '%s' "$TEST_OUT" | grep -oE '[0-9]+ passed.*' | tail -1 | xargs)"
else
  echo "    WARNING: tests did not pass cleanly:" >&2
  printf '%s\n' "$TEST_OUT" | tail -6 >&2
fi

step "Starting the nurse console on :8000"
# Two traps here, both learned the hard way:
#
#   * A plain `nohup ... &` inside `exec` is reaped when the exec channel closes.
#     `setsid` with stdin detached is what actually survives.
#   * `pkill -f 'uvicorn dashboard.app'` matches the `sh -c` command line that CONTAINS
#     that string -- so it kills its own shell, and the step dies silently with exit 1.
#     Track the pid in a file instead of pattern-matching.
in_sandbox sh -c "
  PIDFILE=$APP/logs/console.pid
  mkdir -p $APP/logs
  [ -f \"\$PIDFILE\" ] && kill \"\$(cat \"\$PIDFILE\")\" 2>/dev/null || true
  setsid env CARESHELL_DB=$APP/workspace/careshell.db PYTHONPATH=$APP \
    $APP/.venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 \
    < /dev/null > $APP/logs/console.log 2>&1 &
  echo \$! > \"\$PIDFILE\"
  sleep 5
  if curl -sf --max-time 5 http://127.0.0.1:8000/healthz > /dev/null; then
    echo '    console healthy'
  else
    echo '    WARNING: console did not answer /healthz'
    tail -5 $APP/logs/console.log
  fi
" || echo "    WARNING: console step failed" >&2

step "Forwarding :8000 to the host"
openshell forward stop 8000 >/dev/null 2>&1 || true
nohup openshell forward service --target-port 8000 --local 8000 "$SANDBOX" \
  > /tmp/careshell-forward.log 2>&1 &
sleep 5
if curl -sf --max-time 5 http://127.0.0.1:8000/healthz >/dev/null; then
  echo "    http://127.0.0.1:8000"
else
  echo "    WARNING: forward not answering. See /tmp/careshell-forward.log" >&2
fi

cat <<EOF

Deployed to sandbox '$SANDBOX'.

  Nurse console:  http://127.0.0.1:8000
  Model:          ${MODEL_ID:-<unresolved>}

Next:
  ./deploy/preflight.sh $SANDBOX     # verify the whole stack
  ./deploy/demo.sh $SANDBOX          # streaming demo
EOF
