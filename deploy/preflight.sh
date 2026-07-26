#!/usr/bin/env bash
# CareShell preflight: verify the GB10 / NemoClaw / OpenShell / vLLM stack before a demo.
#
#     ./deploy/preflight.sh [sandbox-name]
#
# Every check runs from INSIDE the sandbox. The host can reach things the sandbox cannot,
# so a host-side curl proves nothing about what CareShell will be able to do.

set -uo pipefail

SANDBOX="${1:-careshell}"
PASS=0
FAIL=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
note() { printf '        %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

in_sandbox() { nemoclaw "$SANDBOX" exec -- "$@" 2>&1; }

head_ "1. NemoClaw CLI"
if command -v nemoclaw >/dev/null 2>&1; then
  ok "nemoclaw found: $(command -v nemoclaw)"
  note "version: $(nemoclaw --version 2>&1 | head -1)"
else
  bad "nemoclaw not on PATH"
  echo; echo "Cannot continue without the NemoClaw CLI."; exit 1
fi

head_ "2. Sandbox '$SANDBOX' is reachable"
if in_sandbox true >/dev/null 2>&1; then
  ok "exec into '$SANDBOX' works"
else
  bad "cannot exec into '$SANDBOX'"
  note "create it first, then re-run this script"
  echo; echo "Cannot continue."; exit 1
fi

head_ "3. GPU visible inside the sandbox"
GPU="$(in_sandbox nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
if [[ -n "$GPU" && "$GPU" != *"not found"* && "$GPU" != *"Error"* ]]; then
  ok "GPU: $GPU"
else
  bad "no GPU inside the sandbox"
  note "check runtime.gpu in the preset delta"
fi

head_ "4. vLLM reachable at inference.local"
MODELS="$(in_sandbox curl -s --max-time 10 http://inference.local/v1/models)"
if [[ "$MODELS" == *'"id"'* ]]; then
  ok "inference.local responded"
  MODEL_ID="$(printf '%s' "$MODELS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
  if [[ -n "$MODEL_ID" ]]; then
    note "model id: $MODEL_ID"
    note "pass this to careshell.run with --model \"$MODEL_ID\""
  fi
else
  bad "inference.local did not return a model list"
  note "if this broke after a policy change, the baseline was replaced instead of extended"
  note "re-apply with: nemoclaw $SANDBOX policy-add --from-file openshell/careshell-preset.yaml"
fi

head_ "5. Egress is denied"
EGRESS="$(in_sandbox curl -sS -I --max-time 10 https://example.com)"
if [[ "$EGRESS" == *"403"* || "$EGRESS" == *"(56)"* || "$EGRESS" == *"(7)"* || "$EGRESS" == *"Could not resolve"* ]]; then
  ok "outbound blocked"
  note "$(printf '%s' "$EGRESS" | head -1)"
  note "this is a userspace CONNECT-proxy refusal, not a kernel drop -- describe it accurately"
else
  bad "outbound traffic appears to be ALLOWED"
  note "response: $(printf '%s' "$EGRESS" | head -1)"
fi

head_ "6. Mounts present and writable"
for path in /app/data/patients /app/workspace /app/data/live; do
  if in_sandbox test -d "$path" >/dev/null 2>&1; then ok "$path exists"; else bad "$path missing"; fi
done
if in_sandbox sh -c 'touch /app/workspace/.preflight && rm -f /app/workspace/.preflight' >/dev/null 2>&1; then
  ok "/app/workspace is writable (SQLite needs this)"
else
  bad "/app/workspace is not writable -- SQLite cannot open its database"
fi

head_ "7. Python dependencies"
DEPS="$(in_sandbox python3 -c 'import pydantic, yaml, requests, fastapi; print("ok")')"
if [[ "$DEPS" == *ok* ]]; then
  ok "pydantic, pyyaml, requests, fastapi importable"
else
  bad "missing dependencies"
  note "run: nemoclaw $SANDBOX exec -- pip install -r requirements.txt"
fi

head_ "8. Safety logic (no GPU, no model, no network)"
TESTS="$(in_sandbox python3 -m pytest tests/test_reconciler.py -q --no-header 2>&1 | tail -3)"
if [[ "$TESTS" == *"passed"* && "$TESTS" != *"failed"* ]]; then
  ok "reconciler tests pass"
  note "$(printf '%s' "$TESTS" | tail -1)"
else
  bad "reconciler tests did not pass"
  note "$(printf '%s' "$TESTS" | tail -2)"
fi

printf '\n\033[1mSummary:\033[0m %d passed, %d failed\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
