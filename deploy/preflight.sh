#!/usr/bin/env bash
# CareShell preflight: verify the stack before a demo.
#
#     ./deploy/preflight.sh [sandbox-name]
#
# Every check runs from INSIDE the sandbox. The host can reach things the sandbox cannot,
# so a host-side curl proves nothing about what CareShell will be able to do.
#
# Verified against nemoclaw v0.0.93 / openshell 0.0.85.

set -uo pipefail

SANDBOX="${1:-my-gpt-claw}"
APP=/sandbox/app
PASS=0
FAIL=0
WARN=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
note() { printf '        %s\n' "$1"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# `grep -v` exits 1 when it filters out every line, so never branch on this pipeline's
# status -- branch on the captured text instead. `|| true` keeps `set -o pipefail`-ish
# surprises out of the checks below.
in_sandbox() {
  nemoclaw "$SANDBOX" exec --timeout "${TIMEOUT:-90}" -- "$@" 2>&1 \
    | grep -v 'Active gateway set' || true
}

head_ "1. NemoClaw CLI"
if command -v nemoclaw >/dev/null 2>&1; then
  ok "nemoclaw $(nemoclaw --version 2>&1 | head -1 | tr -d '\n')"
else
  bad "nemoclaw not on PATH"; echo; exit 1
fi

head_ "2. Sandbox '$SANDBOX' reachable"
if [[ "$(in_sandbox sh -c 'echo __alive__')" == *__alive__* ]]; then
  ok "exec works"
else
  bad "cannot exec into '$SANDBOX'"
  note "run 'nemoclaw list' to see available sandboxes"
  echo; exit 1
fi

head_ "3. Application present"
if [[ "$(in_sandbox sh -c "test -f $APP/careshell/run.py && echo __ok__")" == *__ok__* ]]; then
  ok "$APP is populated"
else
  bad "$APP/careshell/run.py missing -- run ./deploy/deploy.sh $SANDBOX"
fi

head_ "4. Python environment"
DEPS="$(in_sandbox sh -c "$APP/.venv/bin/python -c 'import pydantic,yaml,requests,fastapi; print(\"ok\")'")"
if [[ "$DEPS" == *ok* ]]; then
  ok "venv has pydantic, pyyaml, requests, fastapi"
else
  bad "dependencies missing from $APP/.venv"
  note "the sandbox image is PEP 668 managed -- a venv is required, plain pip install fails"
fi

head_ "5. Inference reachable over https"
MODELS="$(TIMEOUT=60 in_sandbox sh -c 'curl -s --max-time 30 https://inference.local/v1/models')"
if [[ "$MODELS" == *'"id"'* ]]; then
  ok "https://inference.local responded"
  # Not data[0]: on an OpenAI-backed gateway that is an embedding model, which cannot
  # serve chat completions.
  MODEL_ID="$(printf '%s' "$MODELS" | python3 "$(dirname "${BASH_SOURCE[0]}")/pick_model.py" "" 2>/dev/null)"
  if [[ -n "$MODEL_ID" ]]; then
    note "chat model: $MODEL_ID  (pass with --model)"
  else
    warn "no chat-capable model in the list (embeddings only?)"
  fi
else
  bad "https://inference.local did not return a model list"
  note "if this broke after a policy change, the baseline was replaced instead of extended"
  note "re-apply: nemoclaw $SANDBOX policy-add --from-file openshell/careshell-preset.yaml"
fi

head_ "6. Plain http is refused (scheme matters)"
HTTP_CODE="$(in_sandbox sh -c 'curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://inference.local/v1/models')"
if [[ "$HTTP_CODE" == "403" ]]; then
  ok "http -> 403, https is the correct scheme"
else
  warn "http returned $HTTP_CODE (expected 403)"
fi

head_ "7. Egress denied"
EGRESS="$(in_sandbox sh -c 'curl -sS -I --max-time 10 https://example.com')"
if [[ "$EGRESS" == *"403"* || "$EGRESS" == *"(56)"* || "$EGRESS" == *"(7)"* ]]; then
  ok "outbound blocked"
  note "$(printf '%s' "$EGRESS" | head -1)"
  note "this is a userspace CONNECT-proxy refusal, not a kernel drop -- describe it accurately"
else
  bad "outbound traffic appears to be ALLOWED"
  note "$(printf '%s' "$EGRESS" | head -1)"
fi

head_ "8. Workspace writable (SQLite needs this)"
if [[ "$(in_sandbox sh -c "touch $APP/workspace/.preflight && rm -f $APP/workspace/.preflight && echo __ok__")" == *__ok__* ]]; then
  ok "$APP/workspace is writable"
else
  bad "$APP/workspace is not writable -- SQLite cannot open its database"
fi

head_ "9. Safety logic (no GPU, no model, no network)"
# No extra -q: pytest.ini already sets it, and -qq suppresses the "N passed" line.
TESTS="$(TIMEOUT=300 in_sandbox sh -c "cd $APP && PYTHONPATH=$APP $APP/.venv/bin/python -m pytest -p no:cacheprovider 2>&1 | tail -6")"
if [[ "$TESTS" == *"passed"* && "$TESTS" != *"failed"* && "$TESTS" != *" error"* ]]; then
  ok "test suite passes"
  note "$(printf '%s' "$TESTS" | grep -oE '[0-9]+ passed.*' | tail -1)"
else
  bad "tests did not pass"
  note "$(printf '%s' "$TESTS" | tail -2)"
fi

head_ "10. Nurse console"
if [[ "$(in_sandbox sh -c 'curl -sf --max-time 5 http://127.0.0.1:8000/healthz')" == *'"status"'* ]]; then
  ok "console answering inside the sandbox"
  if curl -sf --max-time 5 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
    ok "reachable from the host at http://127.0.0.1:8000"
  else
    warn "not forwarded to the host"
    note "openshell forward service --target-port 8000 --local 8000 $SANDBOX"
  fi
else
  warn "console not running"
  note "start it with ./deploy/deploy.sh $SANDBOX"
fi

printf '\n\033[1mSummary:\033[0m %d passed, %d failed, %d warnings\n' "$PASS" "$FAIL" "$WARN"
[[ "$FAIL" -eq 0 ]] || exit 1
