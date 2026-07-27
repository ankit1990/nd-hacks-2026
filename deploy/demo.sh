#!/usr/bin/env bash
# Run the streaming CareShell demo inside a NemoClaw sandbox.
#
#     ./deploy/demo.sh [sandbox-name] [seconds-between-entries]
#
# Two processes inside the sandbox, sharing one file:
#   1. CareShell tailing data/live/P104.jsonl
#   2. tools/feed_timeline.py appending to it
#
# Decisions are pushed to the nurse console as they happen, so the page fills up live.
#
# Verified against nemoclaw v0.0.93 / openshell 0.0.85.

set -euo pipefail

SANDBOX="${1:-my-gpt-claw}"
DELAY="${2:-2}"                # real seconds between entries
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP=/sandbox/app
cd "$ROOT"

MODEL="$(cat .careshell-model 2>/dev/null || echo 'gpt-5.4-mini')"
ENTRIES="$(wc -l < data/patients/P104.jsonl | tr -d ' ')"
# Feed time plus per-entry inference latency, with headroom.
BUDGET=$(( ENTRIES * (DELAY + 12) + 90 ))

printf '\033[1m==> Streaming demo in %s (model: %s, %ss between entries)\033[0m\n' \
  "$SANDBOX" "$MODEL" "$DELAY"
printf '    Watch: http://127.0.0.1:8000\n\n'

nemoclaw "$SANDBOX" exec --workdir "$APP" --timeout "$BUDGET" -- sh -c "
set -e
mkdir -p $APP/data/live $APP/logs
: > $APP/data/live/P104.jsonl
rm -f $APP/workspace/careshell.db $APP/workspace/careshell.db-wal $APP/workspace/careshell.db-shm
: > $APP/workspace/MEMORY.md

# Reader. setsid with stdin detached, or it is reaped when the exec channel closes.
setsid env PYTHONPATH=$APP $APP/.venv/bin/python -m careshell.run \
  --follow $APP/data/live/P104.jsonl --from-start --idle-timeout 30 \
  --db $APP/workspace/careshell.db --memory $APP/workspace/MEMORY.md \
  --console http://127.0.0.1:8000/api/event --no-tts \
  --model '$MODEL' --timeout 120 \
  < /dev/null > $APP/logs/reader.log 2>&1 &

sleep 3
python3 tools/feed_timeline.py data/patients/P104.jsonl data/live/P104.jsonl --fixed-delay $DELAY

echo
echo '--- waiting for the reader to drain ---'
for i in \$(seq 1 40); do
  pgrep -f 'careshell.run' > /dev/null || break
  sleep 2
done

echo
grep -E '^[0-9]|^ +(ok|!!|\\?\\?)|decision\(s\)|recorded doses|^ +[0-9]{4}-' $APP/logs/reader.log | tail -45
"

printf '\n\033[1m==> Console history\033[0m\n'
curl -s --max-time 10 "http://127.0.0.1:8000/api/history?patient_id=P104&limit=20" \
  | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('  (console not reachable on the host; forward :8000 to see it)'); raise SystemExit
for x in reversed(d['decisions']):
    print(f\"  {x['ts'][:16]}  {x['code']:<24} {x['message']}\")
" 2>/dev/null || echo "  (console not reachable on the host)"
