#!/usr/bin/env bash
# Live streaming demo: one patient observation every N seconds, decisions appearing on
# the nurse console in real time.
#
#     ./deploy/demo.sh [sandbox-name] [seconds-between-entries]
#
# Default 2s x 17 entries = ~34 seconds of streaming.
#
# Open http://127.0.0.1:8000 BEFORE running this. Rows appear as they are decided.
#
# Two processes inside the sandbox sharing one file:
#   feeder  tools/feed_timeline.py appends a line every N seconds   (detached)
#   reader  careshell.run --follow tails it and decides             (foreground)
#
# The reader runs in the foreground so its classifications and alerts stream to this
# terminal live -- that narration is the demo. The dashboard updates from the same
# decisions over its WebSocket.
#
# Per-entry inference latency is ~1.1s against gpt-5.4-mini, so a 2s cadence keeps the
# reader ahead of the feeder. Drop below ~1.5s and it will start lagging.

set -euo pipefail

SANDBOX="${1:-my-gpt-claw}"
DELAY="${2:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP=/sandbox/app
cd "$ROOT"

MODEL="$(cat .careshell-model 2>/dev/null || echo 'gpt-5.4-mini')"
ENTRIES="$(grep -c . data/patients/P104.jsonl)"
RUNTIME=$(( ENTRIES * DELAY ))
BUDGET=$(( RUNTIME + 120 ))

printf '\033[1m==> CareShell live demo\033[0m\n'
printf '    sandbox %s   model %s\n' "$SANDBOX" "$MODEL"
printf '    %s entries, one every %ss  (~%ss of streaming)\n' "$ENTRIES" "$DELAY" "$RUNTIME"
printf '\n    \033[1mOpen http://127.0.0.1:8000 now\033[0m — rows appear as they are decided.\n'
printf '    Starting in 3s...\n\n'
sleep 3

nemoclaw "$SANDBOX" exec --workdir "$APP" --timeout "$BUDGET" -- sh -c "
set -e
mkdir -p $APP/data/live $APP/logs
: > $APP/data/live/P104.jsonl

# Clear history with SQL rather than deleting the file: the console holds the same
# database open, and unlinking it out from under a live reader invites WAL confusion.
$APP/.venv/bin/python - <<'PY'
import sqlite3, os
db = '$APP/workspace/careshell.db'
if os.path.exists(db):
    c = sqlite3.connect(db)
    for t in ('decisions', 'doses', 'observations'):
        try: c.execute(f'DELETE FROM {t}')
        except sqlite3.OperationalError: pass
    c.commit(); c.close()
    print('    history cleared')
PY
: > $APP/workspace/MEMORY.md

# Feeder detached: it just appends, its output is not the interesting part.
setsid $APP/.venv/bin/python tools/feed_timeline.py \
  data/patients/P104.jsonl data/live/P104.jsonl --fixed-delay $DELAY \
  < /dev/null > $APP/logs/feeder.log 2>&1 &

# Reader in the foreground so the terminal narrates the demo as it happens.
PYTHONPATH=$APP exec $APP/.venv/bin/python -u -m careshell.run \
  --follow $APP/data/live/P104.jsonl --from-start --idle-timeout 12 \
  --db $APP/workspace/careshell.db --memory $APP/workspace/MEMORY.md \
  --console http://127.0.0.1:8000/api/event --no-tts \
  --model '$MODEL' --timeout 60
"

printf '\n\033[1m==> Console history (what the nurse sees)\033[0m\n'
curl -s --max-time 10 "http://127.0.0.1:8000/api/history?patient_id=P104&limit=30" \
  | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('  (console not reachable on the host)'); raise SystemExit
for x in reversed(d['decisions']):
    print(f\"  {x['ts'][:16]}  {x['code']:<24} {x['message']}\")
" 2>/dev/null || echo "  (console not reachable on the host)"
