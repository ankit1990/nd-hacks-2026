#!/usr/bin/env bash
# Run the streaming CareShell demo.
#
#     ./deploy/demo.sh [sandbox-name] [speed]
#
# Two processes:
#   1. CareShell inside the sandbox, tailing data/live/P104.jsonl
#   2. tools/feed_timeline.py on the host, appending to that same file over time
#
# Both see the same file through the rw bind mount declared in the policy delta, so the
# nurse console fills up live in front of the audience.

set -euo pipefail

SANDBOX="${1:-careshell}"
SPEED="${2:-240}"                     # virtual seconds per real second
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="data/patients/P104.jsonl"
LIVE="data/live/P104.jsonl"
MODEL="$(cat .careshell-model 2>/dev/null || echo 'nvidia/Nemotron-Nano-12B-v2')"

mkdir -p data/live logs
: > "$LIVE"                            # fresh feed for a clean demo

cleanup() {
  [[ -n "${READER_PID:-}" ]] && kill "$READER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf '\033[1m==> Starting CareShell in follow mode (model: %s)\033[0m\n' "$MODEL"
nemoclaw "$SANDBOX" exec -- python -m careshell.run \
  --follow "$LIVE" \
  --from-start \
  --idle-timeout 30 \
  --model "$MODEL" \
  --no-tts &
READER_PID=$!

sleep 3

printf '\n\033[1m==> Feeding the timeline at %sx\033[0m\n' "$SPEED"
python3 tools/feed_timeline.py "$SOURCE" "$LIVE" --speed "$SPEED" --max-gap 4

printf '\n\033[1m==> Feed complete. Waiting for CareShell to drain.\033[0m\n'
wait "$READER_PID" 2>/dev/null || true
READER_PID=""

printf '\n\033[1m==> Recorded history\033[0m\n'
nemoclaw "$SANDBOX" exec -- python3 -c "
from careshell.store import CareStore
with CareStore('workspace/careshell.db') as s:
    for d in reversed(s.recent_decisions('P104', 50)):
        print(f\"  {d['ts'][:16]}  {d['code']:<24} {d['message']}\")
"
