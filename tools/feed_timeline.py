"""Push a source timeline into the live feed CareShell is tailing.

This is the demo driver. CareShell runs inside the NemoClaw sandbox with
`--follow data/live/P104.jsonl`; this script appends to that file line by line, so the
nurse console fills up in front of the audience instead of appearing all at once.

    # terminal 1 (inside the sandbox)
    nemoclaw careshell exec -- python -m careshell.run \
        --follow data/live/P104.jsonl --idle-timeout 120

    # terminal 2 (host, writing into the same bind-mounted directory)
    python tools/feed_timeline.py data/patients/P104.jsonl data/live/P104.jsonl --speed 120

Pacing mirrors the gaps in the source timeline's own timestamps, divided by --speed, so
the demo keeps the real rhythm of a day without taking a day.

Deliberately dependency-free (stdlib only): it is also useful from a plain shell on the
host, outside any Python environment the sandbox has.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def read_source(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[feed] {path}:{lineno}: skipping malformed line: {exc}", file=sys.stderr)
                continue
            if "ts" not in row or "id" not in row:
                print(f"[feed] {path}:{lineno}: missing 'id' or 'ts', skipping", file=sys.stderr)
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["ts"])
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="complete timeline to read from")
    p.add_argument("target", help="live feed file CareShell is following")
    p.add_argument("--speed", type=float, default=120.0,
                   help="virtual seconds per real second (default 120)")
    p.add_argument("--max-gap", type=float, default=5.0,
                   help="cap on any single pause, in real seconds (default 5)")
    p.add_argument("--fixed-delay", type=float, default=None,
                   help="ignore timestamps, wait this many seconds between lines")
    p.add_argument("--truncate", action="store_true",
                   help="clear the target file before feeding (fresh demo run)")
    args = p.parse_args(argv)

    source, target = Path(args.source), Path(args.target)
    if not source.exists():
        print(f"[feed] source not found: {source}", file=sys.stderr)
        return 2
    if args.speed <= 0:
        print("[feed] --speed must be positive", file=sys.stderr)
        return 2

    rows = read_source(source)
    if not rows:
        print(f"[feed] no usable entries in {source}", file=sys.stderr)
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    if args.truncate:
        target.write_text("")
        print(f"[feed] truncated {target}")

    print(f"[feed] {len(rows)} entries -> {target} "
          f"({'fixed ' + str(args.fixed_delay) + 's' if args.fixed_delay else f'{args.speed}x'})")

    previous: datetime | None = None
    for i, row in enumerate(rows, start=1):
        if args.fixed_delay is not None:
            if previous is not None:
                time.sleep(args.fixed_delay)
            previous = datetime.now()
        else:
            ts = datetime.fromisoformat(row["ts"])
            if previous is not None:
                delay = (ts - previous).total_seconds() / args.speed
                if delay > 0:
                    time.sleep(min(delay, args.max_gap))
            previous = ts

        # Append and flush immediately: the reader is polling for whole lines.
        with target.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()

        print(f"[feed] {i:>3}/{len(rows)}  {row['ts']}  {row.get('text', '')[:70]}")

    print("[feed] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
