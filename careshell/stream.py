"""Streaming input for CareShell.

Three ways the same pipeline consumes a timeline:

  batch    -- read the whole file, process as fast as possible. Best for tests and CI.
  stream   -- paced replay of a complete file. Wall-clock gaps between entries mirror
              the timeline's own gaps, divided by --speed. This is the demo mode: a
              MISSED_DOSE fires at the dramatic moment the window closes, not lumped
              into whatever event happens next.
  follow   -- tail a file that something else is appending to. This is real ingest:
              a sensor bridge or nurse tablet writes JSONL, CareShell reacts live.

Why `stream` needs checkpoints rather than just sleeping between entries: the reconciler
is time-driven as well as event-driven. A medication window closing at 08:30 with no dose
must surface at 08:30 of virtual time. So we merge entry timestamps with window-close
timestamps into one ordered list of checkpoints and walk that.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from careshell.care_plan import parse_hhmm
from careshell.schemas import TimelineEntry


@dataclass
class Checkpoint:
    """A moment in virtual time. `entry` is None for pure time-advance checkpoints."""

    ts: datetime
    entry: TimelineEntry | None = None

    @property
    def is_event(self) -> bool:
        return self.entry is not None


def window_closes(plan: dict, start: datetime, end: datetime) -> list[datetime]:
    """Every medication window-close moment in (start, end], across all days spanned."""
    out: list[datetime] = []
    day = start.date()
    while day <= end.date():
        for med in plan["medications"]:
            close = datetime.combine(day, parse_hhmm(med["window_end"]))
            if start < close <= end:
                out.append(close)
        day += timedelta(days=1)
    return sorted(set(out))


def build_checkpoints(entries: list[TimelineEntry], plan: dict) -> list[Checkpoint]:
    """Merge entries with window-close moments into one ordered walk."""
    if not entries:
        return []
    # End of the final observed day, not +24h: advancing past midnight would fabricate
    # window closes for a day the timeline says nothing about.
    start = entries[0].ts
    end = entries[-1].ts.replace(hour=23, minute=59, second=59, microsecond=0)
    points = [Checkpoint(e.ts, e) for e in entries]
    points += [Checkpoint(ts) for ts in window_closes(plan, start, end)]
    # Events sort before bare ticks at the same instant: a dose taken exactly at window
    # close counts as taken, not missed.
    return sorted(points, key=lambda c: (c.ts, 0 if c.is_event else 1))


def paced(
    checkpoints: list[Checkpoint],
    speed: float = 60.0,
    max_gap_seconds: float = 4.0,
    sleeper=time.sleep,
) -> Iterator[Checkpoint]:
    """Yield checkpoints with wall-clock pacing derived from virtual-time gaps.

    speed=60 means one minute of timeline per second of real time. `max_gap_seconds`
    caps any single pause so an overnight jump does not stall the demo for minutes.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")

    previous: datetime | None = None
    for cp in checkpoints:
        if previous is not None:
            delay = (cp.ts - previous).total_seconds() / speed
            if delay > 0:
                sleeper(min(delay, max_gap_seconds))
        previous = cp.ts
        yield cp


def follow_timeline(
    path: str | Path,
    patient_id: str | None = None,
    poll_seconds: float = 0.5,
    from_start: bool = True,
    stop_after_idle: float | None = None,
) -> Iterator[TimelineEntry]:
    """Tail a JSONL timeline, yielding entries as they are appended.

    Blocks waiting for new lines. `stop_after_idle` ends the loop after that many
    seconds without a new entry (None = follow forever).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    seen: set[str] = set()
    idle = 0.0
    partial = ""

    with path.open() as f:
        if not from_start:
            f.seek(0, 2)   # jump to EOF: only react to what arrives from now on

        while True:
            chunk = f.readline()
            if not chunk:
                if stop_after_idle is not None and idle >= stop_after_idle:
                    return
                time.sleep(poll_seconds)
                idle += poll_seconds
                continue

            idle = 0.0
            # A writer may flush a partial line; hold it until the newline arrives.
            if not chunk.endswith("\n"):
                partial += chunk
                continue
            line, partial = partial + chunk, ""

            if not line.strip():
                continue
            try:
                entry = TimelineEntry(**json.loads(line))
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                print(f"[warn] {path}: skipping malformed line: {exc}")
                continue

            if entry.id in seen:
                continue
            if patient_id and entry.patient_id != patient_id:
                continue
            seen.add(entry.id)
            yield entry
