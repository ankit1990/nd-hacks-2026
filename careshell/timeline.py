"""Reader for a per-patient timeline file.

Input contract (one JSON object per line):

    {"id":"e001","ts":"2026-03-14T07:12:04","patient_id":"P104",
     "source":"sensor","text":"Bed exit detected."}

Malformed lines are reported and skipped rather than aborting the run: one bad line from
an upstream sensor should not take the bedside safety loop offline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from careshell.schemas import TimelineEntry


class TimelineError(ValueError):
    """Raised only for problems that make the whole file unusable."""


def end_of_day(ts: datetime) -> datetime:
    """The last instant of `ts`'s calendar day.

    Used to close out medication windows after the final observation. Deliberately not
    `ts + 1 day`: advancing past midnight would fabricate MISSED_DOSE alerts for a day
    the timeline says nothing about. Silence after the last observation is missing data,
    not a missed dose.
    """
    return ts.replace(hour=23, minute=59, second=59, microsecond=0)


def parse_entry(
    raw: str,
    seen: set[str],
    patient_id: str | None = None,
    where: str = "",
) -> TimelineEntry | None:
    """Parse one JSONL line into a TimelineEntry, or None if it should be skipped.

    Shared by the batch reader and the live tail so the two cannot drift on what counts
    as a valid entry. Mutates `seen` with the accepted id. Never raises: one bad line
    from an upstream sensor must not take the bedside loop offline.
    """
    if not raw.strip():
        return None

    prefix = f"[warn] {where}: " if where else "[warn] "
    try:
        entry = TimelineEntry(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        print(f"{prefix}skipping malformed line: {exc}")
        return None

    if entry.id in seen:
        print(f"{prefix}duplicate entry id {entry.id!r}, skipping")
        return None
    if patient_id and entry.patient_id != patient_id:
        print(
            f"{prefix}entry {entry.id!r} is for {entry.patient_id!r}, "
            f"expected {patient_id!r}, skipping"
        )
        return None

    seen.add(entry.id)
    return entry


def load_timeline(path: str | Path, patient_id: str | None = None) -> list[TimelineEntry]:
    """Parse, de-duplicate and sort a timeline file.

    Returns entries ordered by timestamp. Duplicate ids keep the first occurrence.
    """
    path = Path(path)
    if not path.exists():
        raise TimelineError(f"timeline not found: {path}")

    entries: list[TimelineEntry] = []
    seen: set[str] = set()

    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            entry = parse_entry(raw, seen, patient_id, where=f"{path}:{lineno}")
            if entry is not None:
                entries.append(entry)

    if not entries:
        raise TimelineError(f"{path}: no usable entries")

    # Sort by timestamp; ties keep file order, which is the only signal we have.
    # TimelineEntry normalises tz-aware timestamps to naive, so this cannot raise.
    return sorted(entries, key=lambda e: e.ts)
