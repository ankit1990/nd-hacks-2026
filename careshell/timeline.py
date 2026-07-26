"""Reader for a per-patient timeline file.

Input contract (one JSON object per line):

    {"id":"e001","ts":"2026-03-14T07:12:04","patient_id":"P104",
     "source":"sensor","text":"Bed exit detected."}

Malformed lines are reported and skipped rather than aborting the run: one bad line from
an upstream sensor should not take the bedside safety loop offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from careshell.schemas import TimelineEntry


class TimelineError(ValueError):
    """Raised only for problems that make the whole file unusable."""


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
            if not raw.strip():
                continue
            try:
                entry = TimelineEntry(**json.loads(raw))
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                print(f"[warn] {path}:{lineno}: skipping malformed line: {exc}")
                continue

            if entry.id in seen:
                print(f"[warn] {path}:{lineno}: duplicate entry id {entry.id!r}, skipping")
                continue
            if patient_id and entry.patient_id != patient_id:
                print(
                    f"[warn] {path}:{lineno}: entry {entry.id!r} is for "
                    f"{entry.patient_id!r}, expected {patient_id!r}, skipping"
                )
                continue

            seen.add(entry.id)
            entries.append(entry)

    if not entries:
        raise TimelineError(f"{path}: no usable entries")

    # Sort by timestamp; ties keep file order, which is the only signal we have.
    return sorted(entries, key=lambda e: e.ts)
