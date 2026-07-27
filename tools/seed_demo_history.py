"""Seed the nurse console with realistic prior history.

A console showing three rows looks like a prototype. A console showing two weeks of
medication history with a handful of genuine incidents looks like software a ward would
actually run. This generates that backdrop so the live demo lands on top of it.

    python tools/seed_demo_history.py --days 14
    python tools/seed_demo_history.py --days 14 --db workspace/careshell.db --clear

Deterministic: the same --days and --seed produce the same history every time, so a demo
can be reset and re-run without the numbers moving around.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careshell.care_plan import load_care_plan          # noqa: E402
from careshell.schemas import Decision                  # noqa: E402
from careshell.store import CareStore                   # noqa: E402

# Weighted so a normal day is mostly uneventful. Real wards are boring; the incidents
# stand out precisely because they are rare.
INCIDENT_WEIGHTS = [
    ("none", 70),
    ("late", 10),
    ("missed", 8),
    ("double", 5),
    ("night", 5),
    ("review", 2),
]

REVIEW_NOTES = [
    "reached for the bottle and put it back",
    "asked whether the morning pill had been taken",
    "held the tablet but no ingestion observed",
    "nurse note ambiguous: 'gave him his usual'",
    "bottle moved on the nightstand, no intake seen",
]

NIGHT_NOTES = [
    "Out of bed for {m} minutes during sleep hours (since {t}).",
]


def pick(rng: random.Random) -> str:
    total = sum(w for _, w in INCIDENT_WEIGHTS)
    roll = rng.uniform(0, total)
    upto = 0.0
    for kind, weight in INCIDENT_WEIGHTS:
        upto += weight
        if roll <= upto:
            return kind
    return "none"


def hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def build(plan: dict, days: int, seed: int, end: date) -> list[tuple[Decision, str | None]]:
    """Return (decision, dose_med_id) pairs, oldest first."""
    rng = random.Random(seed)
    patient = plan["patient_id"]
    first_name = plan["display_name"].split()[0]
    out: list[tuple[Decision, str | None]] = []

    for offset in range(days, 0, -1):
        day = end - timedelta(days=offset)
        incident = pick(rng)

        for med in plan["medications"]:
            h, m = hhmm(med["scheduled"])
            drift = rng.randint(-18, 18)
            taken = datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m)
            taken += timedelta(minutes=drift)
            eid = f"seed:{day}:{med['id']}"

            if incident == "missed" and med is plan["medications"][0]:
                close_h, close_m = hhmm(med["window_end"])
                close = datetime.combine(day, datetime.min.time()).replace(
                    hour=close_h, minute=close_m
                )
                out.append((Decision(
                    ts=close, entry_id=f"{eid}:missed", patient_id=patient,
                    code="MISSED_DOSE", med_id=med["id"],
                    message=(f"{med['name']} window closed at {med['window_end']} on "
                             f"{day:%d %b} with no dose recorded."),
                ), None))
                continue

            if incident == "late" and med is plan["medications"][0]:
                taken += timedelta(minutes=rng.randint(45, 110))
                out.append((Decision(
                    ts=taken, entry_id=eid, patient_id=patient,
                    code="MED_OUT_OF_WINDOW", med_id=med["id"],
                    message=(f"{med['name']} taken at {taken:%H:%M}, outside the scheduled "
                             f"{med['window_start']}-{med['window_end']} window."),
                ), med["id"]))
                continue

            out.append((Decision(
                ts=taken, entry_id=eid, patient_id=patient,
                code="MED_ON_TIME", med_id=med["id"],
                message=f"{med['name']} recorded. Thank you, {first_name}.",
                speak=True,
            ), med["id"]))

            if incident == "double" and med is plan["medications"][0]:
                again = taken + timedelta(minutes=rng.randint(20, 50))
                mins = int((again - taken).total_seconds() // 60)
                out.append((Decision(
                    ts=again, entry_id=f"{eid}:double", patient_id=patient,
                    code="DOUBLE_DOSE_BLOCKED", med_id=med["id"],
                    message=(f"{first_name}, you already took your {med['name']} "
                             f"{mins} minutes ago. Please do not take another."),
                    speak=True,
                ), None))

        if incident == "night":
            start = datetime.combine(day, datetime.min.time()).replace(
                hour=rng.choice([0, 1, 2, 3]), minute=rng.randint(0, 59)
            )
            mins = rng.randint(18, 55)
            out.append((Decision(
                ts=start + timedelta(minutes=mins),
                entry_id=f"seed:{day}:night", patient_id=patient,
                code="NIGHT_OUT_OF_BED", med_id=None,
                message=NIGHT_NOTES[0].format(m=mins, t=f"{start:%H:%M}"),
            ), None))

        if incident == "review":
            when = datetime.combine(day, datetime.min.time()).replace(
                hour=rng.randint(9, 18), minute=rng.randint(0, 59)
            )
            out.append((Decision(
                ts=when, entry_id=f"seed:{day}:review", patient_id=patient,
                code="NEEDS_HUMAN_REVIEW", med_id=None,
                message=(f"Possible medication event, not confirmed (likely): "
                         f"{rng.choice(REVIEW_NOTES)}"),
            ), None))

    out.sort(key=lambda pair: pair[0].ts)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="workspace/careshell.db")
    p.add_argument("--plan", default="workspace/care_plan.yaml")
    p.add_argument("--days", type=int, default=14, help="days of history before today")
    p.add_argument("--seed", type=int, default=104, help="rng seed; same seed = same history")
    p.add_argument("--end", default=None, help="last day of seeded history, YYYY-MM-DD")
    p.add_argument("--clear", action="store_true", help="wipe existing rows first")
    args = p.parse_args(argv)

    plan = load_care_plan(args.plan)
    # Default to the day before the demo timeline so seeded history reads as "the two
    # weeks leading up to it" rather than overlapping the live run.
    end = date.fromisoformat(args.end) if args.end else date(2026, 3, 14)

    with CareStore(args.db) as store:
        if args.clear:
            with store.transaction():
                for table in ("decisions", "doses", "observations"):
                    store.conn.execute(f"DELETE FROM {table}")
            print("cleared existing history")

        rows = build(plan, args.days, args.seed, end)
        with store.transaction():
            for decision, dose_med in rows:
                store.record_decision(decision)
                if dose_med:
                    store.record_dose(
                        decision.entry_id, decision.patient_id, dose_med, decision.ts
                    )

    counts: dict[str, int] = {}
    for decision, _ in rows:
        counts[decision.code] = counts.get(decision.code, 0) + 1
    print(f"seeded {len(rows)} decisions across {args.days} days ending {end}")
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {code:<24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
