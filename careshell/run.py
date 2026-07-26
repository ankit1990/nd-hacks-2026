"""CareShell entrypoint: run a patient timeline through the care pipeline.

Three input modes:

    # batch -- as fast as possible, for tests and CI
    python -m careshell.run data/patients/P104.jsonl

    # stream -- paced replay of a complete file, for a self-contained demo
    python -m careshell.run data/patients/P104.jsonl --stream --speed 120

    # follow -- tail a live feed something else is appending to
    python -m careshell.run --follow data/live/P104.jsonl --idle-timeout 120
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from alerting.lan_broadcaster import LANBroadcaster
from alerting.tts_engine import LocalTTS
from careshell.care_plan import CarePlanError, load_care_plan, med_catalog
from careshell.extractor import DEFAULT_ENDPOINT, DEFAULT_MODEL, build_extractor
from careshell.reconciler import CareReconciler
from careshell.schemas import Decision, TimelineEntry
from careshell.store import CareStore, SchemaVersionError
from careshell.stream import build_checkpoints, follow_timeline, paced
from careshell.timeline import TimelineError, end_of_day, load_timeline

CODE_MARKS = {
    "MED_ON_TIME": "ok",
    "MED_OUT_OF_WINDOW": "!!",
    "DOUBLE_DOSE_BLOCKED": "!!",
    "MAX_DAILY_DOSES_BLOCKED": "!!",
    "MISSED_DOSE": "!!",
    "NIGHT_OUT_OF_BED": "!!",
    "NEEDS_HUMAN_REVIEW": "??",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="careshell", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("timeline", nargs="?", default=None,
                   help="patient timeline JSONL (batch/stream mode)")
    p.add_argument("--follow", metavar="PATH", default=None,
                   help="tail this file for appended entries instead of reading once")
    p.add_argument("--idle-timeout", type=float, default=None,
                   help="follow mode: stop after this many seconds with no new entry")
    p.add_argument("--from-start", action="store_true",
                   help="follow mode: also process entries already in the file")
    p.add_argument("--stream", action="store_true",
                   help="pace a complete file by its own timestamps")
    p.add_argument("--speed", type=float, default=120.0,
                   help="stream mode: virtual seconds per real second (default 120)")
    p.add_argument("--max-gap", type=float, default=4.0,
                   help="stream mode: cap on any single pause, real seconds (default 4)")

    p.add_argument("--plan", default="workspace/care_plan.yaml", help="care plan YAML")
    p.add_argument("--db", default="workspace/careshell.db", help="SQLite history database")
    p.add_argument("--memory", default="workspace/MEMORY.md",
                   help="human-readable narrative log ('' to disable)")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="vLLM chat completions URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model id served by vLLM")
    p.add_argument("--timeout", type=int, default=30, help="per-request timeout, seconds")
    p.add_argument("--console", default="http://127.0.0.1:8000/api/event",
                   help="nurse console ingress ('' to disable)")
    p.add_argument("--offline", action="store_true",
                   help="use the keyword extractor instead of the LLM")
    p.add_argument("--trust-keywords", action="store_true",
                   help="offline mode: let unambiguous phrasing record a dose (demo only)")
    p.add_argument("--no-tts", action="store_true", help="print spoken lines instead of speaking")
    return p


class Pipeline:
    """Wires extractor -> reconciler -> alert sinks, and reports each decision."""

    def __init__(self, plan, store, extractor, tts, lan, memory_path):
        self.plan = plan
        self.catalog = med_catalog(plan)
        self.extractor = extractor
        self.reconciler = CareReconciler(plan, store, memory_path=memory_path)
        self.tts = tts
        self.lan = lan
        self.alerts = 0

    def on_entry(self, entry: TimelineEntry) -> None:
        event = self.extractor.extract(entry, self.catalog)
        print(
            f"{entry.ts:%m-%d %H:%M}  {event.kind:<18} "
            f"{event.certainty:<10} {event.summary or entry.text[:80]}"
        )
        self._report(self.reconciler.process(entry, event))

    def on_tick(self, ts: datetime) -> None:
        self._report(self.reconciler.tick_and_emit(ts))

    def flush(self, through: datetime) -> None:
        self._report(self.reconciler.flush(through))

    def _report(self, decisions: list[Decision]) -> None:
        for d in decisions:
            mark = CODE_MARKS.get(d.code, "  ")
            print(f"          {mark} {d.code}: {d.message}")
            if self.lan:
                self.lan.send(d.model_dump(mode="json"))
            if d.speak:
                self.tts.speak(d.message)
            self.alerts += 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.follow and args.timeline:
        print("[fatal] pass either a timeline path or --follow, not both", file=sys.stderr)
        return 2
    if args.follow and args.stream:
        print("[fatal] --stream and --follow are mutually exclusive", file=sys.stderr)
        return 2
    source = args.follow or args.timeline or "data/patients/P104.jsonl"

    try:
        plan = load_care_plan(args.plan)
    except CarePlanError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    if args.trust_keywords and not args.offline:
        print("[fatal] --trust-keywords only applies with --offline", file=sys.stderr)
        return 2
    # Validate here rather than letting paced() raise mid-run: every other bad flag
    # combination exits 2 with a [fatal] line, and this one should too.
    if args.speed <= 0:
        print("[fatal] --speed must be positive", file=sys.stderr)
        return 2
    if args.max_gap < 0:
        print("[fatal] --max-gap cannot be negative", file=sys.stderr)
        return 2

    extractor = build_extractor(
        args.offline, args.endpoint, args.model, args.timeout, args.trust_keywords
    )
    tts = LocalTTS(enabled=not args.no_tts)
    lan = LANBroadcaster(args.console) if args.console else None
    if args.memory:
        Path(args.memory).parent.mkdir(parents=True, exist_ok=True)

    mode = "follow" if args.follow else ("stream" if args.stream else "batch")
    print(f"[careshell] patient {plan['patient_id']} ({plan['display_name']})")
    print(f"[careshell] mode: {mode}  source: {source}")
    print(f"[careshell] extractor: {'keyword (offline)' if args.offline else args.model}\n")

    try:
        store = CareStore(args.db)
    except SchemaVersionError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    with store:
        pipeline = Pipeline(plan, store, extractor, tts, lan, args.memory or None)
        try:
            if args.follow:
                rc = _run_follow(pipeline, plan, args)
            else:
                rc = _run_file(pipeline, plan, source, args)
        except KeyboardInterrupt:
            print("\n[careshell] interrupted.")
            rc = 0

        print(f"\n[careshell] {pipeline.alerts} decision(s) emitted this run.")
        _print_adherence(store, plan)
    return rc


def _run_file(pipeline: Pipeline, plan: dict, source: str, args) -> int:
    try:
        entries = load_timeline(source, patient_id=plan["patient_id"])
    except TimelineError as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        return 2

    print(f"[careshell] {len(entries)} timeline entries loaded\n")

    if not args.stream:
        for entry in entries:
            pipeline.on_entry(entry)
        pipeline.flush(end_of_day(entries[-1].ts))
        return 0

    # Paced replay: walk entries merged with window-close moments.
    checkpoints = build_checkpoints(entries, plan)
    for cp in paced(checkpoints, speed=args.speed, max_gap_seconds=args.max_gap):
        if cp.is_event:
            pipeline.on_entry(cp.entry)
        else:
            pipeline.on_tick(cp.ts)
    return 0


def _run_follow(pipeline: Pipeline, plan: dict, args) -> int:
    print("[careshell] following for new entries; Ctrl-C to stop\n")
    last: datetime | None = None
    for entry in follow_timeline(
        args.follow,
        patient_id=plan["patient_id"],
        from_start=args.from_start,
        stop_after_idle=args.idle_timeout,
    ):
        pipeline.on_entry(entry)
        last = entry.ts

    if last is not None:
        # The feed stopped. Close out the remaining windows on the final observed day.
        pipeline.flush(end_of_day(last))
    else:
        print("[careshell] no entries arrived before the idle timeout.")
    return 0


def _print_adherence(store: CareStore, plan: dict) -> None:
    rows = store.recent_adherence(plan["patient_id"], limit=10)
    if not rows:
        print("[careshell] no doses recorded.")
        return
    print("[careshell] recorded doses:")
    for row in rows:
        print(f"             {row['day']}  {row['med_id']}: {row['doses']}")


if __name__ == "__main__":
    raise SystemExit(main())
