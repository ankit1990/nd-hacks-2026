"""Deterministic care-plan policy engine.

Nothing in this module calls a language model, opens a network socket, or reads the wall
clock. Every decision is a pure function of (care plan, stored history, current event,
current timeline time). That is what makes the safety logic unit-testable without a GPU
and makes a demo run reproduce exactly.

`now` always comes from the timeline entry's own timestamp. A recorded timeline replayed
at 3pm must still evaluate an 08:00 medication window as 08:00.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from careshell.care_plan import parse_hhmm
from careshell.schemas import CareEvent, Decision, TimelineEntry
from careshell.store import CareStore

MAX_MISSED_DOSE_BACKFILL_DAYS = 31


class CareReconciler:
    def __init__(self, plan: dict, store: CareStore, memory_path: str | None = None):
        self.plan = plan
        self.store = store
        self.patient_id = plan["patient_id"]
        self.memory_path = memory_path
        self._meds = {m["id"]: m for m in plan["medications"]}
        self._first_name = plan["display_name"].split()[0]

        # Per-run behavioural state. Batch replay only; see "Known Gaps" in spec.md.
        self._out_of_bed_since: datetime | None = None
        self._out_of_bed_alerted = False
        self._checked_through: datetime | None = None

    # --------------------------------------------------------------- public entry

    def process(self, entry: TimelineEntry, event: CareEvent) -> list[Decision]:
        """Advance the clock to `entry.ts`, then evaluate this observation."""
        if entry.patient_id != self.patient_id:
            raise ValueError(
                f"entry {entry.id} is for patient {entry.patient_id}, "
                f"but this reconciler is bound to {self.patient_id}"
            )

        now = entry.ts
        decisions = self.tick(now)

        if self.store.has_observation(entry.id):
            return self._emit(decisions)   # already processed; only time-based checks run

        if event.kind == "MED_INTAKE":
            decisions += self._handle_med(entry, event, now)
        elif event.kind == "BED_EXIT":
            # Only start the clock on a fresh exit. Sensor feeds repeat "still out of
            # bed" lines, and restarting the timer on each one would mean a long absence
            # never crosses the threshold.
            if self._out_of_bed_since is None:
                self._out_of_bed_since = now
                self._out_of_bed_alerted = False
        elif event.kind == "BED_RETURN":
            self._out_of_bed_since = None
            self._out_of_bed_alerted = False

        self.store.record_observation(entry, event)
        return self._emit(decisions)

    def tick(self, now: datetime) -> list[Decision]:
        """Time-driven checks: closed medication windows and prolonged night absence.

        Idempotent and safe to call repeatedly; it only considers the span since the
        last call.
        """
        since = self._checked_through
        self._checked_through = now
        if since is not None and now < since:
            return []   # out-of-order entry; timeline.py sorts, so this is defensive
        decisions = self._closed_windows(since, now)
        decisions += self._prolonged_out_of_bed(now)
        return decisions

    def tick_and_emit(self, now: datetime) -> list[Decision]:
        """Advance the clock with no observation attached, persisting what fires.

        Used by streaming replay, where a window closing at 08:30 should surface at
        08:30 of virtual time rather than waiting for the next observation.
        """
        return self._emit(self.tick(now))

    def flush(self, through: datetime) -> list[Decision]:
        """Run the final time-based checks after the last observation."""
        return self._emit(self.tick(through))

    # ------------------------------------------------------------ medication policy

    def _handle_med(self, entry: TimelineEntry, event: CareEvent, now: datetime) -> list[Decision]:
        # Guard 1: the model must be sure, and must name a real medication.
        # "Reached for the bottle" is not "took the pill".
        if event.certainty != "CONFIRMED" or event.med_id is None:
            return [self._decide(
                now, entry.id, "NEEDS_HUMAN_REVIEW", event.med_id,
                f"Possible medication event, not confirmed ({event.certainty.lower()}): "
                f"{event.summary or entry.text}",
            )]

        med = self._meds.get(event.med_id)
        if med is None:
            return [self._decide(
                now, entry.id, "NEEDS_HUMAN_REVIEW", event.med_id,
                f"Medication '{event.med_id}' is not in the care plan.",
            )]

        # Guard 2: per-medication lockout. Not one global timer -- that would block the
        # evening statin because of the morning beta blocker.
        lockout = timedelta(hours=float(med["lockout_hours"]))
        recent = self.store.doses_since(self.patient_id, med["id"], now - lockout)
        recent = [d for d in recent if d <= now]
        if recent:
            minutes = int((now - max(recent)).total_seconds() // 60)
            return [self._decide(
                now, entry.id, "DOUBLE_DOSE_BLOCKED", med["id"],
                f"{self._first_name}, you already took your {med['name']} "
                f"{self._humanise(minutes)} ago. Please do not take another.",
                speak=True,
            )]

        # Guard 3: daily cap.
        taken_today = self.store.dose_count_on(self.patient_id, med["id"], now.date())
        if taken_today >= med["max_daily_doses"]:
            return [self._decide(
                now, entry.id, "MAX_DAILY_DOSES_BLOCKED", med["id"],
                f"{self._first_name}, the daily limit for {med['name']} "
                f"({med['max_daily_doses']}) has already been reached today.",
                speak=True,
            )]

        # Cleared. Record the dose before emitting, so a crash cannot lose it.
        self.store.record_dose(entry.id, self.patient_id, med["id"], now)

        if self._within(now, med["window_start"], med["window_end"]):
            return [self._decide(
                now, entry.id, "MED_ON_TIME", med["id"],
                f"{med['name']} recorded. Thank you, {self._first_name}.",
                speak=True,
            )]
        return [self._decide(
            now, entry.id, "MED_OUT_OF_WINDOW", med["id"],
            f"{med['name']} taken at {now:%H:%M}, outside the scheduled "
            f"{med['window_start']}-{med['window_end']} window.",
        )]

    # ---------------------------------------------------------------- missed doses

    def _closed_windows(self, since: datetime | None, now: datetime) -> list[Decision]:
        """Fire MISSED_DOSE for any window that closed in (since, now] with no dose.

        A window closing empty is the highest-value eldercare signal, and the v1 spec had
        no code path that could ever produce it.
        """
        if since is None:
            return []   # first event of the run establishes the baseline, nothing to backfill

        out: list[Decision] = []
        for day in self._days_between(since.date(), now.date()):
            for med in self.plan["medications"]:
                close = datetime.combine(day, parse_hhmm(med["window_end"]))
                if not (since < close <= now):
                    continue
                if self.store.dose_count_on(self.patient_id, med["id"], day) > 0:
                    continue
                out.append(self._decide(
                    close,
                    f"missed:{self.patient_id}:{med['id']}:{day.isoformat()}",
                    "MISSED_DOSE",
                    med["id"],
                    f"{med['name']} window closed at {med['window_end']} on "
                    f"{day:%d %b} with no dose recorded.",
                ))
        return out

    @staticmethod
    def _days_between(start: date, end: date) -> list[date]:
        """Inclusive list of dates, capped so a stale timeline cannot emit thousands."""
        span = (end - start).days
        if span < 0:
            return []
        if span > MAX_MISSED_DOSE_BACKFILL_DAYS:
            start = end - timedelta(days=MAX_MISSED_DOSE_BACKFILL_DAYS)
            span = MAX_MISSED_DOSE_BACKFILL_DAYS
        return [start + timedelta(days=i) for i in range(span + 1)]

    # -------------------------------------------------------------------- behaviour

    def _prolonged_out_of_bed(self, now: datetime) -> list[Decision]:
        """Alert once per episode when a night-time absence exceeds the plan threshold."""
        if self._out_of_bed_since is None or self._out_of_bed_alerted:
            return []

        behavior = self.plan["behavior"]
        if not self._within(now, behavior["night_start"], behavior["night_end"]):
            return []

        elapsed = (now - self._out_of_bed_since).total_seconds() / 60
        if elapsed < behavior["max_out_of_bed_minutes_at_night"]:
            return []

        self._out_of_bed_alerted = True
        return [self._decide(
            now,
            f"night:{self.patient_id}:{self._out_of_bed_since.isoformat()}",
            "NIGHT_OUT_OF_BED",
            None,
            f"Out of bed since {self._out_of_bed_since:%H:%M} "
            f"({int(elapsed)} minutes) during sleep hours.",
        )]

    # ---------------------------------------------------------------------- helpers

    def _decide(
        self,
        ts: datetime,
        entry_id: str,
        code: str,
        med_id: str | None,
        message: str,
        speak: bool = False,
    ) -> Decision:
        return Decision(
            ts=ts,
            entry_id=entry_id,
            patient_id=self.patient_id,
            code=code,
            med_id=med_id,
            message=message,
            speak=speak,
        )

    def _emit(self, decisions: list[Decision]) -> list[Decision]:
        """Persist decisions, dropping any that were already emitted on a prior run."""
        fresh = [d for d in decisions if self.store.record_decision(d)]
        for d in fresh:
            self._log_memory(d)
        return fresh

    @staticmethod
    def _within(now: datetime, start: str, end: str) -> bool:
        s, e, t = parse_hhmm(start), parse_hhmm(end), now.time()
        return s <= t <= e if s <= e else (t >= s or t <= e)   # handles overnight spans

    @staticmethod
    def _humanise(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        hours, mins = divmod(minutes, 60)
        if mins == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{hours} hour{'s' if hours != 1 else ''} and {mins} minute{'s' if mins != 1 else ''}"

    def _log_memory(self, d: Decision) -> None:
        """Human-readable narrative. SQLite is authoritative; this is for people."""
        if not self.memory_path:
            return
        with open(self.memory_path, "a") as f:
            f.write(f"- `{d.ts:%Y-%m-%d %H:%M}` **{d.code}** — {d.message}\n")
