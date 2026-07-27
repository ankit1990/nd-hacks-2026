"""Safety-logic tests. No GPU, no model, no network -- pure functions over fixtures.

If only one thing works on demo day, make it this file.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from careshell.reconciler import MAX_MISSED_DOSE_BACKFILL_DAYS, CareReconciler
from tests.conftest import PLAN, bed, codes, entry, intake


# ------------------------------------------------------------------ happy path

def test_dose_inside_window_is_recorded_on_time(reconciler, store):
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert codes(out) == ["MED_ON_TIME"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_on_time_message_names_the_real_medication(reconciler):
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert "Beta Blocker" in out[0].message
    # Derived from the plan, not hardcoded: the patient's name lives in care_plan.yaml.
    assert PLAN["display_name"].split()[0] in out[0].message
    assert out[0].speak is True


def test_dose_outside_window_is_recorded_but_flagged(reconciler, store):
    out = reconciler.process(entry("e1", "2026-03-14T11:00:00"), intake())
    assert codes(out) == ["MED_OUT_OF_WINDOW"]
    # Still a real dose -- it must count against the lockout.
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


# ------------------------------------------------------------- double dosing

def test_second_dose_inside_lockout_is_blocked(reconciler, store):
    reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    out = reconciler.process(entry("e2", "2026-03-14T08:31:00"), intake())
    assert codes(out) == ["DOUBLE_DOSE_BLOCKED"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_block_message_reports_real_elapsed_time(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    out = reconciler.process(entry("e2", "2026-03-14T08:31:00"), intake())
    assert "33 minutes" in out[0].message
    assert "Beta Blocker" in out[0].message
    assert out[0].speak is True


def test_lockout_is_per_medication_not_global(reconciler, store):
    """The v1 bug: one global 4h timer blocked unrelated medications."""
    reconciler.process(entry("e1", "2026-03-14T19:52:00"), intake("MED_EVENING"))
    out = reconciler.process(entry("e2", "2026-03-14T20:10:00"), intake("MED_MORNING"))
    assert "DOUBLE_DOSE_BLOCKED" not in codes(out)
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_dose_after_lockout_expires_is_allowed(reconciler):
    reconciler.process(entry("e1", "2026-03-13T07:58:00"), intake())
    out = reconciler.process(entry("e2", "2026-03-14T07:58:00"), intake())
    # The intervening evening window also closes empty, which is correct; assert on the
    # medication under test rather than the whole batch.
    assert "MED_ON_TIME" in codes(out)
    assert "DOUBLE_DOSE_BLOCKED" not in codes(out)


def test_daily_cap_blocks_after_lockout_expires_same_day(plan, store):
    plan["medications"][0]["lockout_hours"] = 1
    r = CareReconciler(plan, store, memory_path=None)
    r.process(entry("e1", "2026-03-14T07:58:00"), intake())
    out = r.process(entry("e2", "2026-03-14T14:00:00"), intake())
    assert codes(out) == ["MAX_DAILY_DOSES_BLOCKED"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


# -------------------------------------------------------- uncertainty guard

@pytest.mark.parametrize("certainty", ["LIKELY", "UNCLEAR"])
def test_unconfirmed_intake_never_records_a_dose(reconciler, store, certainty):
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake(certainty=certainty))
    assert codes(out) == ["NEEDS_HUMAN_REVIEW"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 0


def test_intake_without_med_id_needs_review(reconciler, store):
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake(med_id=None))
    assert codes(out) == ["NEEDS_HUMAN_REVIEW"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 0


def test_unknown_med_id_needs_review(reconciler, store):
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake(med_id="MED_GHOST"))
    assert codes(out) == ["NEEDS_HUMAN_REVIEW"]
    assert store.dose_count_on("P104", "MED_GHOST", date(2026, 3, 14)) == 0


def test_unconfirmed_then_confirmed_records_exactly_one_dose(reconciler, store):
    """The real demo sequence: reaches for the bottle, then actually takes it."""
    reconciler.process(entry("e1", "2026-03-14T07:44:00"), intake(certainty="LIKELY"))
    out = reconciler.process(entry("e2", "2026-03-14T07:58:00"), intake())
    assert codes(out) == ["MED_ON_TIME"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


# ------------------------------------------------------------- missed doses

def test_window_closing_empty_fires_missed_dose(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    missed = [d for d in out if d.code == "MISSED_DOSE"]
    assert len(missed) == 1
    assert missed[0].med_id == "MED_MORNING"


def test_missed_dose_is_stamped_at_window_close_not_discovery(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    missed = next(d for d in out if d.code == "MISSED_DOSE")
    assert missed.ts == datetime(2026, 3, 14, 8, 30)


def test_no_missed_dose_when_medication_was_taken(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    reconciler.process(entry("e2", "2026-03-14T07:58:00"), intake())
    out = reconciler.process(entry("e3", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    assert "MISSED_DOSE" not in codes(out)


def test_missed_dose_fires_once_across_repeated_ticks(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    first = reconciler.process(entry("e2", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    second = reconciler.tick_and_emit(datetime(2026, 3, 14, 10, 0))
    third = reconciler.tick_and_emit(datetime(2026, 3, 14, 11, 0))
    assert codes(first).count("MISSED_DOSE") == 1
    assert "MISSED_DOSE" not in codes(second) + codes(third)


def test_dose_exactly_at_window_close_is_not_missed(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    reconciler.process(entry("e2", "2026-03-14T08:30:00"), intake())
    out = reconciler.process(entry("e3", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    assert "MISSED_DOSE" not in codes(out)


def test_multi_day_gap_reports_each_missed_day(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-17T09:00:00"), bed("BED_RETURN"))
    missed = [d for d in out if d.code == "MISSED_DOSE" and d.med_id == "MED_MORNING"]
    assert len(missed) == 4          # 14th, 15th, 16th, 17th
    assert {d.ts.date().day for d in missed} == {14, 15, 16, 17}


def test_missed_dose_backfill_is_capped(reconciler):
    """A months-old timeline must not emit thousands of retroactive alerts."""
    reconciler.process(entry("e1", "2026-01-01T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-06-01T09:00:00"), bed("BED_RETURN"))
    missed = [d for d in out if d.code == "MISSED_DOSE"]
    ceiling = 2 * (MAX_MISSED_DOSE_BACKFILL_DAYS + 1)   # 2 medications x days inclusive
    assert 0 < len(missed) <= ceiling


def test_first_event_does_not_backfill_history(reconciler):
    """A cold start must not dump weeks of retroactive alerts."""
    out = reconciler.process(entry("e1", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    assert "MISSED_DOSE" not in codes(out)


# ---------------------------------------------------------------- behaviour

def test_prolonged_night_absence_alerts(reconciler):
    reconciler.process(entry("e1", "2026-03-15T02:14:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-15T02:41:00"), bed("BED_EXIT"))
    assert "NIGHT_OUT_OF_BED" in codes(out)


def test_brief_night_absence_does_not_alert(reconciler):
    reconciler.process(entry("e1", "2026-03-15T02:14:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-15T02:20:00"), bed("BED_RETURN"))
    assert "NIGHT_OUT_OF_BED" not in codes(out)


def test_daytime_absence_does_not_alert(reconciler):
    reconciler.process(entry("e1", "2026-03-14T09:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T11:00:00"), bed("BED_EXIT"))
    assert "NIGHT_OUT_OF_BED" not in codes(out)


def test_night_alert_fires_once_per_episode(reconciler):
    reconciler.process(entry("e1", "2026-03-15T02:14:00"), bed("BED_EXIT"))
    first = reconciler.process(entry("e2", "2026-03-15T02:41:00"), bed("BED_EXIT"))
    second = reconciler.process(entry("e3", "2026-03-15T02:50:00"), bed("BED_EXIT"))
    assert codes(first).count("NIGHT_OUT_OF_BED") == 1
    assert "NIGHT_OUT_OF_BED" not in codes(second)


def test_returning_to_bed_rearms_the_night_alert(reconciler):
    reconciler.process(entry("e1", "2026-03-15T00:10:00"), bed("BED_EXIT"))
    reconciler.process(entry("e2", "2026-03-15T00:40:00"), bed("BED_EXIT"))
    reconciler.process(entry("e3", "2026-03-15T00:45:00"), bed("BED_RETURN"))
    reconciler.process(entry("e4", "2026-03-15T03:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e5", "2026-03-15T03:30:00"), bed("BED_EXIT"))
    assert "NIGHT_OUT_OF_BED" in codes(out)


# --------------------------------------------------------------- idempotency

def test_replaying_the_same_entry_records_nothing_new(reconciler, store):
    first = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    second = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert codes(first) == ["MED_ON_TIME"]
    assert second == []
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_fresh_reconciler_over_same_store_is_still_idempotent(plan, store):
    """Simulates re-running the pipeline against an existing database."""
    r1 = CareReconciler(plan, store, memory_path=None)
    r1.process(entry("e1", "2026-03-14T07:58:00"), intake())

    r2 = CareReconciler(plan, store, memory_path=None)
    out = r2.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert out == []
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_wrong_patient_is_rejected(reconciler):
    e = entry("e1", "2026-03-14T07:58:00")
    e.patient_id = "P999"
    with pytest.raises(ValueError, match="P999"):
        reconciler.process(e, intake())


def test_no_wall_clock_dependence(reconciler):
    """Decisions must depend on the timeline, never on when the demo is run."""
    out = reconciler.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert out[0].ts == datetime(2026, 3, 14, 7, 58)
