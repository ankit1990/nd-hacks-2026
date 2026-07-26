"""Input-handling tests: timeline parsing, care-plan validation, streaming pacing."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from careshell.care_plan import CarePlanError, load_care_plan, med_catalog, parse_hhmm
from careshell.schemas import TimelineEntry
from careshell.stream import build_checkpoints, follow_timeline, paced, window_closes
from careshell.timeline import TimelineError, load_timeline

ROWS = [
    {"id": "b", "ts": "2026-03-14T09:00:00", "patient_id": "P104",
     "source": "sensor", "text": "later"},
    {"id": "a", "ts": "2026-03-14T07:00:00", "patient_id": "P104",
     "source": "sensor", "text": "earlier"},
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------- timeline

def test_entries_are_sorted_by_timestamp(tmp_path):
    p = write_jsonl(tmp_path / "t.jsonl", ROWS)
    entries = load_timeline(p)
    assert [e.id for e in entries] == ["a", "b"]


def test_duplicate_ids_keep_the_first(tmp_path):
    rows = ROWS + [dict(ROWS[0], text="a duplicate")]
    p = write_jsonl(tmp_path / "t.jsonl", rows)
    entries = load_timeline(p)
    assert len(entries) == 2
    assert next(e for e in entries if e.id == "b").text == "later"


def test_malformed_line_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(ROWS[1]) + "\n{ not json\n" + json.dumps(ROWS[0]) + "\n")
    entries = load_timeline(p)
    assert [e.id for e in entries] == ["a", "b"]


def test_other_patients_are_filtered(tmp_path):
    rows = ROWS + [dict(ROWS[0], id="z", patient_id="P999")]
    p = write_jsonl(tmp_path / "t.jsonl", rows)
    entries = load_timeline(p, patient_id="P104")
    assert {e.patient_id for e in entries} == {"P104"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(TimelineError, match="not found"):
        load_timeline(tmp_path / "nope.jsonl")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("")
    with pytest.raises(TimelineError, match="no usable entries"):
        load_timeline(p)


def test_blank_text_is_rejected_by_the_schema():
    with pytest.raises(Exception):
        TimelineEntry(id="x", ts=datetime.now(), patient_id="P104", source="sensor", text="   ")


# ------------------------------------------------------------------ care plan

MINIMAL = """
patient_id: P1
display_name: Jane Roe
medications:
  - id: M1
    name: Pill
    scheduled: "08:00"
    window_start: "07:30"
    window_end: "08:30"
    max_daily_doses: 1
    lockout_hours: 8
behavior:
  night_start: "22:00"
  night_end: "06:00"
  max_out_of_bed_minutes_at_night: 15
"""


def plan_file(tmp_path, text):
    p = tmp_path / "plan.yaml"
    p.write_text(text)
    return p


def test_minimal_plan_loads(tmp_path):
    plan = load_care_plan(plan_file(tmp_path, MINIMAL))
    assert plan["medications"][0]["description"] == "Pill"   # defaults to name
    assert med_catalog(plan) == [{"id": "M1", "name": "Pill", "description": "Pill"}]


def test_unquoted_times_are_coerced(tmp_path):
    """YAML 1.1 turns an unquoted 08:00 into the integer 480."""
    plan = load_care_plan(plan_file(tmp_path, MINIMAL.replace('"08:30"', "08:30")))
    assert plan["medications"][0]["window_end"] == "08:30"


def test_missing_key_is_fatal(tmp_path):
    with pytest.raises(CarePlanError, match="lockout_hours"):
        load_care_plan(plan_file(tmp_path, MINIMAL.replace("    lockout_hours: 8\n", "")))


def test_duplicate_med_id_is_fatal(tmp_path):
    doubled = MINIMAL.replace(
        "behavior:",
        '  - id: M1\n    name: Other\n    scheduled: "20:00"\n'
        '    window_start: "19:30"\n    window_end: "20:30"\n'
        "    max_daily_doses: 1\n    lockout_hours: 8\nbehavior:",
    )
    with pytest.raises(CarePlanError, match="duplicate medication id"):
        load_care_plan(plan_file(tmp_path, doubled))


def test_inverted_window_is_fatal(tmp_path):
    with pytest.raises(CarePlanError, match="window_start must be before"):
        load_care_plan(plan_file(tmp_path, MINIMAL.replace('"07:30"', '"09:30"')))


def test_non_positive_lockout_is_fatal(tmp_path):
    with pytest.raises(CarePlanError, match="lockout_hours"):
        load_care_plan(plan_file(tmp_path, MINIMAL.replace("lockout_hours: 8", "lockout_hours: 0")))


def test_missing_plan_is_fatal(tmp_path):
    with pytest.raises(CarePlanError, match="not found"):
        load_care_plan(tmp_path / "absent.yaml")


def test_parse_hhmm():
    assert parse_hhmm("08:05").hour == 8
    assert parse_hhmm("08:05").minute == 5


def test_shipped_plan_is_valid():
    plan = load_care_plan("workspace/care_plan.yaml")
    assert plan["patient_id"] == "P104"
    assert {m["id"] for m in plan["medications"]} == {"MED_MORNING", "MED_EVENING"}


def test_shipped_timeline_matches_shipped_plan():
    plan = load_care_plan("workspace/care_plan.yaml")
    entries = load_timeline("data/patients/P104.jsonl", patient_id=plan["patient_id"])
    assert len(entries) >= 10
    assert entries == sorted(entries, key=lambda e: e.ts)


# -------------------------------------------------------------------- stream

def test_window_closes_spans_days():
    plan = load_care_plan("workspace/care_plan.yaml")
    closes = window_closes(plan, datetime(2026, 3, 14, 0, 0), datetime(2026, 3, 15, 23, 59))
    assert len(closes) == 4                       # 2 medications x 2 days
    assert closes == sorted(closes)


def test_checkpoints_interleave_events_and_ticks(tmp_path):
    plan = load_care_plan("workspace/care_plan.yaml")
    entries = load_timeline("data/patients/P104.jsonl", patient_id="P104")
    cps = build_checkpoints(entries, plan)
    assert len(cps) > len(entries)
    assert cps == sorted(cps, key=lambda c: c.ts)


def test_event_sorts_before_tick_at_the_same_instant():
    """A dose taken exactly at window close counts as taken, not missed."""
    plan = load_care_plan("workspace/care_plan.yaml")
    e = TimelineEntry(id="x", ts=datetime(2026, 3, 14, 8, 30), patient_id="P104",
                      source="nurse_note", text="took it")
    cps = build_checkpoints([e], plan)
    at_close = [c for c in cps if c.ts == datetime(2026, 3, 14, 8, 30)]
    assert at_close[0].is_event


def test_paced_delays_track_virtual_gaps():
    plan = load_care_plan("workspace/care_plan.yaml")
    entries = [
        TimelineEntry(id="a", ts=datetime(2026, 3, 14, 7, 0), patient_id="P104",
                      source="sensor", text="one"),
        TimelineEntry(id="b", ts=datetime(2026, 3, 14, 7, 2), patient_id="P104",
                      source="sensor", text="two"),
    ]
    slept: list[float] = []
    cps = [c for c in build_checkpoints(entries, plan) if c.is_event]
    list(paced(cps, speed=60.0, max_gap_seconds=10.0, sleeper=slept.append))
    assert slept == [2.0]            # 120 virtual seconds / 60 = 2 real seconds


def test_paced_caps_long_gaps():
    entries = [
        TimelineEntry(id="a", ts=datetime(2026, 3, 14, 7, 0), patient_id="P104",
                      source="sensor", text="one"),
        TimelineEntry(id="b", ts=datetime(2026, 3, 15, 7, 0), patient_id="P104",
                      source="sensor", text="two"),
    ]
    slept: list[float] = []
    from careshell.stream import Checkpoint

    list(paced([Checkpoint(e.ts, e) for e in entries], speed=60.0,
               max_gap_seconds=3.0, sleeper=slept.append))
    assert slept == [3.0]


def test_paced_rejects_non_positive_speed():
    with pytest.raises(ValueError, match="speed"):
        list(paced([], speed=0))


def test_follow_reads_appended_lines(tmp_path):
    p = write_jsonl(tmp_path / "live.jsonl", ROWS)
    entries = list(follow_timeline(p, patient_id="P104", poll_seconds=0.01,
                                   stop_after_idle=0.05))
    assert [e.id for e in entries] == ["b", "a"]   # follow preserves arrival order


def test_follow_skips_other_patients(tmp_path):
    rows = ROWS + [dict(ROWS[0], id="z", patient_id="P999")]
    p = write_jsonl(tmp_path / "live.jsonl", rows)
    entries = list(follow_timeline(p, patient_id="P104", poll_seconds=0.01,
                                   stop_after_idle=0.05))
    assert "z" not in [e.id for e in entries]


def test_follow_from_end_ignores_existing_content(tmp_path):
    p = write_jsonl(tmp_path / "live.jsonl", ROWS)
    entries = list(follow_timeline(p, poll_seconds=0.01, from_start=False,
                                   stop_after_idle=0.05))
    assert entries == []


def test_follow_creates_a_missing_file(tmp_path):
    p = tmp_path / "nested" / "live.jsonl"
    entries = list(follow_timeline(p, poll_seconds=0.01, stop_after_idle=0.05))
    assert entries == []
    assert p.exists()
