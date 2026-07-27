"""Regressions for bugs found in adversarial review. Each test failed before its fix."""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from careshell.extractor import LLMExtractor
from careshell.reconciler import CareReconciler
from careshell.schemas import CareEvent, Decision, TimelineEntry
from careshell.store import MAX_HISTORY_ROWS, CareStore, SchemaVersionError
from careshell.timeline import end_of_day, load_timeline
from tests.conftest import bed, codes, entry, intake


# ---------------------------------------------------------------- tz-aware input

def test_tz_aware_timestamp_is_normalised_to_naive():
    """A sensor bridge emitting UTC 'Z' must not crash the run."""
    e = TimelineEntry(id="x", ts="2026-03-14T07:00:00+00:00", patient_id="P104",
                      source="sensor", text="bed exit")
    assert e.ts.tzinfo is None


def test_tz_aware_timeline_does_not_crash_the_reconciler(reconciler):
    """Previously raised TypeError comparing naive window closes to aware timestamps."""
    reconciler.process(entry("e1", "2026-03-14T07:00:00+00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T09:00:00+00:00"), bed("BED_RETURN"))
    assert isinstance(out, list)
    assert all(d.ts.tzinfo is None for d in out)


def test_aware_and_equivalent_naive_input_agree(plan, tmp_path):
    """An offset timestamp and the same local wall-clock instant must decide alike."""
    local_offset = datetime.now().astimezone().strftime("%z")
    aware = f"2026-03-14T07:58:00{local_offset[:3]}:{local_offset[3:]}"

    def run(ts: str, db: str) -> list[str]:
        with CareStore(tmp_path / db) as store:
            r = CareReconciler(plan, store, memory_path=None)
            r.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
            return codes(r.process(entry("e2", ts), intake()))

    assert run(aware, "aware.db") == run("2026-03-14T07:58:00", "naive.db") == ["MED_ON_TIME"]


def test_mixed_naive_and_aware_timeline_sorts(tmp_path):
    rows = [
        {"id": "b", "ts": "2026-03-14T09:00:00Z", "patient_id": "P104",
         "source": "sensor", "text": "later"},
        {"id": "a", "ts": "2026-03-14T07:00:00", "patient_id": "P104",
         "source": "sensor", "text": "earlier"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    entries = load_timeline(p)          # previously raised TypeError in sorted()
    assert len(entries) == 2
    assert all(e.ts.tzinfo is None for e in entries)


def test_tz_aware_stream_checkpoints_build(plan):
    from careshell.stream import build_checkpoints

    e = TimelineEntry(id="x", ts="2026-03-14T07:00:00+05:30", patient_id="P104",
                      source="sensor", text="bed exit")
    assert build_checkpoints([e], plan)  # previously raised TypeError


# ------------------------------------------- no false MISSED_DOSE on a recorded dose

def test_dose_at_window_close_emits_no_missed_dose(reconciler, store):
    """tick() used to run before the dose was recorded, so both fired at once."""
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T08:30:00"), intake())
    assert codes(out) == ["MED_ON_TIME"]
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1


def test_late_dose_emits_out_of_window_not_missed(reconciler):
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T09:00:00"), intake())
    assert "MED_OUT_OF_WINDOW" in codes(out)
    assert "MISSED_DOSE" not in codes(out), "contradicts the dose recorded in the same call"


def test_missed_dose_still_fires_when_no_dose_arrives(reconciler):
    """The fix must not suppress genuine missed doses."""
    reconciler.process(entry("e1", "2026-03-14T07:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T09:00:00"), bed("BED_RETURN"))
    assert "MISSED_DOSE" in codes(out)


def test_returning_after_long_night_absence_still_alerts(reconciler):
    """Bed state is updated after tick(), so the ending absence is still evaluated."""
    reconciler.process(entry("e1", "2026-03-15T02:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-15T02:45:00"), bed("BED_RETURN"))
    assert "NIGHT_OUT_OF_BED" in codes(out)


# ---------------------------------------- uninterpretable observations reach a human

def test_extraction_failure_produces_a_review_decision(reconciler):
    """kind=OTHER fell through every branch and emitted nothing at all."""
    failed = CareEvent(kind="OTHER", certainty="UNCLEAR", needs_review=True,
                       summary="Extraction failed (ConnectionError): bed exit detected")
    out = reconciler.process(entry("e1", "2026-03-15T02:00:00"), failed)
    assert codes(out) == ["NEEDS_HUMAN_REVIEW"]


def test_ordinary_other_events_stay_quiet(reconciler):
    """Only failures escalate; a blood-pressure reading is not a review item."""
    ordinary = CareEvent(kind="OTHER", certainty="UNCLEAR", summary="BP 128/76")
    assert reconciler.process(entry("e1", "2026-03-14T13:00:00"), ordinary) == []


def test_review_event_from_extractor_sets_the_flag():
    ex = LLMExtractor(endpoint="http://127.0.0.1:9/v1/chat/completions", model="stub", timeout=1)
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), [])
    assert event.needs_review is True


def test_llm_outage_surfaces_every_line_for_review(tmp_path, capsys):
    from careshell.run import main

    rows = [{"id": "e1", "ts": "2026-03-14T07:58:00", "patient_id": "P104",
             "source": "nurse_note", "text": "swallowed a tablet"}]
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    db = str(tmp_path / "d.db")
    assert main([str(p), "--plan", "workspace/care_plan.yaml", "--db", db,
                 "--memory", "", "--console", "", "--no-tts",
                 "--endpoint", "http://127.0.0.1:9/v1/chat/completions", "--timeout", "1"]) == 0
    with CareStore(db) as store:
        assert store.conn.execute("SELECT COUNT(*) c FROM doses").fetchone()["c"] == 0
        assert any(d["code"] == "NEEDS_HUMAN_REVIEW" for d in store.recent_decisions())


# ------------------------------------------------------------------ store hardening

def test_negative_limit_does_not_dump_history(store):
    for i in range(5):
        store.record_decision(Decision(ts=datetime(2026, 3, 14, 8, i), entry_id=f"x{i}",
                                       patient_id="P104", code="MED_ON_TIME", message="m"))
    assert len(store.recent_decisions("P104", -1)) == 1   # SQLite reads -1 as "no limit"
    assert len(store.recent_decisions("P104", 0)) == 1


def test_limit_is_capped_at_the_maximum(store):
    assert len(store.recent_decisions("P104", 10_000)) <= MAX_HISTORY_ROWS


def test_schema_version_mismatch_is_fatal(tmp_path):
    path = tmp_path / "old.db"
    CareStore(path).close()
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaVersionError, match="99"):
        CareStore(path)


def test_recent_adherence_groups_by_day_and_med(store):
    store.record_dose("a", "P104", "MED_MORNING", datetime(2026, 3, 14, 8, 0))
    store.record_dose("b", "P104", "MED_EVENING", datetime(2026, 3, 14, 20, 0))
    store.record_dose("c", "P104", "MED_MORNING", datetime(2026, 3, 15, 8, 0))
    rows = store.recent_adherence("P104")
    assert rows[0]["day"] == "2026-03-15"
    assert {(r["day"], r["med_id"], r["doses"]) for r in rows} == {
        ("2026-03-15", "MED_MORNING", 1),
        ("2026-03-14", "MED_MORNING", 1),
        ("2026-03-14", "MED_EVENING", 1),
    }


# -------------------------------------------- night absence counts only sleep hours

def test_daytime_absence_is_not_reported_as_night_wandering(reconciler):
    """Live-run bug: a 07:41 bed exit with no return read as 978 night-time minutes."""
    reconciler.process(entry("e1", "2026-03-14T07:41:00"), bed("BED_EXIT"))
    out = reconciler.flush(datetime(2026, 3, 14, 23, 59, 59))
    assert "NIGHT_OUT_OF_BED" not in codes(out)


def test_absence_crossing_into_night_counts_from_night_start(reconciler):
    reconciler.process(entry("e1", "2026-03-14T20:00:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-14T22:40:00"), bed("BED_EXIT"))
    alert = next(d for d in out if d.code == "NIGHT_OUT_OF_BED")
    assert "40 minutes" in alert.message      # from 22:00, not from 20:00
    assert "22:00" in alert.message


def test_absence_wholly_inside_night_is_unchanged(reconciler):
    reconciler.process(entry("e1", "2026-03-15T02:14:00"), bed("BED_EXIT"))
    out = reconciler.process(entry("e2", "2026-03-15T02:41:00"), bed("BED_EXIT"))
    alert = next(d for d in out if d.code == "NIGHT_OUT_OF_BED")
    assert "27 minutes" in alert.message
    assert "02:14" in alert.message


def test_night_start_resolves_to_previous_day_after_midnight(plan):
    """At 02:00 the relevant night_start is 22:00 yesterday, not today."""
    assert CareReconciler._night_start_before(
        datetime(2026, 3, 15, 2, 0), plan["behavior"]
    ) == datetime(2026, 3, 14, 22, 0)


# ------------------------------------------------------- crash atomicity (invariants)

def test_partial_write_rolls_back_completely(plan, tmp_path):
    """A crash mid-process must leave no dose, no observation, no decision."""
    with CareStore(tmp_path / "rb.db") as store:
        r = CareReconciler(plan, store, memory_path=None)
        e, ev = entry("e1", "2026-03-14T07:58:00"), intake()

        def boom(*_a, **_k):
            raise RuntimeError("simulated crash")

        store.record_observation = boom
        with pytest.raises(RuntimeError):
            r.process(e, ev)

        del store.record_observation
        assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 0
        assert not store.has_observation("e1")
        assert store.recent_decisions("P104") == []


def test_replay_after_a_rolled_back_crash_decides_correctly(plan, tmp_path):
    db = tmp_path / "rb2.db"
    e, ev = entry("e1", "2026-03-14T07:58:00"), intake()

    with CareStore(db) as store:
        r = CareReconciler(plan, store, memory_path=None)

        def boom(*_a, **_k):
            raise RuntimeError("simulated crash")

        store.record_observation = boom
        with pytest.raises(RuntimeError):
            r.process(e, ev)

    with CareStore(db) as store:
        out = CareReconciler(plan, store, memory_path=None).process(e, ev)
    assert codes(out) == ["MED_ON_TIME"], "must not block a dose against its own remnant"


def test_an_entry_can_never_block_itself(store, plan):
    """Defence in depth: even a stray dose row for this entry must not block it."""
    store.record_dose("e1", "P104", "MED_MORNING", datetime(2026, 3, 14, 7, 58))
    out = CareReconciler(plan, store, memory_path=None).process(
        entry("e1", "2026-03-14T07:58:00"), intake()
    )
    assert codes(out) == ["MED_ON_TIME"]


def test_a_different_entry_still_blocks(store, plan):
    """The self-exclusion must not weaken the real double-dose guard."""
    store.record_dose("earlier", "P104", "MED_MORNING", datetime(2026, 3, 14, 7, 58))
    out = CareReconciler(plan, store, memory_path=None).process(
        entry("e2", "2026-03-14T08:31:00"), intake()
    )
    assert codes(out) == ["DOUBLE_DOSE_BLOCKED"]


def test_nested_transactions_commit_once(store):
    with store.transaction():
        store.record_dose("a", "P104", "MED_MORNING", datetime(2026, 3, 14, 8, 0))
        with store.transaction():
            store.record_dose("b", "P104", "MED_EVENING", datetime(2026, 3, 14, 20, 0))
    assert store.dose_count_on("P104", "MED_MORNING", date(2026, 3, 14)) == 1
    assert store.dose_count_on("P104", "MED_EVENING", date(2026, 3, 14)) == 1


def test_memory_log_is_not_written_for_a_rolled_back_decision(plan, tmp_path):
    memory = tmp_path / "MEMORY.md"
    with CareStore(tmp_path / "m.db") as store:
        r = CareReconciler(plan, store, memory_path=str(memory))

        def boom(*_a, **_k):
            raise RuntimeError("simulated crash")

        store.record_observation = boom
        with pytest.raises(RuntimeError):
            r.process(entry("e1", "2026-03-14T07:58:00"), intake())
    assert not memory.exists() or memory.read_text() == ""


# ----------------------------------------------------------------- shared helpers

def test_end_of_day_does_not_cross_midnight():
    assert end_of_day(datetime(2026, 3, 14, 9, 0)) == datetime(2026, 3, 14, 23, 59, 59)


def test_stream_and_timeline_share_one_parser():
    """Both readers must agree on what a valid entry is."""
    from careshell import stream, timeline

    assert stream.parse_entry is timeline.parse_entry


def test_unterminated_line_is_discarded_not_buffered_forever(tmp_path):
    from careshell.stream import MAX_PARTIAL_LINE_BYTES, follow_timeline

    p = tmp_path / "live.jsonl"
    p.write_text("x" * (MAX_PARTIAL_LINE_BYTES + 10))   # no newline, ever
    entries = list(follow_timeline(p, poll_seconds=0.01, stop_after_idle=0.05))
    assert entries == []


# ------------------------------------------------------------------- CLI guards

@pytest.mark.parametrize("extra", [["--stream", "--speed", "0"],
                                   ["--stream", "--speed", "-5"],
                                   ["--max-gap", "-1"]])
def test_invalid_pacing_exits_two_not_traceback(tmp_path, extra):
    from careshell.run import main

    assert main(["data/patients/P104.jsonl", "--plan", "workspace/care_plan.yaml",
                 "--db", str(tmp_path / "d.db"), "--console", "", "--offline", *extra]) == 2


# -------------------------------------------------------------------- dashboard

@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.app as app_module

    db = tmp_path / "dash.db"
    with CareStore(db) as store:
        store.record_decision(Decision(ts=datetime(2026, 3, 14, 8, 31), entry_id="e7",
                                       patient_id="P104", code="DOUBLE_DOSE_BLOCKED",
                                       med_id="MED_MORNING", message="blocked", speak=True))
        store.record_dose("e5", "P104", "MED_MORNING", datetime(2026, 3, 14, 7, 58))
    monkeypatch.setattr(app_module, "DB_PATH", str(db))
    return TestClient(app_module.app)


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_index_renders(client):
    assert "CareShell" in client.get("/").text


def test_history_returns_decisions_and_adherence(client):
    body = client.get("/api/history?patient_id=P104").json()
    assert body["decisions"][0]["code"] == "DOUBLE_DOSE_BLOCKED"
    assert body["adherence"][0]["med_id"] == "MED_MORNING"


def test_history_negative_limit_is_clamped(client):
    assert len(client.get("/api/history?patient_id=P104&limit=-1").json()["decisions"]) == 1


def test_event_rejects_a_malformed_payload(client):
    """A bare dict used to be broadcast verbatim to every nurse console."""
    assert client.post("/api/event", json={"not": "a decision"}).status_code == 422
    assert client.post("/api/event", json={"ts": "2026-03-14T08:00:00", "entry_id": "x",
                                           "patient_id": "P104", "code": "MADE_UP",
                                           "message": "spoof"}).status_code == 422


def test_event_accepts_a_real_decision_and_broadcasts(client):
    payload = Decision(ts=datetime(2026, 3, 14, 8, 31), entry_id="e9", patient_id="P104",
                       code="MED_ON_TIME", message="ok").model_dump(mode="json")
    with client.websocket_connect("/ws/events") as ws:
        assert client.post("/api/event", json=payload).status_code == 200
        assert ws.receive_json()["code"] == "MED_ON_TIME"


# ---------------------------------------------------- fixtures track the real plan

def test_test_plan_is_the_shipped_plan():
    from careshell.care_plan import load_care_plan
    from tests.conftest import PLAN

    assert PLAN == load_care_plan("workspace/care_plan.yaml")
