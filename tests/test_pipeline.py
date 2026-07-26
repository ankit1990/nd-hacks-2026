"""End-to-end pipeline tests. No network: the extractor is stubbed or offline."""

from __future__ import annotations

import json

import pytest

from careshell import run as run_module
from careshell.extractor import KeywordExtractor
from careshell.run import main
from careshell.store import CareStore

TIMELINE = [
    {"id": "e1", "ts": "2026-03-14T07:44:18", "patient_id": "P104", "source": "nurse_note",
     "text": "John reached for the blue bottle but set it back down."},
    {"id": "e2", "ts": "2026-03-14T07:58:22", "patient_id": "P104", "source": "nurse_note",
     "text": "John swallowed one blue beta blocker tablet with water."},
    {"id": "e3", "ts": "2026-03-14T08:31:10", "patient_id": "P104", "source": "nurse_note",
     "text": "He swallowed another blue beta blocker tablet."},
    {"id": "e4", "ts": "2026-03-14T21:00:00", "patient_id": "P104", "source": "sensor",
     "text": "Patient returned to bed."},
]


@pytest.fixture
def workspace(tmp_path):
    timeline = tmp_path / "t.jsonl"
    timeline.write_text("".join(json.dumps(r) + "\n" for r in TIMELINE))
    return {
        "timeline": str(timeline),
        "db": str(tmp_path / "careshell.db"),
        "memory": str(tmp_path / "MEMORY.md"),
    }


def base_args(ws, *extra):
    return [
        ws["timeline"], "--plan", "workspace/care_plan.yaml",
        "--db", ws["db"], "--memory", ws["memory"],
        "--console", "", "--no-tts", "--offline", "--trust-keywords", *extra,
    ]


def decisions(db):
    with CareStore(db) as store:
        return store.recent_decisions()


def test_full_run_exits_clean(workspace, capsys):
    assert main(base_args(workspace)) == 0
    out = capsys.readouterr().out
    assert "MED_ON_TIME" in out
    assert "DOUBLE_DOSE_BLOCKED" in out


def test_only_one_dose_survives_the_double(workspace):
    main(base_args(workspace))
    with CareStore(workspace["db"]) as store:
        rows = store.conn.execute(
            "SELECT * FROM doses WHERE med_id = 'MED_MORNING'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["entry_id"] == "e2"


def test_hedged_line_becomes_human_review(workspace):
    main(base_args(workspace))
    review = [d for d in decisions(workspace["db"]) if d["code"] == "NEEDS_HUMAN_REVIEW"]
    assert any(d["entry_id"] == "e1" for d in review)


def test_evening_dose_is_reported_missing(workspace):
    main(base_args(workspace))
    missed = [d for d in decisions(workspace["db"]) if d["code"] == "MISSED_DOSE"]
    assert {d["med_id"] for d in missed} == {"MED_EVENING"}


def test_rerun_is_idempotent(workspace):
    main(base_args(workspace))
    first = decisions(workspace["db"])
    main(base_args(workspace))
    second = decisions(workspace["db"])
    assert len(first) == len(second)

    with CareStore(workspace["db"]) as store:
        n = store.conn.execute("SELECT COUNT(*) AS n FROM doses").fetchone()["n"]
    assert n == 1


def test_memory_log_is_written(workspace):
    main(base_args(workspace))
    text = open(workspace["memory"]).read()
    assert "DOUBLE_DOSE_BLOCKED" in text
    assert text.startswith("- `2026-03-14")


def test_stream_mode_produces_the_same_decisions(workspace, tmp_path):
    main(base_args(workspace))
    batch = {(d["entry_id"], d["code"]) for d in decisions(workspace["db"])}

    streamed_db = str(tmp_path / "stream.db")
    args = base_args(workspace, "--stream", "--speed", "1000000", "--max-gap", "0")
    args[args.index("--db") + 1] = streamed_db
    main(args)
    streamed = {(d["entry_id"], d["code"]) for d in decisions(streamed_db)}

    assert batch == streamed


def test_follow_mode_consumes_a_live_feed(tmp_path):
    live = tmp_path / "live.jsonl"
    live.write_text("".join(json.dumps(r) + "\n" for r in TIMELINE))
    db = str(tmp_path / "follow.db")
    rc = main([
        "--follow", str(live), "--idle-timeout", "0.2", "--from-start",
        "--plan", "workspace/care_plan.yaml", "--db", db,
        "--memory", "", "--console", "", "--no-tts", "--offline", "--trust-keywords",
    ])
    assert rc == 0
    codes = {d["code"] for d in decisions(db)}
    assert "MED_ON_TIME" in codes
    assert "DOUBLE_DOSE_BLOCKED" in codes


def test_llm_outage_degrades_to_review_not_crash(workspace, monkeypatch):
    """Point the extractor at a dead endpoint: every line becomes human review."""
    args = [
        workspace["timeline"], "--plan", "workspace/care_plan.yaml",
        "--db", workspace["db"], "--memory", "", "--console", "", "--no-tts",
        "--endpoint", "http://127.0.0.1:9/v1/chat/completions", "--timeout", "1",
    ]
    assert main(args) == 0
    with CareStore(workspace["db"]) as store:
        n = store.conn.execute("SELECT COUNT(*) AS n FROM doses").fetchone()["n"]
    assert n == 0


def test_missing_plan_exits_two(workspace):
    assert main([workspace["timeline"], "--plan", "nope.yaml", "--console", ""]) == 2


def test_missing_timeline_exits_two(workspace):
    assert main([
        "nope.jsonl", "--plan", "workspace/care_plan.yaml",
        "--db", workspace["db"], "--console", "", "--offline",
    ]) == 2


def test_conflicting_modes_exit_two(workspace):
    assert main([workspace["timeline"], "--follow", "x.jsonl"]) == 2


def test_trust_keywords_requires_offline(workspace):
    assert main([workspace["timeline"], "--trust-keywords", "--plan",
                 "workspace/care_plan.yaml", "--console", ""]) == 2


def test_shipped_demo_timeline_runs(tmp_path, capsys):
    rc = main([
        "data/patients/P104.jsonl", "--plan", "workspace/care_plan.yaml",
        "--db", str(tmp_path / "demo.db"), "--memory", "",
        "--console", "", "--no-tts", "--offline", "--trust-keywords",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    for expected in ("MED_ON_TIME", "DOUBLE_DOSE_BLOCKED", "NIGHT_OUT_OF_BED", "MISSED_DOSE"):
        assert expected in out, f"demo timeline should surface {expected}"
