from __future__ import annotations

from datetime import datetime

import pytest

from careshell.reconciler import CareReconciler
from careshell.schemas import CareEvent, TimelineEntry
from careshell.store import CareStore

PLAN = {
    "patient_id": "P104",
    "display_name": "John Doe",
    "room": "104",
    "medications": [
        {
            "id": "MED_MORNING",
            "name": "Beta Blocker",
            "description": "blue round tablet",
            "scheduled": "08:00",
            "window_start": "07:30",
            "window_end": "08:30",
            "max_daily_doses": 1,
            "lockout_hours": 8,
        },
        {
            "id": "MED_EVENING",
            "name": "Evening Statin",
            "description": "white round tablet",
            "scheduled": "20:00",
            "window_start": "19:30",
            "window_end": "20:30",
            "max_daily_doses": 1,
            "lockout_hours": 8,
        },
    ],
    "behavior": {
        "night_start": "22:00",
        "night_end": "06:00",
        "max_out_of_bed_minutes_at_night": 15,
    },
}


@pytest.fixture
def plan() -> dict:
    import copy

    return copy.deepcopy(PLAN)


@pytest.fixture
def store(tmp_path) -> CareStore:
    s = CareStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def reconciler(plan, store) -> CareReconciler:
    return CareReconciler(plan, store, memory_path=None)


def entry(eid: str, ts: str, text: str = "observation", source: str = "nurse_note") -> TimelineEntry:
    return TimelineEntry(
        id=eid, ts=datetime.fromisoformat(ts), patient_id="P104", source=source, text=text
    )


def intake(med_id: str | None = "MED_MORNING", certainty: str = "CONFIRMED") -> CareEvent:
    return CareEvent(kind="MED_INTAKE", med_id=med_id, certainty=certainty, summary="took a pill")


def bed(kind: str) -> CareEvent:
    return CareEvent(kind=kind, certainty="CONFIRMED", summary=kind.lower())


def codes(decisions) -> list[str]:
    return [d.code for d in decisions]
