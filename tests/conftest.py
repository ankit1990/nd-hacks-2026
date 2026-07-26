from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pytest

from careshell.care_plan import load_care_plan
from careshell.reconciler import CareReconciler
from careshell.schemas import CareEvent, TimelineEntry
from careshell.store import CareStore

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_PLAN_PATH = REPO_ROOT / "workspace" / "care_plan.yaml"

# Load the SHIPPED plan rather than hand-copying it. test_reconciler.py is the file the
# spec calls "if only one thing works on demo day, make it this" -- if it ran against a
# hardcoded duplicate, tuning a real window or lockout in care_plan.yaml would leave the
# safety tests passing against stale parameters.
PLAN = load_care_plan(SHIPPED_PLAN_PATH)
PATIENT_ID = PLAN["patient_id"]


@pytest.fixture
def plan() -> dict:
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
        id=eid, ts=datetime.fromisoformat(ts), patient_id=PATIENT_ID, source=source, text=text
    )


def intake(med_id: str | None = "MED_MORNING", certainty: str = "CONFIRMED") -> CareEvent:
    return CareEvent(kind="MED_INTAKE", med_id=med_id, certainty=certainty, summary="took a pill")


def bed(kind: str) -> CareEvent:
    return CareEvent(kind=kind, certainty="CONFIRMED", summary=kind.lower())


def codes(decisions) -> list[str]:
    return [d.code for d in decisions]
