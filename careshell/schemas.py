"""Typed contracts for CareShell.

Three layers, deliberately separated:

  TimelineEntry  -- raw input, one line of the patient timeline file
  CareEvent      -- the LLM's interpretation of exactly one TimelineEntry
  Decision       -- the reconciler's deterministic output

The LLM only ever produces CareEvent. It never produces a Decision.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EventKind = Literal[
    "MED_INTAKE",         # patient ingested, or clearly attempted to ingest, a medication
    "BED_EXIT",
    "BED_RETURN",
    "SLEEP_DISTURBANCE",
    "OTHER",              # escape hatch; never triggers a safety action
]

Certainty = Literal["CONFIRMED", "LIKELY", "UNCLEAR"]

DecisionCode = Literal[
    "MED_ON_TIME",
    "MED_OUT_OF_WINDOW",
    "DOUBLE_DOSE_BLOCKED",
    "MAX_DAILY_DOSES_BLOCKED",
    "MISSED_DOSE",
    "NIGHT_OUT_OF_BED",
    "NEEDS_HUMAN_REVIEW",
]


class TimelineEntry(BaseModel):
    """One observation line from data/patients/<patient_id>.jsonl."""

    id: str = Field(description="Stable unique id. Reconciliation is idempotent on this.")
    ts: datetime = Field(description="ISO 8601. This is the system clock during a replay.")
    patient_id: str
    source: Literal["sensor", "nurse_note", "device"]
    text: str

    @field_validator("id", "patient_id", "text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("ts")
    @classmethod
    def _facility_local_naive(cls, v: datetime) -> datetime:
        """Normalise to a naive, facility-local wall clock.

        The input contract says ISO 8601, and a sensor bridge emitting UTC ("...Z") or
        an offset is entirely valid ISO 8601. Everything downstream compares entry
        timestamps against medication windows built with `datetime.combine`, which
        produces naive datetimes -- and Python raises TypeError when a naive and an
        aware datetime are compared. Without this, one offset-bearing line crashes the
        whole run.

        Converting rather than rejecting: on a facility appliance the host clock is the
        facility clock, so `astimezone()` gives the wall-clock time a nurse would read
        off the wall, which is what medication windows are expressed in.
        """
        return v.astimezone().replace(tzinfo=None) if v.tzinfo is not None else v


class CareEvent(BaseModel):
    """Structured interpretation of one TimelineEntry. Produced by the extractor."""

    kind: EventKind
    med_id: str | None = Field(
        default=None,
        description="An id from the patient's care plan, or null. Never invented.",
    )
    certainty: Certainty = Field(
        description="CONFIRMED only when the text states ingestion happened, not intent."
    )
    restlessness: int | None = Field(default=None, ge=1, le=5)
    summary: str = Field(default="", description="One clause restating the observation.")
    needs_review: bool = Field(
        default=False,
        description=(
            "Set when the observation could not be interpreted at all -- inference "
            "outage, unparseable output. Always surfaces on the nurse console, "
            "whatever `kind` says."
        ),
    )


class Decision(BaseModel):
    """Reconciler output. Deterministic function of (plan, state, event, time)."""

    ts: datetime
    entry_id: str
    patient_id: str
    code: DecisionCode
    med_id: str | None = None
    message: str
    speak: bool = False   # routed to the bedside speaker
