"""Loader and validator for workspace/care_plan.yaml.

The v1 spec kept the schedule in markdown prose that no code ever opened, so the
medication windows and dose caps were decorative. This module is the fix: the plan is
machine-read, and a malformed plan fails loudly at startup rather than silently
degrading a safety rule at 3am.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import yaml

REQUIRED_MED_FIELDS = (
    "id",
    "name",
    "scheduled",
    "window_start",
    "window_end",
    "max_daily_doses",
    "lockout_hours",
)

REQUIRED_BEHAVIOR_FIELDS = ("night_start", "night_end", "max_out_of_bed_minutes_at_night")


class CarePlanError(ValueError):
    """Raised when care_plan.yaml is missing or internally inconsistent."""


def _coerce_hhmm(value: object, where: str) -> str:
    """Normalise a time field to 'HH:MM'.

    YAML 1.1 parses an unquoted 08:00 as sexagesimal (int 480) and 8:00 likewise, so a
    plan written without quotes would otherwise reach the reconciler as an integer and
    blow up mid-run. Accept what PyYAML hands us and normalise it here.
    """
    if isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) != 2:
            raise CarePlanError(f"{where}: expected 'HH:MM', got {value!r}")
        try:
            hh, mm = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise CarePlanError(f"{where}: expected 'HH:MM', got {value!r}") from exc
    elif isinstance(value, int):
        hh, mm = divmod(value, 60)          # YAML sexagesimal: 8:00 -> 480
    elif isinstance(value, _dt.time):
        hh, mm = value.hour, value.minute
    else:
        raise CarePlanError(f"{where}: expected 'HH:MM' string, got {type(value).__name__}")

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise CarePlanError(f"{where}: {hh:02d}:{mm:02d} is not a valid time of day")
    return f"{hh:02d}:{mm:02d}"


def parse_hhmm(value: str) -> _dt.time:
    """'08:30' -> datetime.time(8, 30). Assumes an already-normalised string."""
    hh, mm = value.split(":")
    return _dt.time(int(hh), int(mm))


def load_care_plan(path: str | Path) -> dict:
    """Read, validate and normalise a care plan. Raises CarePlanError on any problem."""
    path = Path(path)
    if not path.exists():
        raise CarePlanError(f"care plan not found: {path}")

    try:
        plan = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise CarePlanError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(plan, dict):
        raise CarePlanError(f"{path}: expected a mapping at the top level")

    for key in ("patient_id", "display_name", "medications", "behavior"):
        if key not in plan:
            raise CarePlanError(f"{path}: missing required key '{key}'")

    meds = plan["medications"]
    if not isinstance(meds, list) or not meds:
        raise CarePlanError(f"{path}: 'medications' must be a non-empty list")

    seen_ids: set[str] = set()
    for i, med in enumerate(meds):
        if not isinstance(med, dict):
            raise CarePlanError(f"{path}: medications[{i}] must be a mapping")
        for field in REQUIRED_MED_FIELDS:
            if field not in med:
                raise CarePlanError(f"{path}: medications[{i}] missing '{field}'")

        med_id = str(med["id"])
        if med_id in seen_ids:
            raise CarePlanError(f"{path}: duplicate medication id '{med_id}'")
        seen_ids.add(med_id)
        med["id"] = med_id
        med.setdefault("description", med["name"])

        for field in ("scheduled", "window_start", "window_end"):
            med[field] = _coerce_hhmm(med[field], f"{path}: {med_id}.{field}")

        if parse_hhmm(med["window_start"]) >= parse_hhmm(med["window_end"]):
            raise CarePlanError(
                f"{path}: {med_id} window_start must be before window_end "
                f"({med['window_start']} >= {med['window_end']}); "
                "overnight medication windows are not supported"
            )

        for field in ("max_daily_doses", "lockout_hours"):
            value = med[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise CarePlanError(f"{path}: {med_id}.{field} must be a positive number")

    behavior = plan["behavior"]
    if not isinstance(behavior, dict):
        raise CarePlanError(f"{path}: 'behavior' must be a mapping")
    for field in REQUIRED_BEHAVIOR_FIELDS:
        if field not in behavior:
            raise CarePlanError(f"{path}: behavior missing '{field}'")
    for field in ("night_start", "night_end"):
        behavior[field] = _coerce_hhmm(behavior[field], f"{path}: behavior.{field}")

    minutes = behavior["max_out_of_bed_minutes_at_night"]
    if not isinstance(minutes, int) or isinstance(minutes, bool) or minutes < 0:
        raise CarePlanError(
            f"{path}: behavior.max_out_of_bed_minutes_at_night must be a non-negative integer"
        )

    return plan


def med_catalog(plan: dict) -> list[dict]:
    """The subset of the plan the extractor prompt needs. Keeps PHI out of the prompt."""
    return [
        {"id": m["id"], "name": m["name"], "description": m["description"]}
        for m in plan["medications"]
    ]
