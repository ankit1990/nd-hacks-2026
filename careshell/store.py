"""SQLite store for historical patient data.

Everything the reconciler needs to remember lives here: which observations have been
processed, which doses were actually recorded, and every decision ever emitted.

Why SQLite and not the append-only markdown of the v1 spec: dose state was being
recovered by string-splitting log lines (`line.split("]")[0]`), which is both fragile and
unqueryable. A single file-backed database gives idempotency via UNIQUE constraints,
indexed per-medication lookups, and a real history the nurse console can page through --
with no server to run and no dependency to install.

Timestamps are stored as ISO 8601 strings. SQLite has no datetime type, and ISO 8601
sorts and compares correctly as text.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

from careshell.schemas import CareEvent, Decision, TimelineEntry

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Every timeline line we have interpreted. Primary key gives us replay idempotency.
CREATE TABLE IF NOT EXISTS observations (
    entry_id     TEXT PRIMARY KEY,
    patient_id   TEXT NOT NULL,
    ts           TEXT NOT NULL,
    source       TEXT NOT NULL,
    text         TEXT NOT NULL,
    kind         TEXT,
    certainty    TEXT,
    med_id       TEXT,
    summary      TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_patient_ts ON observations (patient_id, ts);

-- Doses we actually believe happened. Only CONFIRMED intakes reach this table.
CREATE TABLE IF NOT EXISTS doses (
    entry_id     TEXT PRIMARY KEY,
    patient_id   TEXT NOT NULL,
    med_id       TEXT NOT NULL,
    ts           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doses_lookup ON doses (patient_id, med_id, ts);

-- Every decision emitted. UNIQUE(entry_id, code) stops duplicate alerts on replay.
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id     TEXT NOT NULL,
    patient_id   TEXT NOT NULL,
    ts           TEXT NOT NULL,
    code         TEXT NOT NULL,
    med_id       TEXT,
    message      TEXT NOT NULL,
    speak        INTEGER NOT NULL DEFAULT 0,
    UNIQUE (entry_id, code)
);
CREATE INDEX IF NOT EXISTS idx_decisions_patient_ts ON decisions (patient_id, ts DESC);
"""


class CareStore:
    """File-backed history. Safe to open repeatedly against the same path."""

    def __init__(self, db_path: str | Path = "workspace/careshell.db"):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CareStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- observations

    def has_observation(self, entry_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM observations WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return row is not None

    def record_observation(self, entry: TimelineEntry, event: CareEvent) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO observations
               (entry_id, patient_id, ts, source, text, kind, certainty, med_id, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.patient_id,
                entry.ts.isoformat(),
                entry.source,
                entry.text,
                event.kind,
                event.certainty,
                event.med_id,
                event.summary,
            ),
        )

    # ----------------------------------------------------------------------- doses

    def record_dose(self, entry_id: str, patient_id: str, med_id: str, ts: datetime) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO doses (entry_id, patient_id, med_id, ts) VALUES (?, ?, ?, ?)",
            (entry_id, patient_id, med_id, ts.isoformat()),
        )

    def doses_since(self, patient_id: str, med_id: str, since: datetime) -> list[datetime]:
        """Doses of one medication at or after `since`, oldest first."""
        rows = self.conn.execute(
            """SELECT ts FROM doses
               WHERE patient_id = ? AND med_id = ? AND ts >= ?
               ORDER BY ts""",
            (patient_id, med_id, since.isoformat()),
        ).fetchall()
        return [datetime.fromisoformat(r["ts"]) for r in rows]

    def dose_count_on(self, patient_id: str, med_id: str, day: date) -> int:
        """How many doses of one medication were recorded on a calendar day."""
        row = self.conn.execute(
            """SELECT COUNT(*) AS n FROM doses
               WHERE patient_id = ? AND med_id = ? AND substr(ts, 1, 10) = ?""",
            (patient_id, med_id, day.isoformat()),
        ).fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------- decisions

    def record_decision(self, decision: Decision) -> bool:
        """Persist a decision. Returns False if this exact decision already exists."""
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO decisions
               (entry_id, patient_id, ts, code, med_id, message, speak)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.entry_id,
                decision.patient_id,
                decision.ts.isoformat(),
                decision.code,
                decision.med_id,
                decision.message,
                int(decision.speak),
            ),
        )
        return cur.rowcount > 0

    def recent_decisions(self, patient_id: str | None = None, limit: int = 100) -> list[dict]:
        """Newest-first decision history, for the nurse console."""
        if patient_id:
            rows = self.conn.execute(
                """SELECT * FROM decisions WHERE patient_id = ?
                   ORDER BY ts DESC, id DESC LIMIT ?""",
                (patient_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM decisions ORDER BY ts DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_dose_day(self, patient_id: str) -> str | None:
        """The most recent calendar day with a recorded dose, as 'YYYY-MM-DD'."""
        row = self.conn.execute(
            "SELECT substr(MAX(ts), 1, 10) AS day FROM doses WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        return row["day"] if row and row["day"] else None

    def adherence_summary(self, patient_id: str, day: date | str | None = None) -> list[dict]:
        """Per-medication dose counts for one day. Used by the console header.

        `day=None` means the most recent day that has any doses. A replayed timeline is
        historical, so keying strictly off today's date would show an empty panel.
        """
        if day is None:
            day = self.latest_dose_day(patient_id)
            if day is None:
                return []
        key = day if isinstance(day, str) else day.isoformat()
        rows = self.conn.execute(
            """SELECT med_id, COUNT(*) AS doses, MIN(ts) AS first_ts, MAX(ts) AS last_ts
               FROM doses
               WHERE patient_id = ? AND substr(ts, 1, 10) = ?
               GROUP BY med_id ORDER BY med_id""",
            (patient_id, key),
        ).fetchall()
        return [dict(r) for r in rows]
