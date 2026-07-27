"""Local nurse console.

Serves decision history straight out of SQLite and live-streams new decisions over a
WebSocket. Bound to the LAN only -- the OpenShell ingress rule restricts :8000 to the
nurse-station subnet.
"""

from __future__ import annotations

import json
import os

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from careshell.care_plan import CarePlanError, load_care_plan
from careshell.schemas import Decision
from careshell.store import MAX_HISTORY_ROWS, CareStore

DB_PATH = os.environ.get("CARESHELL_DB", "workspace/careshell.db")
PLAN_PATH = os.environ.get("CARESHELL_PLAN", "workspace/care_plan.yaml")
TEMPLATES = Path(__file__).parent / "templates"
TEMPLATE = TEMPLATES / "index.html"        # operational console
PTV_TEMPLATE = TEMPLATES / "ptv.html"      # Personal Timeline View

# Codes that a nurse has to actually do something about, as opposed to a normal
# on-time dose. Drives the "open alerts" count in the header.
ACTIONABLE = {
    "DOUBLE_DOSE_BLOCKED",
    "MAX_DAILY_DOSES_BLOCKED",
    "MISSED_DOSE",
    "NIGHT_OUT_OF_BED",
    "NEEDS_HUMAN_REVIEW",
    "MED_OUT_OF_WINDOW",
}

app = FastAPI(title="CareShell Nurse Console")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        # Collect dead sockets instead of mutating self.active while iterating it.
        payload = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_text(payload)
            except Exception:                        # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def ptv() -> str:
    """Personal Timeline View — the patient-facing care timeline."""
    return PTV_TEMPLATE.read_text()


@app.get("/console", response_class=HTMLResponse)
async def console() -> str:
    """Operational feed: every decision, newest first, with acknowledgement."""
    return TEMPLATE.read_text()


@app.get("/api/history")
def history(patient_id: str | None = None, limit: int = 100) -> dict:
    """Decision history, so a reconnecting console is not blank.

    Deliberately a plain `def`: the body is blocking sqlite3 I/O, and Starlette runs
    sync handlers in a threadpool. As `async def` it would block the event loop and
    stall WebSocket delivery to every connected console.
    """
    # SQLite reads a negative LIMIT as "no limit", so clamp both ends.
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    with CareStore(DB_PATH) as store:
        decisions = store.recent_decisions(patient_id, limit)
        # Most recent day with doses, not today: a replayed timeline is historical.
        adherence = store.adherence_summary(patient_id) if patient_id else []
    return {"decisions": decisions, "adherence": adherence}


@app.get("/api/overview")
def overview(limit: int = 200) -> dict:
    """Everything the console header needs, in one round trip.

    Plain `def`, not `async`: the body is blocking sqlite3 I/O and Starlette runs sync
    handlers in a threadpool. As `async` it would stall WebSocket delivery.
    """
    limit = max(1, min(limit, MAX_HISTORY_ROWS))
    try:
        plan = load_care_plan(PLAN_PATH)
    except CarePlanError:
        plan = None

    patient_id = plan["patient_id"] if plan else None
    with CareStore(DB_PATH) as store:
        decisions = store.recent_decisions(patient_id, limit)
        adherence = store.adherence_summary(patient_id) if patient_id else []
        latest_day = store.latest_dose_day(patient_id) if patient_id else None

    meds = []
    if plan:
        taken = {row["med_id"]: row for row in adherence}
        for med in plan["medications"]:
            row = taken.get(med["id"])
            meds.append({
                "id": med["id"],
                "name": med["name"],
                "scheduled": med["scheduled"],
                "window": f"{med['window_start']}-{med['window_end']}",
                "max_daily_doses": med["max_daily_doses"],
                "doses": row["doses"] if row else 0,
                "last_ts": row["last_ts"] if row else None,
            })

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["code"]] = counts.get(d["code"], 0) + 1

    return {
        "patient": {
            "id": patient_id,
            "name": plan["display_name"] if plan else "Unknown",
            "room": str(plan.get("room", "—")) if plan else "—",
        },
        "medications": meds,
        "counts": counts,
        "actionable": sum(n for c, n in counts.items() if c in ACTIONABLE),
        "latest_dose_day": latest_day,
        "decisions": decisions,
    }


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()                  # client keepalive
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:                                # noqa: BLE001
        manager.disconnect(ws)


@app.post("/api/reset")
async def reset(clear_history: bool = True) -> dict:
    """Start a fresh demo run: wipe history and tell every open console to clear.

    Without the broadcast, a browser that was already open keeps rendering the
    previous run's rows and the new stream appends to stale content.
    """
    removed = 0
    if clear_history:
        with CareStore(DB_PATH) as store:
            with store.transaction():
                for table in ("decisions", "doses", "observations"):
                    removed += store.conn.execute(f"DELETE FROM {table}").rowcount or 0
    await manager.broadcast({"type": "reset"})
    return {"status": "ok", "rows_removed": removed}


@app.post("/api/event")
async def receive_event(event: Decision) -> dict:
    """Live decision push from the bedside loop.

    Typed as `Decision` rather than a bare dict so a malformed or fabricated payload is
    rejected with a 422 instead of being rendered on a nurse's screen as though it were
    a real clinical decision. This endpoint is still unauthenticated and protected only
    by the OpenShell ingress CIDR -- see "Known gaps" in the README.
    """
    payload = event.model_dump(mode="json")
    payload["type"] = "decision"
    await manager.broadcast(payload)
    return {"status": "ok"}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "db": DB_PATH}
