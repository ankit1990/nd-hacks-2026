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

from careshell.store import CareStore

DB_PATH = os.environ.get("CARESHELL_DB", "workspace/careshell.db")
TEMPLATE = Path(__file__).parent / "templates" / "index.html"

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
async def index() -> str:
    return TEMPLATE.read_text()


@app.get("/api/history")
async def history(patient_id: str | None = None, limit: int = 100) -> dict:
    """Decision history, so a reconnecting console is not blank."""
    with CareStore(DB_PATH) as store:
        decisions = store.recent_decisions(patient_id, min(limit, 500))
        # Most recent day with doses, not today: a replayed timeline is historical.
        adherence = store.adherence_summary(patient_id) if patient_id else []
    return {"decisions": decisions, "adherence": adherence}


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


@app.post("/api/event")
async def receive_event(event: dict) -> dict:
    await manager.broadcast(event)
    return {"status": "ok"}


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "db": DB_PATH}
