"""Push decisions to the LAN nurse console.

Failures are swallowed on purpose. The console being down must never interrupt the
bedside loop, and SQLite already holds the authoritative history the console can
re-read on reconnect.
"""

from __future__ import annotations

import requests


class LANBroadcaster:
    def __init__(self, url: str = "http://127.0.0.1:8000/api/event", timeout: int = 2):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self._warned = False

    def send(self, payload: dict) -> bool:
        try:
            self.session.post(self.url, json=payload, timeout=self.timeout)
            self._warned = False
            return True
        except requests.RequestException as exc:
            if not self._warned:                      # warn once, not once per event
                print(f"[console] unreachable at {self.url}, continuing: {exc}")
                self._warned = True
            return False
