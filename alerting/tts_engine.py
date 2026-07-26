"""Bedside speaker output.

pyttsx3 needs a system speech engine, which a headless OpenShell sandbox will not have.
That is expected: the constructor degrades to stdout rather than failing, because a
missing speaker must never stop the safety loop.
"""

from __future__ import annotations


class LocalTTS:
    def __init__(self, enabled: bool = True, rate: int = 140):
        self.engine = None
        if not enabled:
            return
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)     # slower, for elderly clarity
            self.engine.setProperty("volume", 1.0)
        except Exception as exc:                      # noqa: BLE001
            print(f"[tts] unavailable, falling back to stdout: {exc}")
            self.engine = None

    def speak(self, text: str) -> None:
        print(f"          [SPEAKER] {text}")
        if self.engine is None:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:                      # noqa: BLE001
            print(f"[tts] playback failed: {exc}")
            self.engine = None                        # stop retrying a broken engine
