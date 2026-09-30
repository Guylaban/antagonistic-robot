"""Furhat backend via the Furhat Remote API (pip package furhat-remote-api).

Requires the Remote API skill running on the robot or on the virtual
Furhat of the Furhat SDK (it listens on port 54321). Speech uses the
robot's own voice (`say`), optional non-verbal cues use Furhat gestures
and the LED ring.

Interruption: stop() sends `say_stop` and returns control to RAWR at once,
so the session never waits on the robot. In our tests with the virtual
Furhat of SDK 2.9.2, `say_stop` was acknowledged but did not cut the
audio short (the next utterance started only after the previous one
ended), so interruption is reported as "unverified" for this backend.
"""

import logging
import threading
from typing import Optional

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key

log = logging.getLogger(__name__)

# Gesture per condition (Furhat built-in gesture names); used only when expressions are on.
DEFAULT_GESTURES = {
    "support": "Smile", "neutral": None,
    "B": "GazeAway", "C": "BrowRaise", "D": "BrowFrown", "E": "Smile", "F": "ExpressAnger", "G": "ExpressAnger",
}
# LED colors (r, g, b) for state cues; used only when expressions are on.
LED = {"listening": (0, 60, 120), "thinking": (120, 90, 0), "idle": (0, 0, 0)}


class FurhatBackend(RobotBackend):
    """Furhat robot (physical or virtual) through the Remote API."""

    def __init__(self, host: str = "localhost", voice: Optional[str] = None, expressions: bool = False,
                 gestures: Optional[dict] = None, client=None):
        self._host = host
        self._voice = voice
        self._expressions = expressions
        self._gestures = {**DEFAULT_GESTURES, **(gestures or {})}
        self._client = client
        self._stop = threading.Event()
        self._done = threading.Event()
        self.capabilities = Capabilities(
            robot="Furhat", speech="robot TTS (Furhat voice)", interrupt="unverified", expressions=expressions,
            notes="say_stop did not interrupt audio on the virtual Furhat (SDK 2.9.2); verify on your robot",
        )

    def _api(self):
        if self._client is None:
            from furhat_remote_api import FurhatRemoteAPI  # optional dependency
            self._client = FurhatRemoteAPI(self._host)
        return self._client

    def connect(self) -> None:
        try:
            voices = [v.name for v in self._api().get_voices()]
        except Exception as e:
            raise RuntimeError(
                f"Furhat Remote API not reachable at {self._host}:54321 ({e}). Start the robot (or the virtual "
                f"Furhat in the Furhat SDK) and launch the Remote API skill."
            ) from e
        if self._voice:
            if self._voice not in voices:
                raise RuntimeError(f"Furhat voice {self._voice!r} not available; choose one of {voices[:10]}...")
            self._api().set_voice(name=self._voice)

    def _led(self, state: str) -> None:
        if self._expressions:
            r, g, b = LED[state]
            try:
                self._api().set_led(red=r, green=g, blue=b)
            except Exception as e:
                log.warning("Furhat LED failed: %s", e)

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        self._done.clear()
        gesture = self._gestures.get(expression_key(cue)) if self._expressions else None
        error = {}

        def worker():
            try:
                if gesture:
                    self._api().gesture(name=gesture, blocking=False)
                self._api().say(text=text, blocking=True)
            except Exception as e:
                error["e"] = e
            finally:
                self._done.set()

        threading.Thread(target=worker, daemon=True, name="furhat-say").start()
        while not self._done.wait(0.05):
            if self._stop.is_set():
                return False          # control returns to RAWR immediately
        if "e" in error:
            raise RuntimeError(f"Furhat say failed: {error['e']}")
        return not self._stop.is_set()

    def stop(self) -> bool:
        self._stop.set()
        try:
            self._api().say_stop()
            return True
        except Exception as e:
            log.warning("Furhat say_stop failed: %s", e)
            return False

    def on_listening(self) -> None:
        self._led("listening")
        if self._expressions:
            try:
                self._api().attend(user="CLOSEST")
            except Exception:
                pass

    def on_thinking(self) -> None:
        self._led("thinking")

    def on_idle(self) -> None:
        self._led("idle")
