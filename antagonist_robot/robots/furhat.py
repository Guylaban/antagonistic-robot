"""Furhat backend via the Furhat Remote API (pip package furhat-remote-api).

Requires the Remote API skill running on the robot or on the virtual
Furhat of the Furhat SDK (it listens on port 54321). Speech uses the
robot's own voice (`say`); listening uses the robot's microphones and its
own speech recognition (`listen`), which returns text only. Optional
non-verbal cues use Furhat gestures and the LED ring.

Interruption: stop() sends `say_stop` and returns control to RAWR at once,
so the session never waits on the robot. In our tests with the virtual
Furhat of SDK 2.9.2, `say_stop` was acknowledged but did not cut the
audio short (the next utterance started only after the previous one
ended), so interruption is reported as "unverified" for this backend.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from antagonist_robot.pipeline.types import ASRResult, AudioData
from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key

log = logging.getLogger(__name__)

# listen() returns these instead of user speech
_NO_SPEECH = {"SILENCE", "INTERRUPTED", "FAILED", ""}


class FurhatRecognizer:
    """Listening through Furhat's own microphones and speech recognition (Remote API listen()).

    Furhat's recognizer runs on the robot's side (a cloud ASR service), so no
    raw audio reaches RAWR and none is archived; only the transcript is logged.
    """

    def __init__(self, api: Callable, language: str = "en-US"):
        self._api = api
        self._language = language
        self._text: Optional[str] = None

    def record_utterance(self, is_active: Optional[Callable[[], bool]] = None) -> Optional[AudioData]:
        is_active = is_active or (lambda: True)
        while is_active():
            started = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            try:
                status = self._api().listen(language=self._language)
                text = (getattr(status, "message", "") or "").strip()
            except Exception as e:
                log.warning("Furhat listen failed: %s", e)
                time.sleep(0.5)
                continue
            if not is_active():
                return None
            if text.upper() in _NO_SPEECH:
                continue
            self._text = text
            self._elapsed = time.monotonic() - t0
            return AudioData(samples=np.zeros(0, dtype=np.float32), sample_rate=16000, duration_seconds=0.0,
                             recording_started=started, recording_ended=datetime.now(timezone.utc).isoformat())
        return None

    def transcribe(self, audio: AudioData) -> ASRResult:
        text, self._text = self._text or "", None
        return ASRResult(text=text, language=self._language, confidence=0.0, transcription_time_seconds=0.0)

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
                 gestures: Optional[dict] = None, client=None, language: str = "en-US"):
        self._host = host
        self._voice = voice
        self._language = language
        self._expressions = expressions
        self._gestures = {**DEFAULT_GESTURES, **(gestures or {})}
        self._client = client
        self._stop = threading.Event()
        self._done = threading.Event()
        self.capabilities = Capabilities(
            robot="Furhat", speech="robot TTS (Furhat voice)", interrupt="unverified", expressions=expressions,
            listening="robot microphones and Furhat's own (cloud) speech recognition; no audio archived",
            notes="say_stop did not interrupt audio on the virtual Furhat (SDK 2.9.2); verify on your robot",
        )

    def recognizer(self):
        return FurhatRecognizer(self._api, self._language)

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
        try:
            self._api().listen_stop()   # release a pending listen() when the session ends
        except Exception:
            pass
