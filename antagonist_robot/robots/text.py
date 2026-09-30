"""Text backend: prints replies instead of speaking them (dry runs, operator training)."""

import threading
import time
from typing import Optional

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue

SECONDS_PER_WORD = 0.3   # simulated speaking time, similar to NAO at speed 85


class TextBackend(RobotBackend):
    """No robot: each reply is printed and 'spoken' for a length-proportional time."""

    def __init__(self, seconds_per_word: float = SECONDS_PER_WORD):
        self._spw = seconds_per_word
        self._stop = threading.Event()
        self.capabilities = Capabilities(robot="none (text)", speech="printed to the terminal",
                                         interrupt="verified", expressions=False,
                                         listening="none (use --script, or audio.input: computer)")

    def connect(self) -> None:
        pass

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        print(f"[ROBOT] {text}", flush=True)
        return not self._stop.wait(self._spw * max(1, len(text.split())))

    def stop(self) -> bool:
        self._stop.set()
        return True
