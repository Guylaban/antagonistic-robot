"""Scripted participant for dry runs, operator training, and demos.

Replaces the microphone and speech recognition with a fixed list of
participant utterances (a YAML file with a top-level "utterances" list).
Everything after ASR is the production path: prompt compilation, LLM,
safety rating, operator review, robot speech, and logging. Used by
`python main.py --script FILE` (with tools/mock_nao.py for a robot-free dry run).
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import yaml

from antagonist_robot.pipeline.types import ASRResult, AudioData


def load_script(path: str) -> List:
    """Read the utterance list from a YAML script file.

    Each entry is a string, or {"text": ..., "delay_s": ...} to set how long the
    participant 'speaks' before the utterance is delivered (e.g. its audio duration).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    utterances = data.get("utterances", [])
    if not utterances:
        raise ValueError(f"{path} has no 'utterances' list")
    return [u if isinstance(u, dict) else str(u) for u in utterances]


class ScriptedParticipant:
    """Stands in for AudioCapture and ASREngine with scripted utterances."""

    def __init__(self, utterances: List, delay_s: float = 1.5):
        self._utterances = [u["text"] if isinstance(u, dict) else u for u in utterances]
        self._delays = [float(u.get("delay_s", delay_s)) if isinstance(u, dict) else delay_s for u in utterances]
        self._delay_s = delay_s
        self._next = 0
        self._pending: Optional[str] = None

    @property
    def exhausted(self) -> bool:
        return self._next >= len(self._utterances)

    def record_utterance(self, is_active: Optional[Callable[[], bool]] = None) -> Optional[AudioData]:
        """Wait delay_s (the participant 'speaking'), then return the next utterance.

        When the script is exhausted, waits until the session ends.
        """
        is_active = is_active or (lambda: True)
        started = datetime.now(timezone.utc).isoformat()
        delay = self._delays[self._next] if not self.exhausted else self._delay_s
        deadline = time.monotonic() + delay
        while is_active() and (time.monotonic() < deadline or self.exhausted):
            time.sleep(0.05)
        if not is_active():
            return None
        self._pending = self._utterances[self._next]
        self._next += 1
        return AudioData(
            samples=np.zeros(0, dtype=np.float32), sample_rate=16000, duration_seconds=0.0,
            recording_started=started, recording_ended=datetime.now(timezone.utc).isoformat(),
        )

    def transcribe(self, audio: AudioData) -> ASRResult:
        text, self._pending = self._pending or "", None
        return ASRResult(text=text, language="en", confidence=0.0, transcription_time_seconds=0.0)
