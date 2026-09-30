"""Robot backend interface: the only way RAWR talks to a robot.

A backend speaks released replies, can interrupt its own speech, listens
through the robot, and reports what it can do. The conversation manager calls:

    connect()                        once at startup; raises RuntimeError if the robot is unreachable
    on_listening() / on_thinking()   state cues while the participant speaks / a reply is prepared
    speak(text, cue)                 blocks until speech ends; returns False if stop() interrupted it
    stop()                           interrupt speech now (operator Stop speech / End session)
    on_idle()                        session over
    close()                          release the connection

Listening goes through the robot too. A backend provides one of:
    mic_source()   the robot's microphone as an audio source; RAWR runs VAD + ASR locally
    recognizer()   the robot's own speech recognition (record_utterance / transcribe)

Optional non-verbal cues (`expressions: true` in the backend's config
section) are chosen from the behavioral condition of the reply being
spoken. They are off by default so that, across robots, the manipulation
stays verbal unless a study deliberately adds a non-verbal channel.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeechCue:
    """Behavioral condition of the reply being spoken (for optional non-verbal cues)."""
    polar_level: int = 0
    category: Optional[str] = None
    subtype: int = 1
    modifiers: list = field(default_factory=list)


@dataclass
class Capabilities:
    """What a backend can do; shown in the console and stored with each session."""
    robot: str
    speech: str                 # "robot TTS" or "computer TTS on robot speaker"
    interrupt: str              # "verified", "unverified", or "none"
    expressions: bool           # non-verbal cues enabled
    listening: str = "none"     # how participant speech is captured
    notes: str = ""

    def as_dict(self) -> dict:
        return {"robot": self.robot, "speech": self.speech, "interrupt": self.interrupt,
                "expressions": self.expressions, "listening": self.listening, "notes": self.notes}


def expression_key(cue: Optional[SpeechCue]) -> str:
    """Map a condition to an expression key: support, neutral, or a category letter B-G."""
    if cue is None or cue.polar_level == 0:
        return "neutral"
    if cue.polar_level < 0:
        return "support"
    return cue.category or "D"


class RobotBackend(ABC):
    """Abstract robot backend."""

    capabilities: Capabilities

    @abstractmethod
    def connect(self) -> None:
        """Verify the robot answers; raise RuntimeError with a fix-it message if not."""

    @abstractmethod
    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        """Speak text; block until done. Return False if interrupted by stop()."""

    @abstractmethod
    def stop(self) -> bool:
        """Interrupt current speech. Return True if a stop was issued."""

    def mic_source(self):
        """Factory for the robot's microphone as a frame source, or None."""
        return None

    def recognizer(self):
        """The robot's own speech recognizer (record_utterance/transcribe), or None."""
        return None

    def on_listening(self) -> None:
        """The participant may speak now."""

    def on_thinking(self) -> None:
        """A reply is being generated or reviewed."""

    def on_idle(self) -> None:
        """The session is over."""

    def close(self) -> None:
        """Release the connection."""
