"""Robot backend interface: the only way CRAB talks to a robot.

A backend speaks released replies, can interrupt its own speech, listens
through the robot, and reports what it can do. The conversation manager calls:

    connect()                        once at startup; raises RuntimeError if the robot is unreachable
    on_listening() / on_thinking()   state cues while the participant speaks / a reply is prepared
    speak(text, cue)                 blocks until speech ends; returns False if stop() interrupted it
    stop()                           interrupt speech now (operator Stop speech / End session)
    on_idle()                        session over
    close()                          release the connection

Listening goes through the robot too. A backend provides one of:
    mic_source()   the robot's microphone as an audio source; CRAB runs VAD + ASR locally
    recognizer()   the robot's own speech recognition (record_utterance / transcribe)

Non-verbal cues (`robot.expressions`, on by default) follow the reply
being spoken: the expression comes from the category the fidelity judge
finds in the reply (or, without the judge, the requested category), and
its strength from the antagonism level, so voice and body express the
same thing. A reply the judge finds neutral gets neutral body language
even under an antagonistic condition. Switch cues off for a study whose
manipulation must be verbal only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeechCue:
    """The reply being spoken, for non-verbal cues: the requested condition and, when the fidelity
    judge scored the reply, the category (B-G, NEUTRAL, REFUSAL) and intensity (0-3) it actually shows."""
    polar_level: int = 0
    category: Optional[str] = None
    subtype: int = 1
    modifiers: list = field(default_factory=list)
    exhibited_category: Optional[str] = None
    exhibited_intensity: Optional[int] = None


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
    """Expression for the reply being spoken: support, neutral, or a category letter B-G.

    Uses the category the judge found in the reply when there is one (NEUTRAL or REFUSAL -> neutral),
    otherwise the requested category.
    """
    if cue is None or cue.polar_level == 0:
        return "neutral"
    if cue.polar_level < 0:
        return "support"
    shown = (cue.exhibited_category or "").upper()
    if shown in ("B", "C", "D", "E", "F", "G"):
        return shown
    if shown in ("NEUTRAL", "REFUSAL"):
        return "neutral"
    return cue.category or "D"


def expression_strength(cue: Optional[SpeechCue]) -> float:
    """How strongly to express the cue, 0-1: the level of antagonism (or support) in the reply.

    Antagonism: the mean of the intensity (the judge's enacted intensity, else the requested
    intensity class, 1-3) and the polar level (1-3), each as a fraction of 3. Support: |polar| / 3.
    Neutral: 0.
    """
    key = expression_key(cue)
    if key == "neutral":
        return 0.0
    if key == "support":
        return min(1.0, abs(cue.polar_level) / 3)
    intensity = cue.exhibited_intensity if cue.exhibited_intensity is not None else cue.subtype
    intensity = max(1, min(3, int(intensity or 1)))
    return round((intensity / 3 + min(3, cue.polar_level) / 3) / 2, 3)


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
