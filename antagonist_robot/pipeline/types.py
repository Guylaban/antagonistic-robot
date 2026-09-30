"""Data types passed between pipeline components.

Every pipeline stage returns a dataclass that includes timing information
for latency logging. These are the only data structures that flow between
pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class AudioData:
    """Recorded audio from the microphone."""
    samples: np.ndarray          # float32 numpy array, normalized [-1, 1]
    sample_rate: int             # always 16000
    duration_seconds: float      # len(samples) / sample_rate
    recording_started: str       # ISO-format timestamp
    recording_ended: str         # ISO-format timestamp


@dataclass
class ASRResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str
    confidence: float            # average log probability from segments
    transcription_time_seconds: float


@dataclass
class LLMResult:
    """Result from LLM generation."""
    text: str
    model: str
    total_tokens: int
    generation_time_seconds: float
    reasoning: Optional[str] = None  # reasoning trace, if the provider returns one


@dataclass
class TurnResult:
    """Complete result for one conversation turn."""
    turn_number: int
    user_audio: Optional[AudioData]
    transcript: str
    llm_response: str
    polar_level: int             # polar level of the response that was spoken
    category: str
    subtype: int
    modifiers: list
    risk_rating: str             # max(content_risk, config_risk) of the spoken response
    latency: Dict[str, int]      # vad_ms, asr_ms, llm_ms, review_ms, speech_ms, total_ms
    timestamp: str               # ISO-format
    requested_polar_level: int = 0   # session polar level when the turn started
    content_risk: str = "Green"
    config_risk: str = "Green"
    candidate_id: Optional[int] = None
    n_candidates: int = 1
    operator_action: str = ""    # send | auto_send
    decided_by: str = ""         # operator | timer
    participant_distress: list = field(default_factory=list)
