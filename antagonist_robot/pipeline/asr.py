"""Single-shot ASR using faster-whisper (CTranslate2).

Takes a complete AudioData from the capture module and returns
the full transcription. This is NOT streaming — the full recorded
audio goes in, the full text comes out.
"""

import time

from faster_whisper import WhisperModel

from antagonist_robot.config.settings import ASRConfig
from antagonist_robot.pipeline.types import AudioData, ASRResult


class ASREngine:
    """Single-shot speech-to-text using faster-whisper.

    Loads the Whisper model once at initialization. The transcribe method
    accepts an AudioData dataclass and returns an ASRResult.
    """

    def __init__(self, config: ASRConfig):
        device = config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
            # The model is public; don't send a stored (possibly expired)
            # Hugging Face token, which makes the download fail with 401.
            use_auth_token=False,
        )

    def transcribe(self, audio: AudioData) -> ASRResult:
        """Transcribe audio to text. Blocks until complete.

        Args:
            audio: AudioData with float32 samples at 16kHz mono.

        Returns:
            ASRResult with transcribed text, language, confidence, and timing.
        """
        start = time.monotonic()

        segments, info = self._model.transcribe(
            audio.samples,
            language="en",
            beam_size=1,
            vad_filter=False,  # VAD already done in the capture stage
        )

        # Collect all segments
        texts: list[str] = []
        log_probs: list[float] = []
        for segment in segments:
            texts.append(segment.text)
            log_probs.append(segment.avg_logprob)

        text = " ".join(texts).strip()
        avg_confidence = sum(log_probs) / len(log_probs) if log_probs else 0.0
        elapsed = time.monotonic() - start

        return ASRResult(
            text=text,
            language=info.language,
            confidence=avg_confidence,
            transcription_time_seconds=elapsed,
        )
