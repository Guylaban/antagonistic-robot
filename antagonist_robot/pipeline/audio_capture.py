"""Audio capture with Silero VAD for speech endpoint detection.

Reads 16 kHz mono audio from a frame source and uses Silero VAD to detect
when the participant starts and stops speaking. record_utterance() blocks
until a complete utterance is captured.

Frame sources (context managers with read(n) -> float32 array of n samples):
    the robot's microphones   robots/*: NAO/Pepper stream, Reachy Mini SDK (default)
    ComputerMic               the computer's default input device (fallback: audio.input: computer)

A new source is opened for every utterance, so audio recorded while the
robot was speaking is never used.
"""

import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import torch

from antagonist_robot.config.settings import AudioConfig
from antagonist_robot.pipeline.types import AudioData


class ComputerMic:
    """The computer's default input device via sounddevice."""

    def __init__(self, sample_rate: int = 16000, blocksize: int = 512):
        self._sr, self._block = sample_rate, blocksize
        self._stream = None

    def __enter__(self):
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=self._sr, channels=1, dtype="float32", blocksize=self._block)
        self._stream.start()
        return self

    def read(self, n: int) -> np.ndarray:
        frame, _ = self._stream.read(n)
        return frame[:, 0]

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()
        return False


class AudioCapture:
    """Records a single utterance using VAD-based endpoint detection.

    Uses Silero VAD to detect speech start and end. Blocks until the user
    has spoken and then gone silent for longer than the configured threshold.
    """

    def __init__(self, config: AudioConfig, source_factory: Optional[Callable[[], object]] = None,
                 source_name: str = "computer microphone"):
        self.sample_rate = config.sample_rate
        self.silence_threshold_ms = config.silence_threshold_ms
        self.min_speech_duration_ms = config.min_speech_duration_ms
        self.source_name = source_name
        self._source_factory = source_factory or (lambda: ComputerMic(self.sample_rate, 512))

        # Load Silero VAD model once (bundled with the silero-vad package,
        # so no download from GitHub is needed at startup)
        from silero_vad import load_silero_vad
        self._vad_model = load_silero_vad()
        self._vad_model.eval()

        # Frame size for VAD: 512 samples = 32ms at 16kHz
        # Silero VAD supports 256, 512, or 768 samples at 16kHz
        self._frame_size = 512

    def record_utterance(self, is_active: Optional[Callable[[], bool]] = None) -> Optional[AudioData]:
        """Block until the participant speaks and goes silent. Return the recorded audio.

        Flow:
        1. Continuously read frames from the source
        2. Pass each frame through Silero VAD
        3. Wait for VAD to indicate speech has started
        4. Keep recording while speech continues
        5. When silence exceeds threshold, stop and return
        6. If speech is shorter than min_duration, discard and keep listening
        """
        if is_active is None:
            is_active = lambda: True

        while is_active():  # Outer loop handles too-short utterances
            recording_started = datetime.now(timezone.utc).isoformat()
            speech_frames: list[np.ndarray] = []
            is_speaking = False
            silence_start: float | None = None

            # Reset VAD state for a fresh detection
            self._vad_model.reset_states()

            with self._source_factory() as source:
                while is_active():
                    frame_1d = np.asarray(source.read(self._frame_size), dtype=np.float32)

                    has_speech = self._check_speech(frame_1d)

                    if has_speech:
                        is_speaking = True
                        silence_start = None
                        speech_frames.append(frame_1d.copy())
                    elif is_speaking:
                        # Speech was happening, now we have silence
                        speech_frames.append(frame_1d.copy())
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elapsed_silence_ms = (time.monotonic() - silence_start) * 1000
                        if elapsed_silence_ms >= self.silence_threshold_ms:
                            break  # End of utterance detected

            if not is_active():
                return None

            recording_ended = datetime.now(timezone.utc).isoformat()

            if not speech_frames:
                continue

            samples = np.concatenate(speech_frames)
            duration_seconds = len(samples) / self.sample_rate
            duration_ms = duration_seconds * 1000

            # Ignore utterances shorter than the minimum (coughs, noise)
            if duration_ms < self.min_speech_duration_ms:
                continue

            return AudioData(
                samples=samples,
                sample_rate=self.sample_rate,
                duration_seconds=duration_seconds,
                recording_started=recording_started,
                recording_ended=recording_ended,
            )

    def _check_speech(self, frame: np.ndarray) -> bool:
        """Run Silero VAD on a single frame and return True if speech detected."""
        tensor = torch.from_numpy(frame).float()
        speech_prob = self._vad_model(tensor, self.sample_rate).item()
        return speech_prob > 0.5


if __name__ == "__main__":
    # Standalone test: record one utterance from the computer microphone
    config = AudioConfig()
    capture = AudioCapture(config)
    print("Speak now (will detect when you stop)...")
    audio = capture.record_utterance()
    print(f"Recorded {audio.duration_seconds:.2f}s of audio")
