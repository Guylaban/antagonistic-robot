"""Audio capture with Silero VAD for speech endpoint detection.

Records from the laptop microphone and uses Silero VAD to detect when
the user starts and stops speaking. The record_utterance method blocks
until a complete utterance is captured.

Recording is done at 16kHz, 16-bit, mono — the format faster-whisper expects.
"""

import time
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import torch
from typing import Callable, Optional

from antagonist_robot.config.settings import AudioConfig
from antagonist_robot.pipeline.types import AudioData


class AudioCapture:
    """Records a single utterance using VAD-based endpoint detection.

    Uses Silero VAD to detect speech start and end. Blocks until the user
    has spoken and then gone silent for longer than the configured threshold.
    """

    def __init__(self, config: AudioConfig):
        self.sample_rate = config.sample_rate
        self.silence_threshold_ms = config.silence_threshold_ms
        self.min_speech_duration_ms = config.min_speech_duration_ms

        # Load Silero VAD model once (bundled with the silero-vad package,
        # so no download from GitHub is needed at startup)
        from silero_vad import load_silero_vad
        self._vad_model = load_silero_vad()
        self._vad_model.eval()

        # Frame size for VAD: 512 samples = 32ms at 16kHz
        # Silero VAD supports 256, 512, or 768 samples at 16kHz
        self._frame_size = 512

    def record_utterance(self, is_active: Optional[Callable[[], bool]] = None) -> Optional[AudioData]:
        """Block until user speaks and goes silent. Return recorded audio.

        Flow:
        1. Continuously read microphone frames
        2. Pass each frame through Silero VAD
        3. Wait for VAD to indicate speech has started
        4. Keep recording while speech continues
        5. When silence exceeds threshold, stop and return
        6. If speech is shorter than min_duration, discard and keep listening

            AudioData with the captured utterance.
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

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._frame_size,
            ) as stream:
                while is_active():
                    frame, _ = stream.read(self._frame_size)
                    frame_1d = frame[:, 0]  # mono channel

                    # Run VAD on this frame
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
        # Silero VAD __call__ returns a probability tensor
        speech_prob = self._vad_model(tensor, self.sample_rate).item()
        return speech_prob > 0.5


if __name__ == "__main__":
    # Standalone test: record one utterance and print info
    config = AudioConfig()
    capture = AudioCapture(config)
    print("Speak now (will detect when you stop)...")
    audio = capture.record_utterance()
    print(f"Recorded {audio.duration_seconds:.2f}s of audio")
    print(f"  Sample rate: {audio.sample_rate}")
    print(f"  Samples: {len(audio.samples)}")
    print(f"  Started: {audio.recording_started}")
    print(f"  Ended: {audio.recording_ended}")
