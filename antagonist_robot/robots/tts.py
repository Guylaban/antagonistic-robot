"""Text-to-speech engines for backends that speak audio produced on the computer.

    system   SAPI on Windows, espeak-ng on Linux/macOS (offline, no download; robotic)
    kokoro   Kokoro-82M neural TTS (offline after a one-time ~330 MB download; natural
             prosody). Runs on the GPU when available. pip install kokoro (on Python 3.13:
             pip install --no-deps --ignore-requires-python kokoro misaki, then
             pip install loguru num2words spacy phonemizer-fork espeakng-loader addict)

Each engine yields audio sentence by sentence (stream), so playback can start after
the first sentence instead of after the whole reply.
"""

import logging
import os
import queue
import sys
import tempfile
import threading
import wave
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

ENGINES = ("system", "kokoro")
_SENTENCE = r"(?<=[.!?;:])\s+"


class SystemTTS:
    """Offline text-to-speech to a WAV file, on one dedicated thread.

    Windows: SAPI through comtypes (COM objects stay on the thread that
    created them). Linux/macOS: the espeak-ng (or espeak) command line.
    pyttsx3 is not used: its runAndWait() hangs or writes empty files on
    repeated calls.
    """

    def __init__(self, rate: Optional[int] = None, voice: Optional[str] = None):
        self._jobs: queue.Queue = queue.Queue()
        self._rate, self._voice = rate, voice
        self._error: Optional[str] = None
        threading.Thread(target=self._run, daemon=True, name="system-tts").start()

    def _run(self):
        synth = self._sapi() if sys.platform == "win32" else self._espeak()
        while True:
            text, path, done = self._jobs.get()
            try:
                synth(text, path)
            except Exception as e:
                self._error = str(e)
            finally:
                done.set()

    def _sapi(self):
        import comtypes
        import comtypes.client
        comtypes.CoInitialize()
        voice = comtypes.client.CreateObject("SAPI.SpVoice")
        if self._rate:  # SAPI rate is -10..10 (0 is about 180 words per minute)
            voice.Rate = max(-10, min(10, round((self._rate - 180) / 18)))
        if self._voice:
            tokens = voice.GetVoices()
            for i in range(tokens.Count):
                if self._voice.lower() in tokens.Item(i).GetDescription().lower():
                    voice.Voice = tokens.Item(i)
                    break

        def synth(text, path):
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            stream.Open(path, 3)  # SSFMCreateForWrite
            voice.AudioOutputStream = stream
            voice.Speak(text)
            stream.Close()
        return synth

    def _espeak(self):
        import shutil
        import subprocess
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            raise RuntimeError("install espeak-ng for computer-side speech (e.g. sudo apt install espeak-ng)")

        def synth(text, path):
            cmd = [exe, "-w", path] + (["-s", str(self._rate)] if self._rate else []) \
                + (["-v", self._voice] if self._voice else []) + [text]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return synth

    def synthesize(self, text: str) -> tuple:
        """Return (float32 mono samples, sample rate)."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        done = threading.Event()
        self._error = None
        self._jobs.put((text, path, done))
        if not done.wait(30) or self._error:
            os.remove(path)
            raise RuntimeError(f"TTS failed: {self._error or 'timeout'}")
        try:
            with wave.open(path) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
        finally:
            os.remove(path)
        data = np.frombuffer(raw, dtype={1: np.int8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
        data /= float(2 ** (8 * width - 1))
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        return data, sr

    def stream(self, text: str) -> Iterator[tuple]:
        yield self.synthesize(text)

    def warm_up(self) -> None:
        pass


class KokoroTTS:
    """Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) neural TTS, 24 kHz, on the GPU when available.

    voice: a Kokoro voice id; its first letter sets the accent (a = American, b = British),
    e.g. af_heart, af_bella, am_michael, bf_emma, bm_george.
    """

    SR = 24000
    streams_sentences = True    # stream() yields one part per sentence (split on _SENTENCE)

    def __init__(self, voice: Optional[str] = "af_heart", rate: Optional[int] = None, device: str = "auto",
                 sentence_pause_s: float = 0.12):
        self._voice = voice or "af_heart"
        self._speed = (rate / 175.0) if rate else 1.0          # tts_rate in words per minute; 175 = normal
        self._device = device
        self._pause = np.zeros(int(self.SR * sentence_pause_s), dtype=np.float32)
        self._pipe = None
        self._lock = threading.Lock()

    def _pipeline(self):
        if self._pipe is None:
            try:
                import torch
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError("tts_engine 'kokoro' needs the kokoro package (see antagonist_robot/robots/tts.py)") from e
            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe = KPipeline(lang_code=self._voice[0], repo_id="hexgrad/Kokoro-82M", device=device)
            log.info("Kokoro TTS loaded on %s (voice %s)", device, self._voice)
        return self._pipe

    def stream(self, text: str) -> Iterator[tuple]:
        with self._lock:
            pipe = self._pipeline()
            first = True
            for _, _, audio in pipe(text, voice=self._voice, speed=self._speed, split_pattern=_SENTENCE):
                if audio is None:
                    continue
                a = audio.detach().cpu().numpy().astype(np.float32) if hasattr(audio, "detach") else np.asarray(audio, np.float32)
                yield (a if first else np.concatenate([self._pause, a])), self.SR
                first = False

    def synthesize(self, text: str) -> tuple:
        parts = [a for a, _ in self.stream(text)]
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), self.SR

    def warm_up(self) -> None:
        """Load the model and run it once, so the first reply is not delayed by loading."""
        for _ in self.stream("Hello."):
            pass


def create_tts(engine: str = "system", rate: Optional[int] = None, voice: Optional[str] = None, device: str = "auto"):
    if engine == "system":
        return SystemTTS(rate, voice)
    if engine == "kokoro":
        return KokoroTTS(voice, rate, device)
    raise ValueError(f"tts_engine must be one of {ENGINES}, got {engine!r}")


def to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()
