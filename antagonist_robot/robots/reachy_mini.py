"""Reachy Mini backend (Pollen Robotics / Hugging Face, pip package reachy-mini).

Reachy Mini has no built-in text-to-speech, so replies are synthesized
offline on the computer (SAPI on Windows, espeak-ng on Linux) and
streamed to the robot's speaker through the SDK in 0.1 s chunks, which
makes stop() take effect within one chunk. Listening uses the robot's
microphones through the SDK (RAWR runs VAD and ASR locally). Optional
non-verbal cues use head pose and antennas (goto_target).

Works with the physical robot and with the MuJoCo simulation
(`reachy-mini-daemon --sim`; the simulation uses the computer's default
microphone and speaker in place of the robot's). The daemon's HTTP port defaults to 8000,
the same as RAWR's console, so run the console on another port
(server.port in config.yaml) when both are on one computer.
"""

import logging
import os
import queue
import tempfile
import threading
import time
import wave
from typing import Optional

import numpy as np

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key

log = logging.getLogger(__name__)

CHUNK_S = 0.1

# Head pose (roll, pitch, yaw in degrees) and antenna angles (degrees) per condition;
# used only when expressions are on. Adjust for your study.
DEFAULT_POSES = {
    "neutral": ((0, 0, 0), (0, 0)),
    "support": ((0, -5, 0), (25, 25)),
    "B": ((0, 0, 25), (-20, -20)),
    "C": ((12, 0, 0), (25, -15)),
    "D": ((0, -8, 0), (-10, -10)),
    "E": ((8, 0, 0), (10, -10)),
    "F": ((0, 8, 0), (-35, -35)),
    "G": ((0, 8, 0), (-40, -40)),
    "listening": ((0, 0, 0), (10, 10)),
    "thinking": ((6, -4, 0), (0, 15)),
}


class _TTSWorker:
    """Offline text-to-speech to a WAV file, on one dedicated thread.

    Windows: SAPI through comtypes (COM objects stay on the thread that
    created them). Linux/macOS: the espeak-ng (or espeak) command line.
    pyttsx3 is not used: its runAndWait() hangs or writes empty files on
    repeated calls.
    """

    def __init__(self, rate: Optional[int], voice: Optional[str]):
        self._jobs: queue.Queue = queue.Queue()
        self._rate, self._voice = rate, voice
        self._error: Optional[str] = None
        threading.Thread(target=self._run, daemon=True, name="reachy-tts").start()

    def _run(self):
        import sys
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
            raise RuntimeError("install espeak-ng for Reachy Mini speech (e.g. sudo apt install espeak-ng)")

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
            raise RuntimeError(f"Reachy Mini TTS failed: {self._error or 'timeout'}")
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


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampling (speech-quality, no extra dependency)."""
    if sr_in == sr_out or len(samples) == 0:
        return samples.astype(np.float32)
    n_out = int(round(len(samples) * sr_out / sr_in))
    x_out = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_out, np.arange(len(samples)), samples).astype(np.float32)


class ReachyMicSource:
    """Reachy Mini's microphones through the SDK (mixed to mono, resampled to 16 kHz)."""

    def __init__(self, media, sample_rate: int = 16000):
        self._media, self._sr = media, sample_rate
        self._buf = np.zeros(0, dtype=np.float32)

    def __enter__(self):
        self._media.start_recording()
        self._sr_in = self._media.get_input_audio_samplerate()
        self._buf = np.zeros(0, dtype=np.float32)
        return self

    def read(self, n: int) -> np.ndarray:
        deadline = time.monotonic() + 3.0
        while len(self._buf) < n:
            chunk = self._media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                if time.monotonic() > deadline:
                    return np.zeros(n, dtype=np.float32)   # no audio: treat as silence
                time.sleep(0.01)
                continue
            chunk = np.asarray(chunk, dtype=np.float32)
            mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
            self._buf = np.concatenate([self._buf, resample(mono, self._sr_in, self._sr)])
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def __exit__(self, *exc):
        self._media.stop_recording()
        return False


class ReachyMiniBackend(RobotBackend):
    """Reachy Mini (physical or MuJoCo simulation) through the reachy-mini SDK."""

    def __init__(self, host: str = "localhost", port: int = 8000, connection_mode: str = "auto",
                 tts_rate: Optional[int] = 175, tts_voice: Optional[str] = None, expressions: bool = False,
                 poses: Optional[dict] = None, mini=None, tts=None):
        self._host, self._port, self._mode = host, port, connection_mode
        self._expressions = expressions
        self._poses = {**DEFAULT_POSES, **(poses or {})}
        self._mini = mini
        self._tts = tts
        self._tts_args = (tts_rate, tts_voice)
        self._stop = threading.Event()
        self._playing = False
        self.capabilities = Capabilities(
            robot="Reachy Mini", speech="computer TTS (SAPI / espeak-ng) on robot speaker", interrupt="verified",
            expressions=expressions, listening="robot microphones through the SDK, to local VAD + ASR",
            notes="speech streamed in 0.1 s chunks; stop takes effect within one chunk",
        )

    def mic_source(self):
        return lambda: ReachyMicSource(self._mini.media)

    def connect(self) -> None:
        if self._mini is None:
            try:
                from reachy_mini import ReachyMini  # optional dependency
                self._mini = ReachyMini(host=self._host, port=self._port, connection_mode=self._mode, timeout=15)
                self._mini.__enter__()
            except Exception as e:
                raise RuntimeError(
                    f"Reachy Mini daemon not reachable at {self._host}:{self._port} ({e}). Start the robot, or the "
                    f"simulation with: reachy-mini-daemon --sim"
                ) from e
        if self._tts is None:
            self._tts = _TTSWorker(*self._tts_args)
        self._pose("neutral", 0.5, force=True)

    def _pose(self, key: str, duration: float = 0.6, force: bool = False) -> None:
        if not (self._expressions or force) or key not in self._poses:
            return
        from reachy_mini.utils import create_head_pose
        (roll, pitch, yaw), (a_left, a_right) = self._poses[key]
        try:
            self._mini.goto_target(head=create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True),
                                   antennas=np.deg2rad([a_left, a_right]), duration=duration)
        except Exception as e:
            log.warning("Reachy Mini pose %s failed: %s", key, e)

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        samples, sr_in = self._tts.synthesize(text)
        media = self._mini.media
        sr_out, channels = media.get_output_audio_samplerate(), media.get_output_channels()
        audio = resample(samples, sr_in, sr_out).reshape(-1, 1)
        if channels > 1:
            audio = np.repeat(audio, channels, axis=1)
        self._pose(expression_key(cue), 0.4)
        media.start_playing()
        self._playing = True
        step = int(sr_out * CHUNK_S)
        t0 = time.monotonic()
        try:
            for i, start in enumerate(range(0, len(audio), step)):
                if self._stop.is_set():
                    return False
                media.push_audio_sample(audio[start:start + step])
                # push_audio_sample is non-blocking: pace pushes in real time
                delay = t0 + (i + 1) * CHUNK_S - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            return not self._stop.is_set()
        finally:
            media.stop_playing()
            self._playing = False
            self._pose("neutral", 0.4)

    def stop(self) -> bool:
        self._stop.set()
        return True

    def on_listening(self) -> None:
        self._pose("listening", 0.5)

    def on_thinking(self) -> None:
        self._pose("thinking", 0.5)

    def on_idle(self) -> None:
        self._pose("neutral", 0.5)

    def close(self) -> None:
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                pass
