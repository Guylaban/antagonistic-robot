"""Reachy Mini backend (Pollen Robotics / Hugging Face, pip package reachy-mini).

Reachy Mini has no built-in text-to-speech, so replies are synthesized
offline on the computer (tts_engine: "system" = SAPI / espeak-ng, or
"kokoro" = neural TTS, see robots/tts.py) and streamed to the robot's
speaker through the SDK in 0.1 s chunks, sentence by sentence, which
makes stop() take effect within one chunk. Listening uses the robot's
microphones through the SDK (RAWR runs VAD and ASR locally).

Optional non-verbal cues (expressions: true) use the head and antennas.
With animate: true (the default when expressions are on) the robot moves
continuously: it eases into a pose per state and condition, breathes and
glances while listening, and moves its head and antennas with the
loudness of its own speech (nods on stressed syllables), in a style
chosen from the condition. With animate: false it only switches between
fixed poses.

Works with the physical robot and with the MuJoCo simulation
(`reachy-mini-daemon --sim`; the simulation uses the computer's default
microphone and speaker in place of the robot's). The daemon's HTTP port defaults to 8000,
the same as RAWR's console, so run the console on another port
(server.port in config.yaml) when both are on one computer.
"""

import collections
import logging
import math
import queue
import threading
import time
import wave
from typing import Optional

import numpy as np

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key
from antagonist_robot.robots.tts import SystemTTS, create_tts

log = logging.getLogger(__name__)

CHUNK_S = 0.1
_TTSWorker = SystemTTS          # earlier name, kept for scripts that import it

# Head pose (roll, pitch, yaw in degrees; positive pitch looks down) and antenna angles
# (degrees) per state and condition; used only when expressions are on. Adjust for your study.
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

# Motion while speaking, per condition (animate: true). Degrees. nod: head dips with loudness;
# beat: quick dip on stressed syllables; sway: slow roll/yaw drift; antenna: flutter with speech;
# tempo: speed of sway and flutter; mirror: antennas move in opposition (asymmetric, "sarcastic").
DEFAULT_STYLES = {
    "neutral": dict(nod=3.0, beat=4.0, sway=2.5, antenna=10.0, tempo=1.0, mirror=False),
    "support": dict(nod=3.0, beat=3.0, sway=3.5, antenna=16.0, tempo=0.9, mirror=False),
    "B": dict(nod=1.5, beat=2.0, sway=1.5, antenna=5.0, tempo=0.7, mirror=False),
    "C": dict(nod=2.5, beat=4.0, sway=4.5, antenna=18.0, tempo=0.8, mirror=True),
    "D": dict(nod=4.5, beat=7.0, sway=2.0, antenna=7.0, tempo=1.2, mirror=False),
    "E": dict(nod=2.0, beat=3.0, sway=3.5, antenna=12.0, tempo=0.8, mirror=True),
    "F": dict(nod=5.0, beat=8.0, sway=2.0, antenna=6.0, tempo=1.3, mirror=False),
    "G": dict(nod=5.0, beat=8.0, sway=2.0, antenna=6.0, tempo=1.3, mirror=False),
}
_LIMITS = np.array([20.0, 20.0, 35.0, 70.0, 70.0])      # |roll|, |pitch|, |yaw|, |antennas| in degrees


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampling (speech-quality, no extra dependency)."""
    if sr_in == sr_out or len(samples) == 0:
        return samples.astype(np.float32)
    n_out = int(round(len(samples) * sr_out / sr_in))
    x_out = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_out, np.arange(len(samples)), samples).astype(np.float32)


def loudness(samples: np.ndarray, sr: int, hop_s: float = 0.02) -> np.ndarray:
    """Speech loudness per hop, mapped to 0..1 (-45 dBFS -> 0, -12 dBFS -> 1)."""
    hop = max(1, int(sr * hop_s))
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(samples[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return np.clip((20 * np.log10(rms) + 45) / 33, 0, 1)


class Animator:
    """Continuous head and antenna motion for Reachy Mini, streamed with set_target at 50 Hz."""

    HZ = 50
    HOP_S = 0.02

    def __init__(self, mini, poses: dict, styles: Optional[dict] = None, seed: int = 7):
        self._mini, self._poses = mini, poses
        self._styles = {**DEFAULT_STYLES, **(styles or {})}
        self._target = self._vec("neutral")
        self._base = self._target.copy()
        self._tau = 0.3
        self._style = self._styles["neutral"]
        self._mode = "idle"
        self._levels: collections.deque = collections.deque()     # (monotonic time, loudness)
        self._level = self._env = self._env_slow = self._beat = self._talk = 0.0
        self._last_beat = -1.0
        self._rng = np.random.default_rng(seed)
        self._glance = np.zeros(3)
        self._glance_target = np.zeros(3)
        self._next_glance = 0.0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _vec(self, key: str) -> np.ndarray:
        (roll, pitch, yaw), (a0, a1) = self._poses[key]
        return np.array([roll, pitch, yaw, a0, a1], dtype=float)

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reachy-animator")
        self._thread.start()

    def close(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(1)

    def set_state(self, key: str, duration: float = 0.5, mode: Optional[str] = None) -> None:
        if key not in self._poses:
            return
        with self._lock:
            self._target = self._vec(key)
            self._tau = max(0.06, duration / 3)
            if mode:
                self._mode = mode

    def set_style(self, key: str) -> None:
        with self._lock:
            self._style = self._styles.get(key, self._styles["neutral"])
            self._mode = "speaking"

    def feed(self, samples: np.ndarray, sr: int, t_start: float) -> None:
        """Loudness of audio that starts playing at monotonic time t_start."""
        lv = loudness(samples, sr, self.HOP_S)
        with self._lock:
            self._levels.extend((t_start + i * self.HOP_S, float(x)) for i, x in enumerate(lv))

    def speech_end(self) -> None:
        with self._lock:
            self._levels.clear()
            self._level = 0.0

    def _loop(self) -> None:
        from reachy_mini.utils import create_head_pose
        last = time.monotonic()
        while self._running.is_set():
            now = time.monotonic()
            head, z, ant = self.tick(now, now - last)
            last = now
            try:
                self._mini.set_target(head=create_head_pose(z=z, roll=head[0], pitch=head[1], yaw=head[2],
                                                            mm=True, degrees=True),
                                      antennas=np.deg2rad(ant))
            except Exception as e:
                log.warning("Reachy Mini animation failed: %s", e)
                time.sleep(0.5)
            time.sleep(max(0.0, 1.0 / self.HZ - (time.monotonic() - now)))

    def tick(self, t: float, dt: float) -> tuple:
        """One animation step: (roll, pitch, yaw) degrees, z mm, (antenna0, antenna1) degrees."""
        with self._lock:
            self._base += (self._target - self._base) * (1 - math.exp(-dt / self._tau))
            while self._levels and self._levels[0][0] <= t:
                self._level = self._levels.popleft()[1]
            style, mode, base = self._style, self._mode, self._base.copy()
        x = self._level
        self._env += (x - self._env) * (0.5 if x > self._env else 0.15)
        self._env_slow += (self._env - self._env_slow) * 0.03
        self._talk += ((1.0 if x > 0.05 else 0.0) - self._talk) * (1 - math.exp(-dt / 0.4))
        if mode == "speaking" and self._env - self._env_slow > 0.22 and t - self._last_beat > 0.28:
            self._beat, self._last_beat = 1.0, t
        self._beat *= math.exp(-dt / 0.12)

        # glances while listening or idle: small, slow shifts of gaze every 3-6 s
        if mode in ("listening", "idle") and t >= self._next_glance:
            self._glance_target = self._rng.uniform([-4, -3, -8], [4, 3, 8])
            self._next_glance = t + self._rng.uniform(3, 6)
        elif mode not in ("listening", "idle"):
            self._glance_target = np.zeros(3)
        self._glance += (self._glance_target - self._glance) * (1 - math.exp(-dt / 0.5))

        ph = t * style["tempo"]
        breathe = math.sin(2 * math.pi * 0.22 * t)
        sway = style["sway"] * (0.35 + 0.65 * self._talk)
        roll = base[0] + sway * math.sin(2 * math.pi * 0.23 * ph) + self._glance[0]
        pitch = base[1] + style["nod"] * self._env + style["beat"] * self._beat + 0.8 * breathe + self._glance[1]
        yaw = base[2] + 1.3 * sway * math.sin(2 * math.pi * 0.17 * ph + 1.0) + self._glance[2]
        z = 2.0 * breathe + 2.5 * self._env
        flutter = style["antenna"] * self._env
        a0 = base[3] + flutter * math.sin(2 * math.pi * 1.6 * ph) + 4 * breathe
        a1 = base[4] + flutter * math.sin(2 * math.pi * 1.6 * ph + (math.pi if style["mirror"] else 0.6)) + 4 * breathe
        if mode == "thinking":
            a0 += 12 * math.sin(2 * math.pi * 0.5 * t)
            a1 -= 12 * math.sin(2 * math.pi * 0.5 * t)
        v = np.clip([roll, pitch, yaw, a0, a1], -_LIMITS, _LIMITS)
        return (v[0], v[1], v[2]), float(z), [v[3], v[4]]


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
                 poses: Optional[dict] = None, mini=None, tts=None, speech_log_dir: Optional[str] = None,
                 tts_engine: str = "system", tts_device: str = "auto", animate: bool = True,
                 styles: Optional[dict] = None):
        self._host, self._port, self._mode = host, port, connection_mode
        self._expressions = expressions
        self._speech_log_dir = speech_log_dir
        self._poses = {**DEFAULT_POSES, **(poses or {})}
        self._styles = styles
        self._mini = mini
        self._tts = tts
        self._tts_args = (tts_engine, tts_rate, tts_voice, tts_device)
        self._animate = expressions and animate
        self._anim: Optional[Animator] = None
        self._stop = threading.Event()
        self._playing = False
        speech = {"system": "computer TTS (SAPI / espeak-ng) on robot speaker",
                  "kokoro": "computer neural TTS (Kokoro-82M) on robot speaker"}.get(tts_engine, tts_engine)
        self.capabilities = Capabilities(
            robot="Reachy Mini", speech=speech, interrupt="verified",
            expressions=expressions, listening="robot microphones through the SDK, to local VAD + ASR",
            notes="speech streamed in 0.1 s chunks; stop takes effect within one chunk"
                  + ("; continuous speech-driven head and antenna motion" if self._animate else ""),
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
            engine, rate, voice, device = self._tts_args
            self._tts = create_tts(engine, rate, voice, device)
        if hasattr(self._tts, "warm_up"):
            self._tts.warm_up()
        self._goto("neutral", 0.5, force=True)
        if self._animate and self._anim is None:
            time.sleep(0.5)
            self._anim = Animator(self._mini, self._poses, self._styles)
            self._anim.start()

    def _goto(self, key: str, duration: float = 0.6, force: bool = False) -> None:
        from reachy_mini.utils import create_head_pose
        (roll, pitch, yaw), (a_left, a_right) = self._poses[key]
        try:
            self._mini.goto_target(head=create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True),
                                   antennas=np.deg2rad([a_left, a_right]), duration=duration)
        except Exception as e:
            log.warning("Reachy Mini pose %s failed: %s", key, e)

    def _pose(self, key: str, duration: float = 0.6, mode: Optional[str] = None) -> None:
        if not self._expressions or key not in self._poses:
            return
        if self._anim is not None:
            self._anim.set_state(key, duration, mode)
        else:
            self._goto(key, duration)

    def _stream(self, text: str):
        if hasattr(self._tts, "stream"):
            return self._tts.stream(text)
        return iter([self._tts.synthesize(text)])

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        media = self._mini.media
        sr_out, channels = media.get_output_audio_samplerate(), media.get_output_channels()
        key = expression_key(cue)
        self._pose(key, 0.4)
        if self._anim is not None:
            self._anim.set_style(key)

        parts: queue.Queue = queue.Queue()

        def produce():            # synthesize sentence by sentence while earlier ones play
            try:
                for samples, sr_in in self._stream(text):
                    if self._stop.is_set():
                        break
                    parts.put(resample(samples, sr_in, sr_out))
            except Exception as e:
                parts.put(e)
            finally:
                parts.put(None)

        threading.Thread(target=produce, daemon=True, name="reachy-tts-stream").start()
        media.start_playing()
        self._playing = True
        step = int(sr_out * CHUNK_S)
        played: list = []         # audio as heard, including silences while synthesis caught up
        n = 0                     # samples of playback time elapsed since the first push
        t0 = wall_start = None
        try:
            while True:
                try:
                    part = parts.get(timeout=0.05)
                except queue.Empty:
                    if self._stop.is_set():
                        return False
                    continue
                if part is None:
                    break
                if isinstance(part, Exception):
                    raise RuntimeError(f"Reachy Mini TTS failed: {part}")
                now = time.monotonic()
                if t0 is None:
                    t0, wall_start = now, time.time()
                elif now > t0 + n / sr_out:                     # synthesis lagged: the robot was silent
                    gap = int((now - t0) * sr_out) - n
                    played.append(np.zeros(gap, dtype=np.float32))
                    n += gap
                for start in range(0, len(part), step):
                    if self._stop.is_set():
                        return False
                    piece = part[start:start + step]
                    media.push_audio_sample(np.repeat(piece.reshape(-1, 1), channels, axis=1) if channels > 1
                                            else piece.reshape(-1, 1))
                    if self._anim is not None:
                        self._anim.feed(piece, sr_out, t0 + n / sr_out)
                    played.append(piece)
                    n += len(piece)
                    # push_audio_sample is non-blocking: pace pushes in real time
                    delay = t0 + n / sr_out - CHUNK_S * 0.5 - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
            remaining = t0 + n / sr_out - time.monotonic() if t0 is not None else 0
            if remaining > 0:
                self._stop.wait(remaining)
            return not self._stop.is_set()
        finally:
            media.stop_playing()
            self._playing = False
            if self._anim is not None:
                self._anim.speech_end()
            self._pose("neutral", 0.4)
            if self._speech_log_dir and played:
                self._log_speech(text, np.concatenate(played), sr_out, wall_start, self._stop.is_set())

    def _log_speech(self, text: str, samples: np.ndarray, sr: int, wall_start: float, interrupted: bool) -> None:
        """Archive the robot's utterance exactly as played (WAV + one JSON line per utterance)."""
        import json
        from pathlib import Path
        d = Path(self._speech_log_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"robot_{wall_start:.3f}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        with open(d / "speech_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall_start": wall_start, "file": path.name, "seconds": round(len(samples) / sr, 3),
                                "interrupted": interrupted, "text": text}) + "\n")

    def stop(self) -> bool:
        self._stop.set()
        return True

    def on_listening(self) -> None:
        self._pose("listening", 0.5, mode="listening")

    def on_thinking(self) -> None:
        self._pose("thinking", 0.5, mode="thinking")

    def on_idle(self) -> None:
        self._pose("neutral", 0.5, mode="idle")

    def close(self) -> None:
        if self._anim is not None:
            self._anim.close()
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                pass
