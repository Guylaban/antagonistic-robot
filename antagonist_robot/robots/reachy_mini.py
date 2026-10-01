"""Reachy Mini backend (Pollen Robotics / Hugging Face, pip package reachy-mini).

Reachy Mini has no built-in text-to-speech, so replies are synthesized
offline on the computer (tts_engine: "system" = SAPI / espeak-ng, or
"kokoro" = neural TTS, see robots/tts.py) and streamed to the robot's
speaker through the SDK in 0.1 s chunks, sentence by sentence, which
makes stop() take effect within one chunk. Listening uses the robot's
microphones through the SDK (RAWR runs VAD and ASR locally).

Non-verbal cues (robot.expressions, on by default) use the head and
antennas, lean, and body turn, and follow the reply being spoken: the pose,
motion style, and gestures come from the category the judge finds in it
(else the requested one), scaled by its antagonism level, so a mild reply
moves gently and a strong one sharply and often. With animate: true (the
default) the robot moves continuously: it eases into a pose per state and
reply, breathes and glances while listening, moves with the loudness of its
own speech, gestures on stressed syllables (e.g. a jab toward the listener
when confrontational, a head tilt when sarcastic, a nod when supportive),
and marks sentence endings (a head shake on a confrontational question, an
eye-roll after a sarcastic remark, a look away when dismissive). With
animate: false it only switches between fixed poses.

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
import re
import threading
import time
import wave
from typing import Optional

import numpy as np

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key, expression_strength
from antagonist_robot.robots.tts import _SENTENCE, SystemTTS, create_tts

log = logging.getLogger(__name__)

CHUNK_S = 0.1
_TTSWorker = SystemTTS          # earlier name, kept for scripts that import it

# Pose per state and condition, used only when expressions are on. Adjust for your study.
#   head (roll, pitch, yaw) degrees, positive pitch looks down; antennas (a0, a1) degrees;
#   body (lean x mm toward the listener, height z mm, body yaw degrees).
DEFAULT_POSES = {
    "neutral": ((0, 0, 0), (0, 0), (0, 0, 0)),
    "support": ((0, -6, 0), (35, 35), (6, 4, 0)),        # head up, antennas up, leaning in
    "B": ((0, -3, 22), (-25, -25), (-6, 0, 25)),         # looks and turns away, leans back
    "C": ((14, -4, 6), (35, -20), (0, 0, 0)),            # tilted head, lopsided antennas
    "D": ((0, -6, 0), (-15, -15), (6, 0, 0)),            # chin up, leaning in
    "E": ((10, 2, -6), (15, -12), (0, 0, 0)),            # slight tilt, mixed antennas
    "F": ((0, 6, 0), (-35, -35), (8, -3, 0)),          # head down, glaring, antennas back, leaning in
    "G": ((0, 7, 0), (-45, -45), (9, -4, 0)),
    "listening": ((0, 0, 0), (10, 10), (0, 0, 0)),
    "thinking": ((6, -4, 0), (0, 15), (0, 0, 0)),
}

# Motion while speaking, per condition (animate: true). Degrees. nod: head dips with loudness;
# beat: dip on stressed syllables; sway: slow roll/yaw drift; antenna: flutter with speech;
# tempo: speed of sway and flutter; mirror: antennas move in opposition (asymmetric, "sarcastic");
# gesture: what a stressed syllable triggers (see GESTURES); end: gesture per sentence ending.
DEFAULT_STYLES = {
    "neutral": dict(nod=3.0, beat=4.0, sway=2.5, antenna=10.0, tempo=1.0, mirror=False,
                    gesture="nod", end={}),
    "support": dict(nod=3.0, beat=3.0, sway=3.5, antenna=16.0, tempo=0.9, mirror=False,
                    gesture="nod", end={"!": "wiggle", "?": "tilt", ".": "nod"}),
    "B": dict(nod=1.5, beat=2.0, sway=1.5, antenna=5.0, tempo=0.7, mirror=False,
              gesture=None, end={".": "look_away", "?": "look_away", "!": "look_away"}),
    "C": dict(nod=2.5, beat=4.0, sway=4.5, antenna=18.0, tempo=0.8, mirror=True,
              gesture="tilt", end={".": "eye_roll", "?": "eye_roll", "!": "tilt"}),
    "D": dict(nod=4.5, beat=7.0, sway=2.0, antenna=7.0, tempo=1.2, mirror=False,
              gesture="jab", end={"?": "shake", "!": "big_jab"}),
    "E": dict(nod=2.0, beat=3.0, sway=3.5, antenna=12.0, tempo=0.8, mirror=True,
              gesture="small_tilt", end={".": "eye_roll", "?": "tilt"}),
    "F": dict(nod=5.0, beat=8.0, sway=2.0, antenna=6.0, tempo=1.3, mirror=False,
              gesture="jab", end={"?": "shake", "!": "big_jab", ".": "big_jab"}),
    "G": dict(nod=5.0, beat=8.0, sway=2.0, antenna=6.0, tempo=1.3, mirror=False,
              gesture="jab", end={"?": "shake", "!": "big_jab", ".": "big_jab"}),
}


def _bump(u: float) -> float:
    """0 -> 1 -> 0 over u in [0, 1], with a quick attack."""
    return math.sin(math.pi * min(1.0, max(0.0, u)) ** 0.7)


# Short gestures layered on the pose: name -> (duration s, delta(u) for u in [0, 1]) returning
# [roll, pitch, yaw, a0, a1, x mm, z mm, body yaw] in degrees/mm, before scaling by the strength.
GESTURES = {
    "nod": (0.4, lambda u: [0, 5 * _bump(u), 0, 0, 0, 0, 0, 0]),
    "jab": (0.3, lambda u: [0, 7 * _bump(u), 0, -10 * _bump(u), -10 * _bump(u), 5 * _bump(u), 0, 0]),
    "big_jab": (0.45, lambda u: [0, 11 * _bump(u), 0, -20 * _bump(u), -20 * _bump(u), 7 * _bump(u),
                                 -4 * _bump(u), 0]),
    "shake": (0.7, lambda u: [0, 0, 13 * math.sin(4 * math.pi * u) * (1 - u), 0, 0, 0, 0, 0]),
    "tilt": (0.8, lambda u: [11 * _bump(u), 0, 4 * _bump(u), 0, 0, 0, 0, 0]),
    "small_tilt": (0.8, lambda u: [6 * _bump(u), 0, 0, 0, 0, 0, 0, 0]),
    "eye_roll": (1.0, lambda u: [8 * math.sin(math.pi * u), -12 * math.sin(math.pi * u),
                                 10 * math.sin(2 * math.pi * u), 10 * math.sin(math.pi * u), 0, 0, 0, 0]),
    "look_away": (1.2, lambda u: [0, -3 * _bump(u), 16 * _bump(u), 0, 0, -4 * _bump(u), -3 * _bump(u),
                                  10 * _bump(u)]),
    "wiggle": (0.8, lambda u: [0, 0, 0, 25 * math.sin(6 * math.pi * u) * (1 - u),
                               -25 * math.sin(6 * math.pi * u) * (1 - u), 0, 0, 0]),
}
# |roll|, |pitch|, |yaw|, |antennas| degrees; |x|, |z| mm; |body yaw| degrees
_LIMITS = np.array([25.0, 25.0, 40.0, 80.0, 80.0, 18.0, 12.0, 40.0])


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


def expression_gain(cue: Optional[SpeechCue]) -> float:
    """Scale for poses, motion, and gestures: 1 for neutral; 0.25 + 1.5 x strength otherwise
    (0.75 for a mild reply, 1.25 at medium, 1.75 for the strongest)."""
    key = expression_key(cue)
    return 1.0 if key == "neutral" else round(0.25 + 1.5 * expression_strength(cue), 3)


def sentence_endings(text: str) -> list:
    """Final punctuation of each sentence, in the order the TTS speaks them ('.' if none)."""
    parts = [p for p in re.split(_SENTENCE, text.strip()) if p.strip()]
    return [(p.rstrip()[-1] if p.rstrip()[-1] in ".!?" else ".") for p in parts]


def _pose_vec(pose) -> np.ndarray:
    head, ant = pose[0], pose[1]
    body = pose[2] if len(pose) > 2 else (0, 0, 0)
    return np.array([*head, *ant, *body], dtype=float)


class Animator:
    """Continuous head, antenna, and body motion for Reachy Mini, streamed with set_target at 50 Hz.

    The pose and style come from the reply's expression and are scaled by its strength (gain):
    a stronger reply holds a more marked pose, moves more, and gestures more often and more
    sharply. Gestures fire on stressed syllables of the robot's own speech and at sentence
    endings ('!' and '?' have their own gestures per condition).
    """

    HZ = 50
    HOP_S = 0.02

    def __init__(self, mini, poses: dict, styles: Optional[dict] = None, seed: int = 7):
        self._mini, self._poses = mini, poses
        self._styles = {**DEFAULT_STYLES, **(styles or {})}
        self._target = self._vec("neutral")
        self._base = self._target.copy()
        self._tau = 0.3
        self._style = self._styles["neutral"]
        self._gain = 1.0
        self._mode = "idle"
        self._levels: collections.deque = collections.deque()     # (monotonic time, loudness)
        self._marks: collections.deque = collections.deque()      # (monotonic time, gesture name)
        self._active: list = []                                   # (start time, gesture name, scale)
        self._level = self._env = self._env_slow = self._beat = self._talk = 0.0
        self._last_beat = self._last_gesture = -1.0
        self._rng = np.random.default_rng(seed)
        self._glance = np.zeros(3)
        self._glance_target = np.zeros(3)
        self._next_glance = 0.0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _vec(self, key: str) -> np.ndarray:
        return _pose_vec(self._poses[key])

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reachy-animator")
        self._thread.start()

    def close(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(1)

    def set_state(self, key: str, duration: float = 0.5, mode: Optional[str] = None, gain: float = 1.0) -> None:
        if key not in self._poses:
            return
        with self._lock:
            self._target = self._vec(key) * gain
            self._tau = max(0.06, duration / 3)
            if mode:
                self._mode = mode
            if mode in ("listening", "idle", "thinking"):
                self._marks.clear()

    def set_style(self, key: str, gain: float = 1.0) -> None:
        """Motion style for the reply about to be spoken; gain scales its amplitudes (antagonism level)."""
        style = dict(self._styles.get(key, self._styles["neutral"]))
        for k in ("nod", "beat", "sway", "antenna"):
            style[k] *= gain
        with self._lock:
            self._style, self._gain = style, gain
            self._mode = "speaking"

    def feed(self, samples: np.ndarray, sr: int, t_start: float) -> None:
        """Loudness of audio that starts playing at monotonic time t_start."""
        lv = loudness(samples, sr, self.HOP_S)
        with self._lock:
            self._levels.extend((t_start + i * self.HOP_S, float(x)) for i, x in enumerate(lv))

    def sentence_end(self, t_end: float, punctuation: str) -> None:
        """A sentence ending with this punctuation finishes playing at monotonic time t_end."""
        with self._lock:
            name = self._style.get("end", {}).get(punctuation)
            if name:
                self._marks.append((t_end - 0.5 * GESTURES[name][0], name))

    def speech_end(self) -> None:
        with self._lock:
            self._levels.clear()
            self._marks.clear()
            self._level = 0.0

    def _loop(self) -> None:
        from reachy_mini.utils import create_head_pose
        last = time.monotonic()
        while self._running.is_set():
            now = time.monotonic()
            head, z, ant, x, body = self.tick(now, now - last)
            last = now
            try:
                self._mini.set_target(head=create_head_pose(x=x, z=z, roll=head[0], pitch=head[1], yaw=head[2],
                                                            mm=True, degrees=True),
                                      antennas=np.deg2rad(ant), body_yaw=math.radians(body))
            except Exception as e:
                log.warning("Reachy Mini animation failed: %s", e)
                time.sleep(0.5)
            time.sleep(max(0.0, 1.0 / self.HZ - (time.monotonic() - now)))

    def _start_gesture(self, t: float, name: str, scale: float) -> None:
        self._active.append((t, name, scale))
        self._last_gesture = t

    def tick(self, t: float, dt: float) -> tuple:
        """One animation step: (roll, pitch, yaw) degrees, z mm, (a0, a1) degrees, lean x mm, body yaw degrees."""
        with self._lock:
            self._base += (self._target - self._base) * (1 - math.exp(-dt / self._tau))
            while self._levels and self._levels[0][0] <= t:
                self._level = self._levels.popleft()[1]
            due = []
            while self._marks and self._marks[0][0] <= t:
                due.append(self._marks.popleft()[1])
            style, mode, base, gain = self._style, self._mode, self._base.copy(), self._gain
        strength = min(1.0, max(0.0, (gain - 0.25) / 1.5))
        x_lv = self._level
        self._env += (x_lv - self._env) * (0.5 if x_lv > self._env else 0.15)
        self._env_slow += (self._env - self._env_slow) * 0.03
        self._talk += ((1.0 if x_lv > 0.05 else 0.0) - self._talk) * (1 - math.exp(-dt / 0.4))

        # stressed syllable: a beat, and (more often and closer together when stronger) a gesture
        if mode == "speaking" and self._env - self._env_slow > 0.22 - 0.08 * strength \
                and t - self._last_beat > 0.28:
            self._beat, self._last_beat = 1.0, t
            name = style.get("gesture")
            if name and t - self._last_gesture > 0.9 - 0.5 * strength \
                    and self._rng.random() < 0.3 + 0.6 * strength:
                self._start_gesture(t, name, gain)
        for name in due:                                          # sentence endings ('!', '?', '.')
            self._start_gesture(t, name, gain)
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
        v = base.copy()
        v[0] += sway * math.sin(2 * math.pi * 0.23 * ph) + self._glance[0]
        v[1] += style["nod"] * self._env + style["beat"] * self._beat + 0.8 * breathe + self._glance[1]
        v[2] += 1.3 * sway * math.sin(2 * math.pi * 0.17 * ph + 1.0) + self._glance[2]
        v[6] += 2.0 * breathe + 2.5 * self._env
        flutter = style["antenna"] * self._env
        v[3] += flutter * math.sin(2 * math.pi * 1.6 * ph) + 4 * breathe
        v[4] += flutter * math.sin(2 * math.pi * 1.6 * ph + (math.pi if style["mirror"] else 0.6)) + 4 * breathe
        if mode == "thinking":
            v[3] += 12 * math.sin(2 * math.pi * 0.5 * t)
            v[4] -= 12 * math.sin(2 * math.pi * 0.5 * t)
        still = []
        for start, name, scale in self._active:
            duration, delta = GESTURES[name]
            u = (t - start) / duration
            if u < 1:
                v += scale * np.asarray(delta(u), dtype=float)
                still.append((start, name, scale))
        self._active = still
        v = np.clip(v, -_LIMITS, _LIMITS)
        return (v[0], v[1], v[2]), float(v[6]), [v[3], v[4]], float(v[5]), float(v[7])

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

    def _goto(self, key: str, duration: float = 0.6, force: bool = False, gain: float = 1.0) -> None:
        from reachy_mini.utils import create_head_pose
        roll, pitch, yaw, a_left, a_right, x, z, body = np.clip(_pose_vec(self._poses[key]) * gain, -_LIMITS, _LIMITS)
        try:
            self._mini.goto_target(head=create_head_pose(x=x, z=z, roll=roll, pitch=pitch, yaw=yaw, mm=True,
                                                         degrees=True),
                                   antennas=np.deg2rad([a_left, a_right]), duration=duration,
                                   body_yaw=math.radians(body))
        except Exception as e:
            log.warning("Reachy Mini pose %s failed: %s", key, e)

    def _pose(self, key: str, duration: float = 0.6, mode: Optional[str] = None, gain: float = 1.0) -> None:
        if not self._expressions or key not in self._poses:
            return
        if self._anim is not None:
            self._anim.set_state(key, duration, mode, gain)
        else:
            self._goto(key, duration, gain=gain)

    def _stream(self, text: str):
        if hasattr(self._tts, "stream"):
            return self._tts.stream(text)
        return iter([self._tts.synthesize(text)])

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        media = self._mini.media
        sr_out, channels = media.get_output_audio_samplerate(), media.get_output_channels()
        key = expression_key(cue)
        gain = expression_gain(cue)
        self._pose(key, 0.4, gain=gain)
        if self._anim is not None:
            self._anim.set_style(key, gain)

        parts: queue.Queue = queue.Queue()
        endings = sentence_endings(text)

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
        n_parts = 0               # sentences received from the TTS so far
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
                index = n_parts
                n_parts += 1
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
                    if start == 0 and self._anim is not None:     # schedule this sentence's ending gesture
                        if not endings:
                            ending = "."
                        elif getattr(self._tts, "streams_sentences", False):   # one part per sentence
                            ending = endings[min(index, len(endings) - 1)]
                        else:                                                  # the whole reply in one part
                            ending = endings[-1]
                        self._anim.sentence_end(t0 + (n - len(piece) + len(part)) / sr_out, ending)
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
