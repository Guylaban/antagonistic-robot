"""Furhat backend via the Furhat Remote API (pip package furhat-remote-api).

Requires the Remote API skill running on the robot or on the virtual
Furhat of the Furhat SDK (it listens on port 54321). Speech uses the
robot's own voice (`say(text)`), or, with tts_engine "kokoro" or
"system", audio synthesized by RAWR sentence by sentence and played by
the robot with lip sync requested (`say(url, lipsync)`; RAWR serves
the WAV files over HTTP on audio_port, so the robot must be able to
reach this computer). On the virtual Furhat of SDK 2.9.2 the audio
played but the lips did not move, so keep the default ("furhat", the
robot's own voices, which include neural voices and do lip-sync) unless
your robot generates lip sync for audio. Listening uses the robot's
microphones and its own speech recognition (`listen`), which returns
text only. Optional non-verbal cues (expressions: true): a sequence of
Furhat expressions per condition: a facial expression held through the reply and scaled by its
strength, gestures at sentence starts and endings, a thinking
expression while a reply is prepared, and the LED ring.

Interruption: stop() sends `say_stop` and returns control to RAWR at once,
so the session never waits on the robot. In our tests with the virtual
Furhat of SDK 2.9.2, `say_stop` was acknowledged but did not cut the
audio short (the next utterance started only after the previous one
ended), so interruption is reported as "unverified" for this backend.
"""

import logging
import queue
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from antagonist_robot.pipeline.types import ASRResult, AudioData
from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue, expression_key, expression_strength

log = logging.getLogger(__name__)

# listen() returns these instead of user speech
_NO_SPEECH = {"SILENCE", "INTERRUPTED", "FAILED", ""}


class FurhatRecognizer:
    """Listening through Furhat's own microphones and speech recognition (Remote API listen()).

    Furhat's recognizer runs on the robot's side (a cloud ASR service), so no
    raw audio reaches RAWR and none is archived; only the transcript is logged.
    """

    def __init__(self, api: Callable, language: str = "en-US"):
        self._api = api
        self._language = language
        self._text: Optional[str] = None

    def record_utterance(self, is_active: Optional[Callable[[], bool]] = None) -> Optional[AudioData]:
        is_active = is_active or (lambda: True)
        while is_active():
            started = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            try:
                status = self._api().listen(language=self._language)
                text = (getattr(status, "message", "") or "").strip()
            except Exception as e:
                log.warning("Furhat listen failed: %s", e)
                time.sleep(0.5)
                continue
            if not is_active():
                return None
            if text.upper() in _NO_SPEECH:
                continue
            self._text = text
            self._elapsed = time.monotonic() - t0
            return AudioData(samples=np.zeros(0, dtype=np.float32), sample_rate=16000, duration_seconds=0.0,
                             recording_started=started, recording_ended=datetime.now(timezone.utc).isoformat())
        return None

    def transcribe(self, audio: AudioData) -> ASRResult:
        text, self._text = self._text or "", None
        return ASRResult(text=text, language=self._language, confidence=0.0, transcription_time_seconds=0.0)

# Gestures per expression (Furhat built-in gesture names), ordered from mild to strong. For each reply
# the cue strength (0-1, the antagonism or support level of the reply) sets how far into the sequence the
# robot goes and how often it gestures: the first gesture as speech starts, then one every
# gesture_interval(strength) seconds. Used only when expressions are on.
DEFAULT_GESTURES = {
    "support": ["Smile", "Nod", "BigSmile"], "neutral": [],
    "B": ["GazeAway", "Roll"], "C": ["BrowRaise", "Smile", "Roll"], "D": ["BrowFrown", "Shake", "BrowFrown"],
    "E": ["Smile", "BrowRaise"], "F": ["BrowFrown", "Shake", "ExpressAnger"],
    "G": ["BrowFrown", "ExpressDisgust", "ExpressAnger"],
}


def gestures_for(sequence: list, strength: float) -> list:
    """The part of a mild-to-strong gesture sequence a reply of this strength uses (at least one)."""
    if not sequence:
        return []
    return list(sequence[:max(1, round(len(sequence) * strength + 0.25))])


def gesture_interval(strength: float) -> float:
    """Seconds between gestures: about 3.4 s for a mild reply, 1.6 s for the strongest."""
    return 4.0 - 2.4 * strength


# Facial expression held while a reply is spoken (Furhat face and neck parameters; expression
# parameters 0-1, NECK_* in degrees), scaled by the reply's strength, so a mild reply shows a hint
# and a strong one the full expression. Used only when expressions are on.
DEFAULT_FACES = {
    "support": {"SMILE_OPEN": 0.6, "BROW_UP_LEFT": 0.4, "BROW_UP_RIGHT": 0.4},
    "B": {"BROW_UP_LEFT": 0.3, "BROW_UP_RIGHT": 0.3, "EYE_SQUINT_LEFT": 0.3, "EYE_SQUINT_RIGHT": 0.3,
          "NECK_PAN": 12.0},
    "C": {"SMILE_CLOSED": 0.5, "BROW_UP_LEFT": 0.8, "NECK_ROLL": 8.0},
    "D": {"BROW_DOWN_LEFT": 0.6, "BROW_DOWN_RIGHT": 0.6, "BROW_IN_LEFT": 0.4, "BROW_IN_RIGHT": 0.4},
    "E": {"SMILE_CLOSED": 0.4, "BROW_UP_RIGHT": 0.5, "NECK_ROLL": -6.0},
    "F": {"EXPR_ANGER": 0.6, "BROW_DOWN_LEFT": 0.8, "BROW_DOWN_RIGHT": 0.8, "EYE_SQUINT_LEFT": 0.4,
          "EYE_SQUINT_RIGHT": 0.4},
    "G": {"EXPR_ANGER": 0.8, "EXPR_DISGUST": 0.4, "BROW_DOWN_LEFT": 1.0, "BROW_DOWN_RIGHT": 1.0,
          "EYE_SQUINT_LEFT": 0.5, "EYE_SQUINT_RIGHT": 0.5},
}
# Gesture at a sentence ending, by its punctuation (built-in names): a head shake on a confrontational
# question, a raised brow on a sarcastic one, an angry flash on an aggressive exclamation.
DEFAULT_ENDINGS = {
    "support": {"!": "BigSmile", ".": "Nod", "?": "BrowRaise"},
    "B": {".": "GazeAway", "?": "Roll"},
    "C": {"?": "BrowRaise", ".": "Roll", "!": "Smile"},
    "D": {"?": "Shake", "!": "BrowFrown"},
    "E": {"?": "BrowRaise", ".": "Smile"},
    "F": {"?": "Shake", "!": "ExpressAnger", ".": "BrowFrown"},
    "G": {"?": "Shake", "!": "ExpressAnger", ".": "ExpressDisgust"},
}


def face_gain(strength: float) -> float:
    """Scale of the held expression: 0.6 for a mild reply, 1.2 for the strongest (parameters capped at 1)."""
    return 0.3 + 0.9 * strength


def face_gesture(params: dict, gain: float, name: str = "RAWRFace") -> dict:
    """A custom Furhat gesture that eases into the expression and holds it until reset."""
    scaled = {k: (round(v * gain, 2) if k.startswith("NECK") else round(min(1.0, v * gain), 2))
              for k, v in params.items()}
    return {"name": name, "class": "furhatos.gestures.Gesture",
            "frames": [{"time": [0.35], "persist": True, "params": scaled}]}


def sentences(text: str) -> list:
    from antagonist_robot.robots.tts import _SENTENCE
    import re
    return [p for p in re.split(_SENTENCE, text.strip()) if p.strip()] or [text]
# LED colors (r, g, b) for state cues; used only when expressions are on.
LED = {"listening": (0, 60, 120), "thinking": (120, 90, 0), "idle": (0, 0, 0)}


class _AudioServer:
    """Serves synthesized WAV files to the robot (Furhat fetches audio by URL)."""

    def __init__(self, port: int, host: Optional[str] = None, robot_host: str = "localhost"):
        import http.server
        files = self._files = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                data = files.get(self.path.split("?")[0].lstrip("/"))
                if data is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="furhat-audio").start()
        self.base = f"http://{host or self._own_address(robot_host)}:{port}"
        self._n = 0

    @staticmethod
    def _own_address(robot_host: str) -> str:
        if robot_host in ("localhost", "127.0.0.1"):
            return "127.0.0.1"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((robot_host, 54321))      # no packet is sent; picks the interface that reaches the robot
            return s.getsockname()[0]
        finally:
            s.close()

    def add(self, wav: bytes) -> str:
        self._n += 1
        name = f"rawr_{self._n}.wav"
        self._files[name] = wav
        self._files.pop(f"rawr_{self._n - 20}.wav", None)
        return f"{self.base}/{name}"

    def close(self) -> None:
        self._httpd.shutdown()


class FurhatBackend(RobotBackend):
    """Furhat robot (physical or virtual) through the Remote API."""

    def __init__(self, host: str = "localhost", voice: Optional[str] = None, expressions: bool = False,
                 gestures: Optional[dict] = None, client=None, language: str = "en-US",
                 tts_engine: str = "furhat", tts_voice: Optional[str] = None, tts_rate: Optional[int] = None,
                 audio_host: Optional[str] = None, audio_port: int = 8095, speech_log_dir: Optional[str] = None,
                 tts=None):
        self._host = host
        self._voice = voice
        self._tts_engine, self._tts = tts_engine, tts
        self._tts_args = (tts_voice, tts_rate)
        self._audio_host, self._audio_port = audio_host, audio_port
        self._audio: Optional[_AudioServer] = None
        self._speech_log_dir = speech_log_dir
        self._language = language
        self._expressions = expressions
        self._gestures = {**DEFAULT_GESTURES, **(gestures or {})}
        self._faces, self._endings = dict(DEFAULT_FACES), dict(DEFAULT_ENDINGS)
        self._client = client
        self._stop = threading.Event()
        self._done = threading.Event()
        speech = "robot TTS (Furhat voice)" if tts_engine == "furhat" else \
            f"computer TTS ({tts_engine}) played by the robot with lip sync"
        self.capabilities = Capabilities(
            robot="Furhat", speech=speech, interrupt="unverified", expressions=expressions,
            listening="robot microphones and Furhat's own (cloud) speech recognition; no audio archived",
            notes="say_stop did not interrupt audio on the virtual Furhat (SDK 2.9.2); verify on your robot",
        )

    def recognizer(self):
        return FurhatRecognizer(self._api, self._language)

    def _api(self):
        if self._client is None:
            from furhat_remote_api import FurhatRemoteAPI  # optional dependency
            self._client = FurhatRemoteAPI(self._host)
        return self._client

    def connect(self) -> None:
        try:
            voices = [v.name for v in self._api().get_voices()]
        except Exception as e:
            raise RuntimeError(
                f"Furhat Remote API not reachable at {self._host}:54321 ({e}). Start the robot (or the virtual "
                f"Furhat in the Furhat SDK) and launch the Remote API skill."
            ) from e
        if self._voice:
            if self._voice not in voices:
                raise RuntimeError(f"Furhat voice {self._voice!r} not available; choose one of {voices[:10]}...")
            self._api().set_voice(name=self._voice)
        if self._tts_engine != "furhat":
            from antagonist_robot.robots.tts import create_tts
            if self._tts is None:
                self._tts = create_tts(self._tts_engine, self._tts_args[1], self._tts_args[0])
            if hasattr(self._tts, "warm_up"):
                self._tts.warm_up()
            if self._audio is None:
                self._audio = _AudioServer(self._audio_port, self._audio_host, self._host)

    def _led(self, state: str) -> None:
        if self._expressions:
            r, g, b = LED[state]
            try:
                self._api().set_led(red=r, green=g, blue=b)
            except Exception as e:
                log.warning("Furhat LED failed: %s", e)

    def _gesture(self, name: Optional[str] = None, body: Optional[dict] = None) -> None:
        try:
            if body is not None:
                self._api().gesture(body=body, blocking=False)
            elif name:
                self._api().gesture(name=name, blocking=False)
        except Exception as e:
            log.warning("Furhat gesture failed: %s", e)

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        """Speak sentence by sentence; with expressions on, hold a face scaled by the reply's strength,
        gesture at each sentence start (more and stronger gestures for a stronger reply), and mark each
        sentence ending by its punctuation."""
        self._stop.clear()
        self._done.clear()
        key = expression_key(cue)
        strength = expression_strength(cue)
        seq = self._gestures.get(key) if self._expressions else None
        seq = gestures_for([seq] if isinstance(seq, str) else list(seq or []), strength)
        face = self._faces.get(key) if self._expressions else None
        endings = self._endings.get(key, {}) if self._expressions else {}
        parts = sentences(text)
        every = gesture_interval(strength)
        error = {}
        clock = {"next": time.monotonic() + every}

        def worker():
            try:
                if face:
                    self._gesture(body=face_gesture(face, face_gain(strength)))
                for i, sentence in enumerate(parts):
                    if self._stop.is_set():
                        break
                    if seq and (i == 0 or strength >= 0.5 or i % 2 == 0):
                        self._gesture(seq[i % len(seq)])
                    clock["next"] = time.monotonic() + every
                    if self._tts_engine == "furhat":
                        self._api().say(text=sentence, blocking=True)
                    else:
                        self._say_audio(sentence)
                    mark = sentence.rstrip()[-1:] if sentence.rstrip() else ""
                    if not self._stop.is_set() and endings.get(mark) and (mark in "?!" or strength >= 0.4):
                        self._gesture(endings[mark])
            except Exception as e:
                error["e"] = e
            finally:
                if face:
                    self._gesture(body=face_gesture({k: 0.0 for k in face}, 1.0, "RAWRFaceReset"))
                self._done.set()

        threading.Thread(target=worker, daemon=True, name="furhat-say").start()
        while not self._done.wait(0.05):
            if self._stop.is_set():
                return False          # control returns to RAWR immediately
            if seq and time.monotonic() >= clock["next"]:       # long sentence: keep gesturing
                clock["next"] += every
                self._gesture(seq[-1])
        if "e" in error:
            raise RuntimeError(f"Furhat say failed: {error['e']}")
        return not self._stop.is_set()

    def _say_audio(self, text: str) -> None:
        """Synthesize sentence by sentence (the next one while the robot plays the current one)."""
        from antagonist_robot.robots.tts import to_wav_bytes
        parts: queue.Queue = queue.Queue(maxsize=2)

        def produce():
            try:
                for samples, sr in self._tts.stream(text):
                    if self._stop.is_set():
                        break
                    parts.put((samples, sr))
            except Exception as e:
                parts.put(e)
            finally:
                parts.put(None)

        threading.Thread(target=produce, daemon=True, name="furhat-tts").start()
        while True:
            part = parts.get()
            if part is None or self._stop.is_set():
                return
            if isinstance(part, Exception):
                raise part
            samples, sr = part
            wav = to_wav_bytes(samples, sr)
            wall_start = time.time()
            self._api().say(url=self._audio.add(wav), lipsync=True, blocking=True)
            if self._speech_log_dir:
                self._log_speech(text, wav, len(samples) / sr, wall_start)

    def _log_speech(self, text: str, wav: bytes, seconds: float, wall_start: float) -> None:
        """Archive each sentence as sent to the robot (WAV + one JSON line; start = when say() was sent)."""
        import json
        from pathlib import Path
        d = Path(self._speech_log_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"furhat_{wall_start:.3f}.wav"
        path.write_bytes(wav)
        with open(d / "speech_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall_start": wall_start, "file": path.name, "seconds": round(seconds, 3),
                                "interrupted": self._stop.is_set(), "text": text}) + "\n")

    def stop(self) -> bool:
        self._stop.set()
        try:
            self._api().say_stop()
            return True
        except Exception as e:
            log.warning("Furhat say_stop failed: %s", e)
            return False

    def on_listening(self) -> None:
        self._led("listening")
        if self._expressions:
            try:
                self._api().attend(user="CLOSEST")
            except Exception:
                pass

    def on_thinking(self) -> None:
        self._led("thinking")
        if self._expressions:
            try:
                self._api().gesture(name="Thoughtful", blocking=False)
            except Exception:
                pass

    def on_idle(self) -> None:
        self._led("idle")
        try:
            self._api().listen_stop()   # release a pending listen() when the session ends
        except Exception:
            pass
