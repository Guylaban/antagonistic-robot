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
Furhat gestures per condition, played through each reply, a thinking
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

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        self._stop.clear()
        self._done.clear()
        gestures = self._gestures.get(expression_key(cue)) if self._expressions else None
        gestures = [gestures] if isinstance(gestures, str) else list(gestures or [])
        strength = expression_strength(cue)
        gestures = gestures_for(gestures, strength)
        every = gesture_interval(strength)
        error = {}

        def worker():
            try:
                if gestures:
                    self._api().gesture(name=gestures.pop(0), blocking=False)
                if self._tts_engine == "furhat":
                    self._api().say(text=text, blocking=True)
                else:
                    self._say_audio(text)
            except Exception as e:
                error["e"] = e
            finally:
                self._done.set()

        threading.Thread(target=worker, daemon=True, name="furhat-say").start()
        next_gesture = time.monotonic() + every
        while not self._done.wait(0.05):
            if self._stop.is_set():
                return False          # control returns to RAWR immediately
            if gestures and time.monotonic() >= next_gesture:
                next_gesture += every
                try:
                    self._api().gesture(name=gestures.pop(0), blocking=False)
                except Exception as e:
                    log.warning("Furhat gesture failed: %s", e)
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
