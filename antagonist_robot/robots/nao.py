"""NAO / Pepper backend: speech and microphone through nao_speaker_server.py on the robot.

Speech: text to the speaker server (port nao.port), spoken by NAOqi ALTextToSpeech.
Listening: the speaker server streams the robot's front microphone (port
nao.port + 1) while RAWR is listening; RAWR runs VAD and ASR locally.
"""

import socket
from typing import Optional

import numpy as np

from antagonist_robot.nao.host import resolve_ipv4
from antagonist_robot.nao.real import RealNAO
from antagonist_robot.pipeline.audio_output import NAOAudioOutput
from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue

MIC_HEADER = b"RAWRMIC 16000 1 s16le"


class NaoMicSource:
    """The robot's front microphone, streamed by nao_speaker_server.py (16 kHz mono s16le)."""

    def __init__(self, ip: str, port: int, read_timeout_s: float = 3.0):
        self._ip, self._port, self._timeout = ip, port, read_timeout_s
        self._sock = None
        self._buf = b""

    def __enter__(self):
        ip = resolve_ipv4(self._ip, self._port)
        self._sock = socket.create_connection((ip, self._port), timeout=5)
        header = b""
        while not header.endswith(b"\n"):
            chunk = self._sock.recv(1)
            if not chunk:
                break
            header += chunk
        if not header.startswith(MIC_HEADER):
            self._sock.close()
            raise RuntimeError(f"NAO microphone stream unavailable: {header.decode('utf-8', 'replace').strip()}")
        self._sock.settimeout(self._timeout)
        self._buf = b""
        return self

    def read(self, n: int) -> np.ndarray:
        """Return n samples as float32 in [-1, 1]; silence if the robot stops sending."""
        need = 2 * n
        try:
            while len(self._buf) < need:
                chunk = self._sock.recv(max(4096, need - len(self._buf)))
                if not chunk:
                    raise ConnectionError("NAO microphone stream closed")
                self._buf += chunk
        except socket.timeout:
            return np.zeros(n, dtype=np.float32)
        data, self._buf = self._buf[:need], self._buf[need:]
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0

    def __exit__(self, *exc):
        try:
            self._sock.close()
        finally:
            self._sock = None
        return False


class NaoBackend(RobotBackend):
    """SoftBank NAO or Pepper via the speaker server that runs on the robot."""

    def __init__(self, ip: str, port: int = 9600, naoqi_port: int = 9559, password: str = "nao",
                 mic_port: Optional[int] = None):
        self._ip, self._port = ip, port
        self._mic_port = mic_port or port + 1
        self._output = NAOAudioOutput(ip=ip, port=port)
        self._adapter = RealNAO(ip, port, naoqi_port, password)
        self.capabilities = Capabilities(
            robot="NAO/Pepper", speech="robot TTS (ALTextToSpeech)", interrupt="verified", expressions=False,
            listening="robot front microphone, streamed to local VAD + ASR",
            notes="listening/speaking arm-pose cycle on the robot; interrupt via ALTextToSpeech.stopAll",
        )

    def connect(self) -> None:
        self._adapter.connect()
        if not self._adapter.is_connected():
            raise RuntimeError(
                f"NAO speaker server not reachable at {self._ip}:{self._port}. "
                f"Run: python deploy_nao.py (starts the speaker server on the robot); "
                f"see 'Troubleshooting the NAO connection' in README.md."
            )

    def mic_source(self):
        return lambda: NaoMicSource(self._ip, self._mic_port)

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        return self._output.speak_text(text)

    def stop(self) -> bool:
        return self._output.stop()

    def on_listening(self) -> None:
        self._adapter.on_listening()

    def on_idle(self) -> None:
        self._adapter.on_idle()
