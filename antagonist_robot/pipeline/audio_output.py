"""Speech output on the NAO/Pepper robot via nao_speaker_server.py.

The robot speaks with its own ALTextToSpeech engine. The PC sends text
over TCP; the speaker server on the robot speaks it and acknowledges.

Protocol (one newline-terminated UTF-8 line per connection):
    <text>        speak the text; reply "ok" when finished ("stopped" if interrupted)
    __STOP__      interrupt any ongoing speech immediately; reply "stopped"
    __PING__      health check; reply "pong" (nothing is spoken)
"""

import socket

from antagonist_robot.nao.host import resolve_ipv4

STOP_COMMAND = "__STOP__"
PING_COMMAND = "__PING__"


class NAOAudioOutput:
    """Sends robot speech to nao_speaker_server.py and can interrupt it."""

    def __init__(self, ip: str, port: int, speak_timeout_s: float = 60.0):
        self._ip = ip
        self._port = port
        self._speak_timeout_s = speak_timeout_s

    def _request(self, line: str, timeout: float) -> bytes:
        ip = resolve_ipv4(self._ip, self._port)
        with socket.create_connection((ip, self._port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((line.strip() + "\n").encode("utf-8"))
            response = b""
            while not response.endswith(b"\n"):
                chunk = s.recv(64)
                if not chunk:
                    break
                response += chunk
        return response.strip()

    def speak_text(self, text: str) -> bool:
        """Speak text on the robot. Blocks until the robot finishes.

        Returns:
            True if the speech completed, False if it was interrupted by stop().

        Raises:
            RuntimeError: if the robot is unreachable or never acknowledges,
                so the turn fails loudly instead of being logged as spoken.
        """
        try:
            response = self._request(text, timeout=self._speak_timeout_s)
        except OSError as e:
            raise RuntimeError(f"NAO speaker server at {self._ip}:{self._port} failed: {e}") from e
        if response == b"ok":
            return True
        if response == b"stopped":
            return False
        raise RuntimeError(
            f"NAO speaker server at {self._ip}:{self._port} closed without acknowledging speech"
        )

    def stop(self) -> bool:
        """Interrupt the robot's speech immediately (operator emergency stop)."""
        try:
            return self._request(STOP_COMMAND, timeout=3.0) == b"stopped"
        except OSError:
            return False

    def ping(self) -> bool:
        """True if the speaker server answers."""
        try:
            return self._request(PING_COMMAND, timeout=3.0) == b"pong"
        except OSError:
            return False
