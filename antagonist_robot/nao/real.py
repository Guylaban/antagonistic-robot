"""Real NAO adapter — communicates with the robot via TCP.

The PC does NOT use the naoqi SDK directly. All robot interaction
(TTS, gestures) is handled by nao_speaker_server.py running on the
NAO robot. This adapter verifies reachability and logs state
transitions; gesture control is managed server-side.
"""

import logging
import socket

from antagonist_robot.nao.base import NAOAdapter
from antagonist_robot.nao.host import resolve_ipv4

logger = logging.getLogger(__name__)


class RealNAO(NAOAdapter):
    """Real NAO robot adapter via TCP to nao_speaker_server.py.

    The adapter verifies that the robot is reachable on the speaker
    server port (default 9600) during connect(). Gesture and posture
    control is handled entirely by nao_speaker_server.py on the robot.
    """

    def __init__(self, ip: str, speaker_port: int = 9600, naoqi_port: int = 9559, password: str = "nao"):
        self._ip = ip
        self._speaker_port = speaker_port
        self._port = naoqi_port
        self._password = password
        self._connected = False

    def connect(self) -> None:
        """Verify the robot is reachable by TCP-pinging the speaker server port.

        Attempts a TCP connection to the nao_speaker_server.py port
        (nao.port in config). If the connection succeeds, the robot is
        considered reachable; check is_connected() afterwards.
        """
        speaker_port = self._speaker_port
        try:
            ip = resolve_ipv4(self._ip, speaker_port)
            with socket.create_connection(
                (ip, speaker_port), timeout=5
            ):
                pass  # Connection succeeded — robot is reachable
            self._connected = True
            logger.info(
                "[RealNAO] Connected — robot reachable at %s (%s):%d",
                self._ip, ip, speaker_port,
            )
        except OSError as exc:
            self._connected = False
            logger.warning(
                "[RealNAO] Cannot reach robot at %s:%d — %s",
                self._ip, speaker_port, exc,
            )

    def disconnect(self) -> None:
        """Mark as disconnected and log."""
        self._connected = False
        logger.info("[RealNAO] Disconnected")

    def on_response(self, text: str, hostility_level: int) -> None:
        """Log the polar level. Gestures are handled by nao_speaker_server.py."""
        logger.info(
            "[RealNAO] on_response: polar level %d (gestures handled server-side)",
            hostility_level,
        )

    def on_listening(self) -> None:
        """Log that the system entered listening state."""
        logger.info("[RealNAO] on_listening: waiting for user speech")

    def on_idle(self) -> None:
        """Log that the system entered idle state."""
        logger.info("[RealNAO] on_idle: session idle")

    def is_connected(self) -> bool:
        """Return True if the robot was reachable at last connect() call."""
        return self._connected
