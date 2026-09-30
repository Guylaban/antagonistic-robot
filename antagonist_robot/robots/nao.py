"""NAO / Pepper backend: text to nao_speaker_server.py, spoken by NAOqi ALTextToSpeech."""

from typing import Optional

from antagonist_robot.nao.real import RealNAO
from antagonist_robot.pipeline.audio_output import NAOAudioOutput
from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue


class NaoBackend(RobotBackend):
    """SoftBank NAO or Pepper via the TCP speaker server that runs on the robot."""

    def __init__(self, ip: str, port: int = 9600, naoqi_port: int = 9559, password: str = "nao"):
        self._ip, self._port = ip, port
        self._output = NAOAudioOutput(ip=ip, port=port)
        self._adapter = RealNAO(ip, port, naoqi_port, password)
        self.capabilities = Capabilities(
            robot="NAO/Pepper", speech="robot TTS (ALTextToSpeech)", interrupt="verified", expressions=False,
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

    def speak(self, text: str, cue: Optional[SpeechCue] = None) -> bool:
        return self._output.speak_text(text)

    def stop(self) -> bool:
        return self._output.stop()

    def on_listening(self) -> None:
        self._adapter.on_listening()

    def on_idle(self) -> None:
        self._adapter.on_idle()
