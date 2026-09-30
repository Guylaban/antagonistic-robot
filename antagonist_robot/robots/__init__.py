"""Robot backends. Select one with robot.backend in config.yaml."""

from antagonist_robot.robots.base import Capabilities, RobotBackend, SpeechCue

BACKENDS = ("nao", "furhat", "reachy_mini", "text")


def create_backend(config) -> RobotBackend:
    """Build the backend named by config.robot.backend (imports optional SDKs lazily)."""
    name = config.robot.backend
    expressions = config.robot.expressions
    if name == "nao":
        from antagonist_robot.robots.nao import NaoBackend
        n = config.nao
        return NaoBackend(n.ip, n.port, n.naoqi_port, n.password)
    if name == "furhat":
        from antagonist_robot.robots.furhat import FurhatBackend
        f = config.furhat
        return FurhatBackend(host=f.host, voice=f.voice, expressions=expressions, tts_engine=f.tts_engine,
                             tts_voice=f.tts_voice, tts_rate=f.tts_rate, audio_host=f.audio_host,
                             audio_port=f.audio_port, speech_log_dir=f.speech_log_dir)
    if name == "reachy_mini":
        from antagonist_robot.robots.reachy_mini import ReachyMiniBackend
        r = config.reachy_mini
        return ReachyMiniBackend(host=r.host, port=r.port, connection_mode=r.connection_mode,
                                 tts_rate=r.tts_rate, tts_voice=r.tts_voice, expressions=expressions,
                                 speech_log_dir=r.speech_log_dir, tts_engine=r.tts_engine,
                                 tts_device=r.tts_device, animate=r.animate)
    if name == "text":
        from antagonist_robot.robots.text import TextBackend
        return TextBackend()
    raise ValueError(f"robot.backend must be one of {BACKENDS}, got {name!r}")


__all__ = ["BACKENDS", "Capabilities", "RobotBackend", "SpeechCue", "create_backend"]
