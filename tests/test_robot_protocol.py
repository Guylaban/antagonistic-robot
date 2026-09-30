"""nao_speaker_server.py protocol, run against the mock robot (real server code, fake NAOqi)."""

import threading
import time

from antagonist_robot.nao.real import RealNAO
from antagonist_robot.pipeline.audio_output import NAOAudioOutput


def test_ping_and_adapter_connect(mock_robot):
    ip, port = mock_robot
    assert NAOAudioOutput(ip, port).ping()
    nao = RealNAO(ip, port)
    nao.connect()
    assert nao.is_connected()


def test_speak_completes(mock_robot):
    assert NAOAudioOutput(*mock_robot).speak_text("Hello there.") is True


def test_stop_interrupts_speech(mock_robot):
    out = NAOAudioOutput(*mock_robot)
    result = {}
    t0 = time.monotonic()
    th = threading.Thread(target=lambda: result.setdefault("done", out.speak_text(" ".join(["word"] * 40))))
    th.start()
    time.sleep(0.5)
    assert out.stop() is True
    th.join(timeout=5)
    assert result["done"] is False                   # reported as interrupted
    assert time.monotonic() - t0 < 3.0               # 40 words would take ~12 s


def test_unreachable_robot_fails_loudly():
    out = NAOAudioOutput("127.0.0.1", 1)
    assert out.ping() is False
    try:
        out.speak_text("hello")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")
