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


def test_nao_microphone_stream_delivers_robot_audio(mock_robot, mock_mic_wav):
    import numpy as np
    from antagonist_robot.robots.nao import NaoMicSource
    ip, port = mock_robot
    _, expected = mock_mic_wav
    with NaoMicSource(ip, port + 1) as mic:
        got = np.concatenate([mic.read(512) for _ in range(20)])      # 10240 samples, 0.64 s
    np.testing.assert_allclose(got, expected[:len(got)] / 32768.0, atol=1e-6)
    # A new connection starts a fresh stream (audio from before listening is never reused)
    with NaoMicSource(ip, port + 1) as mic:
        again = mic.read(512)
    np.testing.assert_allclose(again, expected[:512] / 32768.0, atol=1e-6)
