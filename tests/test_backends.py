"""Robot backends with fake SDK clients: speech, interruption, cues, factory."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from antagonist_robot.config.settings import (FurhatConfig, NAOConfig, ReachyMiniConfig, RobotConfig)
from antagonist_robot.robots import create_backend
from antagonist_robot.robots.base import SpeechCue, expression_key
from antagonist_robot.robots.furhat import FurhatBackend
from antagonist_robot.robots.reachy_mini import ReachyMiniBackend, resample
from antagonist_robot.robots.text import TextBackend


def speak_in_thread(backend, text, cue=None):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("done", backend.speak(text, cue)), daemon=True)
    th.start()
    return th, out


def test_expression_keys():
    assert expression_key(SpeechCue(-2, "D")) == "support"
    assert expression_key(SpeechCue(0, "D")) == "neutral"
    assert expression_key(SpeechCue(2, "F")) == "F"


def test_text_backend_speaks_and_stops():
    b = TextBackend(seconds_per_word=0.2)
    assert b.speak("one two") is True
    th, out = speak_in_thread(b, " ".join(["w"] * 50))
    time.sleep(0.2)
    b.stop()
    th.join(2)
    assert out["done"] is False


class FakeFurhat:
    def __init__(self, say_s=5.0):
        self.say_s, self.calls = say_s, []

    def get_voices(self):
        return [SimpleNamespace(name="Matthew")]

    def set_voice(self, name):
        self.calls.append(("voice", name))

    def say(self, text, blocking=True):
        self.calls.append(("say", text))
        time.sleep(self.say_s)            # like the virtual Furhat: blocks for the whole utterance

    def say_stop(self):
        self.calls.append(("say_stop",))

    def gesture(self, name, blocking=False):
        self.calls.append(("gesture", name))

    def set_led(self, red, green, blue):
        self.calls.append(("led", red, green, blue))

    def attend(self, user):
        self.calls.append(("attend", user))


def test_furhat_stop_returns_control_immediately():
    f = FakeFurhat(say_s=5.0)
    b = FurhatBackend(client=f)
    b.connect()
    th, out = speak_in_thread(b, "a long reply")
    time.sleep(0.2)
    t0 = time.monotonic()
    assert b.stop()
    th.join(2)
    assert out["done"] is False and time.monotonic() - t0 < 1.0     # did not wait 5 s
    assert ("say_stop",) in f.calls
    assert b.capabilities.interrupt == "unverified"


def test_furhat_cues_only_when_enabled():
    f = FakeFurhat(say_s=0.0)
    FurhatBackend(client=f, expressions=False).speak("x", SpeechCue(2, "F"))
    assert not any(c[0] == "gesture" for c in f.calls)
    FurhatBackend(client=f, expressions=True).speak("x", SpeechCue(2, "F"))
    assert ("gesture", "ExpressAnger") in f.calls


def test_furhat_connect_errors_clearly():
    class Down(FakeFurhat):
        def get_voices(self):
            raise ConnectionError("refused")
    with pytest.raises(RuntimeError, match="Remote API skill"):
        FurhatBackend(client=Down()).connect()
    with pytest.raises(RuntimeError, match="not available"):
        FurhatBackend(client=FakeFurhat(), voice="Nobody").connect()


class FakeMedia:
    def __init__(self):
        self.pushed, self.started, self.stopped = 0, 0, 0

    def get_output_audio_samplerate(self):
        return 16000

    def get_output_channels(self):
        return 2

    def start_playing(self):
        self.started += 1

    def stop_playing(self):
        self.stopped += 1

    def push_audio_sample(self, data):
        assert data.dtype == np.float32 and data.shape[1] == 2
        self.pushed += len(data)


class FakeMini:
    def __init__(self):
        self.media, self.poses = FakeMedia(), []

    def goto_target(self, head=None, antennas=None, duration=0.5):
        self.poses.append((np.round(antennas, 3).tolist(), duration))


class FakeTTS:
    def synthesize(self, text):
        return np.zeros(22050 * 2, dtype=np.float32), 22050     # 2 s of audio


@pytest.fixture(autouse=True)
def fake_reachy_utils(monkeypatch):
    """Stand-in for reachy_mini.utils so the real backend code runs without the SDK installed."""
    import sys
    import types
    pkg, utils = types.ModuleType("reachy_mini"), types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = lambda **kw: np.eye(4)
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "reachy_mini", pkg)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)


def fake_reachy(**kw):
    return ReachyMiniBackend(mini=FakeMini(), tts=FakeTTS(), **kw)


def test_reachy_streams_and_stops_within_a_chunk():
    b = fake_reachy()
    t0 = time.monotonic()
    assert b.speak("hello") is True
    assert 1.8 < time.monotonic() - t0 < 2.8                  # paced in real time
    assert b._mini.media.pushed == 32000                      # 2 s at 16 kHz, resampled from 22.05 kHz
    th, out = speak_in_thread(b, "hello")
    time.sleep(0.5)
    t0 = time.monotonic()
    b.stop()
    th.join(1)
    assert out["done"] is False and time.monotonic() - t0 < 0.3
    assert b._mini.media.stopped == 2


def test_reachy_cues_only_when_enabled():
    b = fake_reachy(expressions=False)
    b.speak("x", SpeechCue(2, "F"))
    assert b._mini.poses == []
    b = fake_reachy(expressions=True)
    b.speak("x", SpeechCue(2, "F"))
    assert any(p[0] == np.round(np.deg2rad([-35, -35]), 3).tolist() for p in b._mini.poses)


def test_resample_length():
    assert len(resample(np.zeros(22050, np.float32), 22050, 16000)) == 16000


def test_factory_builds_each_backend():
    cfg = SimpleNamespace(robot=RobotConfig(backend="text"), nao=NAOConfig(), furhat=FurhatConfig(),
                          reachy_mini=ReachyMiniConfig())
    for name, cls in [("text", "TextBackend"), ("nao", "NaoBackend"), ("furhat", "FurhatBackend"),
                      ("reachy_mini", "ReachyMiniBackend")]:
        cfg.robot.backend = name
        assert type(create_backend(cfg)).__name__ == cls
    cfg.robot.backend = "pepper3000"
    with pytest.raises(ValueError):
        create_backend(cfg)
