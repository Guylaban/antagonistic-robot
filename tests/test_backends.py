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
from antagonist_robot.robots.reachy_mini import Animator, DEFAULT_POSES, ReachyMiniBackend, loudness, resample
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
    assert ("gesture", "BrowFrown") in f.calls


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


def test_furhat_recognizer_skips_silence_and_returns_text():
    class Listening(FakeFurhat):
        def __init__(self):
            super().__init__()
            self.replies = ["SILENCE", "", "I think we should split it equally"]

        def listen(self, language="en-US"):
            return SimpleNamespace(success=True, message=self.replies.pop(0))
    b = FurhatBackend(client=Listening())
    rec = b.recognizer()
    audio = rec.record_utterance(lambda: True)
    assert audio is not None and audio.samples.size == 0            # no raw audio from Furhat
    assert rec.transcribe(audio).text == "I think we should split it equally"
    assert "speech recognition" in b.capabilities.listening


def test_reachy_mic_source_mixes_to_mono_16k():
    class Media(FakeMedia):
        def __init__(self):
            super().__init__()
            self.rec = 0
        def start_recording(self): self.rec += 1
        def stop_recording(self): self.rec -= 1
        def get_input_audio_samplerate(self): return 16000
        def get_audio_sample(self):
            return np.stack([np.full(800, 0.5, np.float32), np.full(800, 0.1, np.float32)], axis=1)
    from antagonist_robot.robots.reachy_mini import ReachyMicSource
    m = Media()
    with ReachyMicSource(m) as mic:
        frame = mic.read(512)
        assert m.rec == 1
    assert m.rec == 0 and frame.shape == (512,) and np.allclose(frame, 0.3)


def test_text_backend_has_no_microphone():
    b = TextBackend()
    assert b.mic_source() is None and b.recognizer() is None


class TwoSentenceTTS:
    """Streams two 0.5 s sentences; the second arrives 0.4 s after the first has finished playing."""

    def stream(self, text):
        yield np.full(8000, 0.1, np.float32), 16000
        time.sleep(0.9)
        yield np.full(8000, 0.1, np.float32), 16000


def test_reachy_streams_sentences_and_archives_gaps(tmp_path):
    import json
    import wave
    b = ReachyMiniBackend(mini=FakeMini(), tts=TwoSentenceTTS(), speech_log_dir=str(tmp_path))
    assert b.speak("One. Two.") is True
    rec = json.loads((tmp_path / "speech_log.jsonl").read_text())
    with wave.open(str(tmp_path / rec["file"])) as w:
        seconds = w.getnframes() / w.getframerate()
    assert b._mini.media.pushed == 16000                       # both sentences played
    assert 1.3 < seconds < 1.6 and rec["interrupted"] is False  # archive keeps the 0.4 s silence in place


def test_animator_moves_with_speech_and_condition():
    a = Animator(mini=None, poses=DEFAULT_POSES)
    t = 100.0

    def run(seconds):
        nonlocal t
        out = []
        for _ in range(int(seconds * 50)):
            t += 0.02
            out.append(a.tick(t, 0.02))
        return out

    a.set_state("F", 0.4)
    a.set_style("F")
    run(2.0)                                                         # settle into the F pose
    quiet = run(2.0)                                                 # same pose, not speaking
    speech = np.sin(np.linspace(0, 2 * np.pi * 400, 16000 * 2)).astype(np.float32)
    speech *= np.tile(np.r_[np.ones(4000), np.zeros(4000)], 4)       # syllable-like bursts
    a.feed(speech, 16000, t)
    loud = run(2.0)
    pitch_quiet = np.ptp([h[1] for h, _, _ in quiet])
    pitch_loud = np.ptp([h[1] for h, _, _ in loud])
    assert pitch_loud > 2 * pitch_quiet                              # nods with its own speech
    assert np.mean([ant[0] for _, _, ant in loud[-25:]]) < -20      # antennas back for condition F
    assert all(abs(h[0]) <= 20 and abs(h[1]) <= 20 for h, _, _ in loud)


def test_loudness_scale():
    assert loudness(np.zeros(1600, np.float32), 16000).max() == 0
    assert loudness(np.full(1600, 0.3, np.float32), 16000).min() == 1


class UrlFurhat(FakeFurhat):
    def say(self, text=None, url=None, lipsync=False, blocking=True):
        import urllib.request
        data = urllib.request.urlopen(url).read() if url else None
        self.calls.append(("say", text, url is not None, lipsync, len(data or b"")))


def test_furhat_plays_rawr_audio_with_lipsync(tmp_path):
    f = UrlFurhat(say_s=0.0)
    b = FurhatBackend(client=f, tts_engine="kokoro", tts=TwoSentenceTTS(), audio_port=18095,
                      speech_log_dir=str(tmp_path))
    b.connect()
    assert b.speak("One. Two.") is True
    says = [c for c in f.calls if c[0] == "say"]
    assert len(says) == 2 and all(c[2] and c[3] and c[4] > 16000 for c in says)   # WAV fetched by URL, lip-synced
    assert len((tmp_path / "speech_log.jsonl").read_text().splitlines()) == 2
    b._audio.close()


def test_furhat_gestures_follow_the_antagonism_level():
    strong, mild = FakeFurhat(say_s=3.6), FakeFurhat(say_s=3.6)
    FurhatBackend(client=strong, expressions=True).speak("x", SpeechCue(3, "F", 3))      # strength 1.0
    FurhatBackend(client=mild, expressions=True).speak("x", SpeechCue(1, "F", 1))        # strength 0.33
    assert [c[1] for c in strong.calls if c[0] == "gesture"] == ["BrowFrown", "Shake", "ExpressAnger"]
    assert [c[1] for c in mild.calls if c[0] == "gesture"] == ["BrowFrown"]


def test_expression_follows_the_reply_not_only_the_request():
    from antagonist_robot.robots.base import expression_strength
    asked_d = SpeechCue(2, "D", 2)
    assert expression_key(asked_d) == "D"
    assert expression_key(SpeechCue(2, "D", 2, exhibited_category="C")) == "C"         # judge saw sarcasm
    assert expression_key(SpeechCue(3, "F", 3, exhibited_category="NEUTRAL")) == "neutral"   # softened reply
    assert expression_key(SpeechCue(2, "F", 2, exhibited_category="REFUSAL")) == "neutral"
    assert expression_strength(SpeechCue(3, "F", 3)) == 1.0
    assert expression_strength(SpeechCue(1, "D", 1)) == round(1 / 3, 3)
    assert expression_strength(SpeechCue(3, "F", 3, exhibited_category="F", exhibited_intensity=1)) == round(2 / 3, 3)
    assert expression_strength(SpeechCue(-3, "D")) == 1.0 and expression_strength(SpeechCue(0, "D")) == 0.0


def test_reachy_pose_scales_with_antagonism():
    from antagonist_robot.robots.reachy_mini import expression_gain
    strong, mild = fake_reachy(expressions=True), fake_reachy(expressions=True)
    strong.speak("x", SpeechCue(3, "F", 3))
    mild.speak("x", SpeechCue(1, "F", 1))
    first = lambda b: b._mini.poses[0][0]                                     # antennas of the condition pose
    assert abs(first(strong)[0]) > abs(first(mild)[0])
    assert expression_gain(SpeechCue(0, "D")) == 1.0 and expression_gain(SpeechCue(3, "F", 3)) == 1.5
