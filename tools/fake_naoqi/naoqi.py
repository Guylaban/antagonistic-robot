"""Minimal stand-in for the NAOqi SDK, for running nao_speaker_server.py off-robot.

ALTextToSpeech.say() prints the text and blocks for a speaking time
proportional to its length (about 0.3 s per word, similar to NAO at
speed 85), and returns early when stopAll() is called, as on the robot.

ALAudioDevice delivers "microphone" buffers to a subscribed ALModule the way
NAOqi does (processRemote with 16 kHz mono signed 16-bit buffers, about
every 85 ms). The audio comes from the WAV file named in the environment
variable RAWR_FAKE_MIC_WAV (16 kHz mono), played once per subscription and
followed by silence; without it, the microphone hears silence.

Motion, posture, and Autonomous Life calls are accepted and ignored.
"""

import os
import sys
import threading
import time
import wave

SECONDS_PER_WORD = 0.3
MIC_CHUNK = 1365          # samples per buffer, ~85 ms at 16 kHz (NAO's own buffer size)
_stop = threading.Event()
_modules = {}


class ALBroker(object):
    def __init__(self, *args, **kwargs):
        pass


class ALModule(object):
    def __init__(self, name):
        _modules[name] = self


def _mic_samples():
    path = os.environ.get("RAWR_FAKE_MIC_WAV")
    if not path:
        return b""
    w = wave.open(path, "rb")
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono s16 WAV"
    data = w.readframes(w.getnframes())
    w.close()
    return data


class _Proxy(object):
    def __init__(self, name):
        self._name = name
        self._subs = {}

    # ALTextToSpeech
    def say(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        print("[MOCK NAO] says: %s" % text)
        sys.stdout.flush()
        _stop.clear()
        _stop.wait(SECONDS_PER_WORD * max(1, len(text.split())))

    def stopAll(self):
        _stop.set()

    def setParameter(self, *args):
        pass

    # ALAudioDevice
    def setClientPreferences(self, name, sample_rate, channels, deinterleave):
        assert sample_rate == 16000 and channels == 3, "the speaker server asks for 16 kHz front mic"

    def subscribe(self, name):
        stop = threading.Event()
        self._subs[name] = stop

        def run():
            data, pos, step = _mic_samples(), 0, MIC_CHUNK * 2
            t0 = time.time()
            i = 0
            while not stop.is_set():
                chunk = data[pos:pos + step]
                pos += step
                if len(chunk) < step:
                    chunk = chunk + b"\x00" * (step - len(chunk))   # silence after the file
                _modules[name].processRemote(1, MIC_CHUNK, [0, 0], chunk)
                i += 1
                time.sleep(max(0, t0 + i * MIC_CHUNK / 16000.0 - time.time()))

        th = threading.Thread(target=run)
        th.daemon = True
        th.start()

    def unsubscribe(self, name):
        stop = self._subs.pop(name, None)
        if stop:
            stop.set()

    # ALMotion / ALRobotPosture / ALAutonomousLife
    def setAngles(self, *args):
        pass

    def wakeUp(self):
        pass

    def goToPosture(self, *args):
        return True

    def setState(self, *args):
        pass


def ALProxy(name, ip=None, port=None):
    return _Proxy(name)
