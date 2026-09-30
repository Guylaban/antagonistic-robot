"""Minimal stand-in for the NAOqi SDK, for running nao_speaker_server.py off-robot.

ALTextToSpeech.say() prints the text and blocks for a speaking time
proportional to its length (about 0.3 s per word, similar to NAO at
speed 85), and returns early when stopAll() is called, as on the robot.
Motion, posture, and Autonomous Life calls are accepted and ignored.
"""

import sys
import threading
import time

SECONDS_PER_WORD = 0.3
_stop = threading.Event()


class _Proxy(object):
    def __init__(self, name):
        self._name = name

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
