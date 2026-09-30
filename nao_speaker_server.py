# -*- coding: utf-8 -*-
from __future__ import print_function
# nao_speaker_server.py
# Runs ON the NAO robot in Python 2.7.
# Listens for text over a TCP socket and speaks it via NAOqi ALTextToSpeech.
#
# HOW TO RUN ON THE ROBOT:
#   ssh nao@<robot_ip>
#   python nao_speaker_server.py
#
# The server listens on port 9600 by default (override: --port N).
# Protocol, one newline-terminated UTF-8 line per connection:
#   <text>     speak it; reply "ok" when done, or "stopped" if interrupted
#   __STOP__   interrupt ongoing speech now (operator emergency stop); reply "stopped"
#   __PING__   health check; reply "pong"
# Each connection is handled in its own thread so that __STOP__ can arrive
# while the robot is still speaking. Keep this file Python 2.7 compatible.
#
# Microphone stream on port 9601 (override: --audio-port N): while a client
# is connected, the robot's front microphone (ALAudioDevice, 16 kHz mono,
# signed 16-bit little-endian) is streamed to it, after the header line
# "RAWRMIC 16000 1 s16le". The PC connects only while the participant may
# speak, so the robot's own speech is not recorded.

import socket
import math
import sys
import threading
import time
from naoqi import ALProxy

LISTEN_PORT = 9600
if "--port" in sys.argv:
    LISTEN_PORT = int(sys.argv[sys.argv.index("--port") + 1])
AUDIO_PORT = LISTEN_PORT + 1
if "--audio-port" in sys.argv:
    AUDIO_PORT = int(sys.argv[sys.argv.index("--audio-port") + 1])
ROBOT_IP    = "127.0.0.1"   # NAOqi runs locally on the robot
NAOQI_PORT  = 9559



def make_proxy(name, attempts=10):
    """Create an ALProxy, retrying while the NAOqi broker starts up.

    Port 9559 accepts connections before the broker is ready, so the first
    ALProxy call can fail with 'Cannot connect' on a healthy robot.
    """
    for i in range(attempts):
        try:
            return ALProxy(name, ROBOT_IP, NAOQI_PORT)
        except Exception as e:
            print("[NAO SERVER] %s not ready (%d/%d): %s" % (name, i + 1, attempts, e))
            time.sleep(2)
    raise RuntimeError("Could not connect to %s" % name)


tts     = make_proxy("ALTextToSpeech")
tts_ctl = make_proxy("ALTextToSpeech")   # separate proxy used only to interrupt speech
motion  = make_proxy("ALMotion")
posture = make_proxy("ALRobotPosture")

# Autonomous Life fights manual arm commands (and after a fall it sits in
# 'safeguard', where goToPosture will not take). Disable it, then wake up.
try:
    make_proxy("ALAutonomousLife").setState("disabled")
except Exception as e:
    print("[NAO SERVER] Could not disable Autonomous Life:", e)
motion.wakeUp()

# Slow down and lower the pitch so the robot sounds more natural
tts.setParameter("speed", 85)       # default 100, range ~50-200
tts.setParameter("pitchShift", 0.9) # default 1.0, lower = deeper voice

# Stand up when the server starts
posture.goToPosture("StandInit", 0.5)

# ------------------------------------------------------------------
# Arm gesture helpers
# Joint order: [RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll,
#               LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll]
# ------------------------------------------------------------------

JOINT_NAMES = [
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
]

# listening: right hand near ear, left arm relaxed
ANGLES_LISTENING = [
    -0.3,            # RShoulderPitch  -- slight forward lift
    -math.radians(75),  # RShoulderRoll   -- arm up ~75 deg sideways
     1.2,            # RElbowYaw       -- rotate forearm toward head
     1.7,            # RElbowRoll      -- strong bend so hand reaches ear
     0.0,            # LShoulderPitch  -- relaxed
     0.0,            # LShoulderRoll
     0.0,            # LElbowYaw
     0.0,            # LElbowRoll
]

# speaking: right arm raised/presenting, left arm relaxed
ANGLES_SPEAKING = [
    math.radians(60),  # RShoulderPitch  -- arm raised to chest height
   -0.15,              # RShoulderRoll   -- slight inward
    1.0,               # RElbowYaw       -- palm up/forward
    0.3,               # RElbowRoll      -- slight bend
    0.0,               # LShoulderPitch
    0.0,               # LShoulderRoll
    0.0,               # LElbowYaw
    0.0,               # LElbowRoll
]

# neutral: arms relaxed at sides
ANGLES_NEUTRAL = [0.0] * 8


def set_arms(angles, speed=0.15):
    """Move arm joints to the given angles at the given fractional speed (0-1)."""
    try:
        motion.setAngles(JOINT_NAMES, angles, speed)
    except Exception as e:
        print("[NAO SERVER] motion.setAngles error:", e)


# ------------------------------------------------------------------
# Background thread: gentle oscillation while speaking
# ------------------------------------------------------------------

_speaking_thread = None
_stop_speaking   = threading.Event()


def _speaking_animation():
    """Oscillate RShoulderPitch slightly while speaking is active."""
    t = 0.0
    dt = 0.1
    base_pitch = math.radians(60)
    while not _stop_speaking.is_set():
        offset = math.radians(10) * math.sin(1.5 * t)
        angles = list(ANGLES_SPEAKING)
        angles[0] = base_pitch + offset   # RShoulderPitch
        set_arms(angles, speed=0.2)
        time.sleep(dt)
        t += dt


def start_speaking_pose():
    global _speaking_thread, _stop_speaking
    _stop_speaking.clear()
    _speaking_thread = threading.Thread(target=_speaking_animation)
    _speaking_thread.daemon = True
    _speaking_thread.start()


def stop_speaking_pose():
    global _speaking_thread
    _stop_speaking.set()
    if _speaking_thread:
        _speaking_thread.join(timeout=1.0)
        _speaking_thread = None
    # Return to listening pose (server is always waiting after speaking)
    set_arms(ANGLES_LISTENING, speed=0.15)


# ------------------------------------------------------------------
# Start in listening pose
# ------------------------------------------------------------------
set_arms(ANGLES_LISTENING, speed=0.1)

_speech_lock = threading.Lock()   # one utterance at a time
_stop_flag = threading.Event()     # set by __STOP__ while an utterance is playing


def read_line(conn):
    """Read one newline-terminated line from a connection."""
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.strip().decode("utf-8")


def speak(text):
    """Speak text with the pose cycle. Returns b"stopped" if interrupted."""
    with _speech_lock:
        _stop_flag.clear()
        print("[NAO SERVER] Speaking:", text.encode("utf-8") if sys.version_info[0] < 3 else text)
        start_speaking_pose()
        try:
            tts.say(text.encode("utf-8") if sys.version_info[0] < 3 else text)
        finally:
            stop_speaking_pose()
        return b"stopped" if _stop_flag.is_set() else b"ok"


def stop_speech():
    """Interrupt ongoing and queued speech immediately."""
    _stop_flag.set()
    try:
        tts_ctl.stopAll()
    except Exception as e:
        print("[NAO SERVER] stopAll error:", e)
    print("[NAO SERVER] Speech stopped by operator")


def handle(conn, addr):
    try:
        line = read_line(conn)
        if line == "__STOP__":
            stop_speech()
            reply = b"stopped"
        elif line == "__PING__" or not line:
            reply = b"pong"
        else:
            reply = speak(line)
        conn.sendall(reply + b"\n")
    except Exception as e:
        print("[NAO SERVER] Error:", e)
    finally:
        conn.close()


# ------------------------------------------------------------------
# Microphone stream (ALAudioDevice remote module)
# ------------------------------------------------------------------

MIC_MODULE_NAME = "RawrMic"
MIC_HEADER = b"RAWRMIC 16000 1 s16le\n"
_mic_error = None

try:
    from naoqi import ALBroker, ALModule

    # A local broker lets ALAudioDevice call back into this process.
    _mic_broker = ALBroker("rawrMicBroker", "0.0.0.0", 0, ROBOT_IP, NAOQI_PORT)

    class RawrMicModule(ALModule):
        """Receives microphone buffers from ALAudioDevice and forwards them to clients."""

        def __init__(self, name):
            ALModule.__init__(self, name)
            self.clients = []
            self.lock = threading.Lock()
            self.audio = make_proxy("ALAudioDevice")
            # 16 kHz allows one channel only; 3 = front microphone; 0 = not deinterleaved
            self.audio.setClientPreferences(name, 16000, 3, 0)

        def add(self, conn):
            with self.lock:
                self.clients.append(conn)
                first = len(self.clients) == 1
            if first:
                self.audio.subscribe(MIC_MODULE_NAME)

        def remove(self, conn):
            with self.lock:
                if conn in self.clients:
                    self.clients.remove(conn)
                last = not self.clients
            if last:
                try:
                    self.audio.unsubscribe(MIC_MODULE_NAME)
                except Exception:
                    pass

        def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
            with self.lock:
                clients = list(self.clients)
            for c in clients:
                try:
                    c.sendall(inputBuffer)
                except Exception:
                    self.remove(c)

    # NAOqi finds the module through a global variable with the module's name.
    RawrMic = RawrMicModule(MIC_MODULE_NAME)
except Exception as e:
    RawrMic = None
    _mic_error = str(e)
    print("[NAO SERVER] Microphone stream unavailable:", e)


def handle_mic(conn, addr):
    """Stream the microphone to one client until it disconnects."""
    try:
        if RawrMic is None:
            conn.sendall(("RAWRMIC ERROR %s\n" % _mic_error).encode("utf-8"))
            return
        conn.sendall(MIC_HEADER)
        RawrMic.add(conn)
        while True:
            if not conn.recv(64):   # the client closes the connection to stop
                break
    except Exception:
        pass
    finally:
        if RawrMic is not None:
            RawrMic.remove(conn)
        conn.close()


# Bind before the speech port opens, so a successful __PING__ implies the microphone port is ready.
_mic_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_mic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_mic_sock.bind(("0.0.0.0", AUDIO_PORT))
_mic_sock.listen(2)
print("[NAO SERVER] Microphone stream on port", AUDIO_PORT)
sys.stdout.flush()


def mic_server():
    while True:
        c, a = _mic_sock.accept()
        t = threading.Thread(target=handle_mic, args=(c, a))
        t.daemon = True
        t.start()


_mic_thread = threading.Thread(target=mic_server)
_mic_thread.daemon = True
_mic_thread.start()


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", LISTEN_PORT))
server.listen(5)

print("[NAO SERVER] Listening on port", LISTEN_PORT)
sys.stdout.flush()

while True:
    conn, addr = server.accept()
    worker = threading.Thread(target=handle, args=(conn, addr))
    worker.daemon = True
    worker.start()
