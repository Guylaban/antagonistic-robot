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
# The server listens on port 9600 by default.
# Your PC sends a line of text, the robot speaks it, then sends back "ok".

import socket
import math
import threading
import time
from naoqi import ALProxy

LISTEN_PORT = 9600
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

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", LISTEN_PORT))
server.listen(5)

print("[NAO SERVER] Listening on port", LISTEN_PORT)

while True:
    conn, addr = server.accept()
    print("[NAO SERVER] Connection from", addr)
    try:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if data.endswith(b"\n"):
                break
        text = data.strip().decode("utf-8").encode("utf-8")
        if text:
            print("[NAO SERVER] Speaking:", text)
            start_speaking_pose()
            tts.say(text)
            stop_speaking_pose()
        conn.sendall(b"ok\n")
    except Exception as e:
        print("[NAO SERVER] Error:", e)
    finally:
        conn.close()
