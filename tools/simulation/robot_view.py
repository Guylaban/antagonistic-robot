"""Reachy Mini view for rehearsal videos, in two passes (nothing is captured from the screen).

1. record: sample the robot's joint positions from the Reachy Mini daemon (SDK) at 25 Hz during
   the session, with wall-clock times (about 1 ms per sample, so the session is not slowed down).
2. render: replay the samples through the robot's MuJoCo model and render each frame offscreen
   to an MP4 (0.1-0.4 s per frame on a laptop, so this runs after the session).

Usage:
    python robot_view.py record --traj traj.jsonl --stamp stamp.json --stop-file STOP [--host localhost]
    python robot_view.py render --traj traj.jsonl --out robot.mp4 [--shadows]
"""
import argparse
import json
import os
import subprocess
import time
from importlib.resources import files

import numpy as np

W, H, FPS = 720, 900, 25


def build():
    import mujoco
    import reachy_mini
    from reachy_mini.reachy_mini import SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_JOINT_POSITIONS
    root = str(files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/"))
    model = mujoco.MjModel.from_xml_path(f"{root}/scenes/empty.xml")
    model.opt.timestep = 0.002
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    addr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    sleep = list(SLEEP_HEAD_JOINT_POSITIONS) + [SLEEP_ANTENNAS_JOINT_POSITIONS[1], SLEEP_ANTENNAS_JOINT_POSITIONS[0]]
    data.qpos[addr] = sleep
    data.ctrl[:] = sleep
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.62, 160, -12
    cam.lookat[:] = [0, 0, 0.17]
    return model, data, cam


def apply(model, data, head, antennas, substeps=20):
    import mujoco
    data.ctrl[:7] = head
    data.ctrl[-2:] = -np.asarray(antennas)   # same convention as the daemon's MuJoCo backend
    for _ in range(substeps):
        mujoco.mj_step(model, data)


def record(a):
    from reachy_mini import ReachyMini
    with ReachyMini(host=a.host, connection_mode="localhost_only" if a.host in ("localhost", "127.0.0.1") else "network",
                    media_backend="no_media", timeout=15) as mini:
        with open(a.traj, "w") as out:
            t0 = time.time()
            json.dump({"robot_video_start": t0}, open(a.stamp, "w"))
            n = 0
            while not os.path.exists(a.stop_file):
                head, ant = mini.get_current_joint_positions()
                out.write(json.dumps({"t": time.time(), "head": list(head), "ant": list(ant)}) + "\n")
                n += 1
                time.sleep(max(0.0, t0 + n / FPS - time.time()))
    print(f"recorded {n} samples ({n / FPS:.1f} s)")


def render(a):
    import mujoco
    samples = [json.loads(l) for l in open(a.traj)]
    model, data, cam = build()
    r = mujoco.Renderer(model, height=H, width=W)
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-preset", "veryfast", "-crf", "20", a.out], stdin=subprocess.PIPE)
    t0 = time.time()
    for i, s in enumerate(samples):
        apply(model, data, s["head"], s["ant"])
        r.update_scene(data, camera=cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = a.shadows
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = a.shadows
        proc.stdin.write(r.render().tobytes())
        if i % 500 == 0:
            print(f"  frame {i}/{len(samples)} ({time.time() - t0:.0f} s)", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"rendered {len(samples)} frames to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "render"])
    ap.add_argument("--traj", required=True)
    ap.add_argument("--stamp")
    ap.add_argument("--stop-file")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--out")
    ap.add_argument("--shadows", action="store_true")
    a = ap.parse_args()
    {"record": record, "render": render}[a.mode](a)
