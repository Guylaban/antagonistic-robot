"""Check a robot backend before a study: connect, speak, and time Stop speech.

Runs the backend selected in config.yaml (or --robot) through:
  1. connect()
  2. a short utterance (must complete)
  3. a long utterance interrupted by stop() after --stop-after seconds:
     the time until speak() returns, and the time until a follow-up
     utterance completes (if the robot really stopped, the follow-up is
     short; if it kept talking, the follow-up waits behind it)

Results are appended as one JSON line to --out (never overwritten), with
the backend's declared capabilities, so a lab can record how its own
robot behaves.

Usage:
    python tools/robot_smoke_test.py --robot furhat
    python tools/robot_smoke_test.py --robot reachy_mini --out logs/robot_checks.jsonl
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHORT = "This is a short test."
LONG = " ".join(["This long sentence keeps going so that we can test whether stopping works."] * 5)
FOLLOW = "Next line."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--robot", help="nao | furhat | reachy_mini | text")
    ap.add_argument("--nao-ip")
    ap.add_argument("--stop-after", type=float, default=2.0)
    ap.add_argument("--out", default="logs/robot_checks.jsonl")
    ap.add_argument("--note", default="", help="free text stored with the result, e.g. 'virtual Furhat SDK 2.9.2'")
    a = ap.parse_args()

    import os
    os.environ.setdefault("GROK_API_KEY", "not-needed-for-this-check")
    from antagonist_robot.config.settings import load_config
    from antagonist_robot.robots import create_backend

    cfg = load_config(a.config)
    if a.robot:
        cfg.robot.backend = a.robot
    if a.nao_ip:
        cfg.nao.ip = a.nao_ip
    robot = create_backend(cfg)
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "backend": cfg.robot.backend,
           "capabilities": robot.capabilities.as_dict(), "note": a.note}

    t = time.monotonic()
    robot.connect()
    rec["connect_s"] = round(time.monotonic() - t, 2)

    t = time.monotonic()
    rec["short_completed"] = robot.speak(SHORT)
    rec["short_s"] = round(time.monotonic() - t, 2)

    out = {}
    th = threading.Thread(target=lambda: out.setdefault("completed", robot.speak(LONG)), daemon=True)
    t0 = time.monotonic()
    th.start()
    time.sleep(a.stop_after)
    t1 = time.monotonic()
    rec["stop_issued"] = robot.stop()
    th.join(120)
    rec["long_completed"] = out.get("completed")
    rec["speak_returned_after_stop_s"] = round(time.monotonic() - t1, 2)
    t = time.monotonic()
    robot.speak(FOLLOW)
    rec["follow_up_s"] = round(time.monotonic() - t, 2)
    rec["long_total_s"] = round(time.monotonic() - t0, 2)
    rec["stop_effective"] = rec["follow_up_s"] < rec["short_s"] + 2.0
    robot.on_idle()
    robot.close()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
