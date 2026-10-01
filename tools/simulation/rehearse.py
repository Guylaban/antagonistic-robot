"""Rehearse a study protocol in simulation before running it with participants.

Starts the simulated robot (unless --no-sim) and the RAWR console, then plays a scenario: a scripted
participant says the scenario's lines, and a scripted operator sets the condition, holds, sends,
tempers, and changes the condition in the console, as a real operator would. Everything goes through
the same pipeline as a study (generation, safety check, fidelity judge and detector, psychosocial
monitor, review gate, robot backend), so you can check what each condition produces, what the gate
holds and why, and how long each step takes, before deployment.

    python tools/simulation/rehearse.py --robot reachy_mini --scenario tools/simulation/scenarios/demo.yaml --out runs/demo
    python tools/simulation/rehearse.py --robot furhat ... --video        # also record a side-by-side video

Robots: reachy_mini (starts `reachy-mini-daemon --sim`, MuJoCo), furhat (the virtual Furhat of the
Furhat SDK with the Remote API skill must already be running), nao (starts tools/mock_nao.py),
text (no robot).

Writes to --out (must be new or empty): config.yaml (the configuration used), participant_script.yaml,
session.db (the session database), console.log, timeline.json (operator actions and moments),
summary.csv (one row per candidate reply: condition, ratings, decision) and, with --video, the
recordings and rehearsal_full.mp4 / rehearsal_short.mp4 (see compose.py).
"""
import argparse
import csv
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import scenario as scn  # noqa: E402


def log(msg):
    print(msg, flush=True)


def port_open(host, port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_http(url, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"{what} did not answer at {url} within {timeout} s")


def kill_tree(proc):
    """Stop a process this script started, with its children (a venv's python.exe starts a child on Windows)."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def participant_audio(sc, out: Path, video: bool):
    """Script for the scripted participant; with --video and Kokoro installed, also the participant's voice."""
    lines = sc["participant"]["lines"]
    voice = sc["participant"].get("voice")
    meta, script = [], []
    tts = None
    if video and voice:
        try:
            from antagonist_robot.robots.tts import KokoroTTS
            tts = KokoroTTS(voice=voice)
            tts.warm_up()
        except Exception as e:
            log(f"participant voice off ({e}); the video will have the robot's voice only")
            tts = None
    for i, text in enumerate(lines, 1):
        if tts:
            import soundfile as sf
            audio, sr = tts.synthesize(text)
            name = f"participant_{i}.wav"
            sf.write(out / name, audio, sr)
            seconds = round(len(audio) / sr, 3)
            meta.append({"file": name, "seconds": seconds, "text": text, "voice": voice})
        else:
            seconds = round(0.4 * len(text.split()) + 0.6, 2)      # about the time it takes to say it
        script.append({"text": text, "delay_s": round(seconds + 0.4, 2)})
    yaml.safe_dump({"utterances": script}, open(out / "participant_script.yaml", "w", encoding="utf-8"),
                   sort_keys=False, width=200)
    if meta:
        json.dump(meta, open(out / "participant_audio.json", "w"), indent=1)


def make_config(base_path, sc, args, out: Path) -> dict:
    cfg = yaml.safe_load(open(base_path, encoding="utf-8")) or {}
    cfg.setdefault("robot", {})["backend"] = args.robot
    if args.no_expressions:
        cfg["robot"]["expressions"] = False
    cfg.setdefault("server", {})["port"] = args.port
    cfg.setdefault("logging", {}).update(db_path=str(out / "session.db"), audio_dir=str(out / "audio"))
    voice = (sc.get("robot_voice") or {}).get(args.robot)
    if args.robot == "reachy_mini":
        r = cfg.setdefault("reachy_mini", {})
        r["speech_log_dir"] = str(out / "robot_speech")
        if voice and r.get("tts_engine", "system") == "system":
            try:
                import kokoro  # noqa: F401
                r.update(tts_engine="kokoro", tts_voice=voice)
            except ImportError:
                log("Kokoro is not installed: Reachy Mini keeps the system voice (see robots/tts.py)")
        elif voice and r.get("tts_engine") == "kokoro" and not r.get("tts_voice"):
            r["tts_voice"] = voice
    elif args.robot == "furhat" and voice and not cfg.get("furhat", {}).get("voice"):
        cfg.setdefault("furhat", {})["voice"] = voice
    elif args.robot == "nao":
        cfg.setdefault("nao", {})["ip"] = "127.0.0.1"
    yaml.safe_dump(cfg, open(out / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    return cfg


def start_simulator(args, cfg, out: Path):
    """Start the robot simulator if needed; returns the process (or None)."""
    if args.robot == "reachy_mini":
        host, port = cfg.get("reachy_mini", {}).get("host", "localhost"), cfg.get("reachy_mini", {}).get("port", 8000)
        if port_open(host, port):
            log(f"Reachy Mini daemon already running on {host}:{port}")
            return None
        exe = shutil.which("reachy-mini-daemon", path=str(Path(sys.executable).parent)) or shutil.which("reachy-mini-daemon")
        if not exe:
            raise RuntimeError("reachy-mini-daemon not found: pip install 'reachy-mini[mujoco]'")
        log("starting the Reachy Mini simulator (MuJoCo, headless)")
        p = subprocess.Popen([exe, "--sim", "--headless"], stdout=open(out / "simulator.log", "w"), stderr=subprocess.STDOUT)
        wait_http(f"http://{host}:{port}/api/daemon/status", 120, "the Reachy Mini simulator")
        return p
    if args.robot == "nao":
        port = cfg.get("nao", {}).get("port", 9600)
        if port_open("127.0.0.1", port):
            return None
        log("starting the mock NAO")
        p = subprocess.Popen([sys.executable, str(ROOT / "tools" / "mock_nao.py"), "--port", str(port)],
                             stdout=open(out / "simulator.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(3)
        return p
    if args.robot == "furhat":
        host = cfg.get("furhat", {}).get("host", "localhost")
        try:
            urllib.request.urlopen(f"http://{host}:54321/furhat/voices", timeout=5)
        except Exception:
            raise RuntimeError("the Furhat Remote API does not answer on port 54321: start the virtual Furhat in the "
                               "Furhat SDK (window not minimized) and launch the Remote API skill, then run again")
    return None


def start_recorders(args, cfg, out: Path, stop_file: Path):
    procs = []
    if args.robot == "reachy_mini":
        procs.append(subprocess.Popen([sys.executable, str(HERE / "robot_view.py"), "record", "--traj",
                                       str(out / "robot_traj.jsonl"), "--stamp", str(out / "robot_stamp.json"),
                                       "--stop-file", str(stop_file),
                                       "--host", cfg.get("reachy_mini", {}).get("host", "localhost")],
                                      stdout=open(out / "robot_view.log", "w"), stderr=subprocess.STDOUT))
    elif args.robot == "furhat" and os.name == "nt":
        procs.append(subprocess.Popen([sys.executable, str(HERE / "window_capture.py"), "Virtual Furhat",
                                       str(out / "robot.mp4"), "--stamp", str(out / "robot_stamp.json"),
                                       "--stop-file", str(stop_file)],
                                      stdout=open(out / "window_capture.log", "w"), stderr=subprocess.STDOUT))
        procs.append(subprocess.Popen([sys.executable, str(HERE / "loopback.py"), str(out / "loopback.wav"),
                                       str(out / "loopback_stamp.json"), str(stop_file)],
                                      stdout=open(out / "loopback.log", "w"), stderr=subprocess.STDOUT))
    else:
        log(f"no robot view for {args.robot} on this system: the video shows the console only")
        return procs
    end = time.time() + 60
    while not (out / "robot_stamp.json").exists():
        if time.time() > end or any(p.poll() is not None for p in procs):
            raise RuntimeError(f"robot recording did not start; see the logs in {out}")
        time.sleep(0.1)
    return procs


def robot_label(robot, cfg, caps):
    """Two caption lines under the robot view, from what the console reported about the robot."""
    cues = caps.get("expressions")
    if robot == "reachy_mini":
        motion = "speech-driven head and antenna motion; " if cues and cfg.get("reachy_mini", {}).get("animate", True) else ""
        voice = "neural voice (Kokoro-82M)" if "Kokoro" in caps.get("speech", "") else "system voice"
        return ["Robot: Reachy Mini (MuJoCo simulation)", motion + voice]
    if robot == "furhat":
        return ["Robot: Furhat (virtual Furhat)", "Furhat's own voice with lip sync" + ("; condition-matched gestures" if cues else "")]
    if robot == "nao":
        return ["Robot: NAO (mock robot)", "no robot view: the mock receives what the robot would say"]
    return ["No robot (text backend)", "replies are printed, not spoken"]


def summary(out: Path):
    db = sqlite3.connect(str(out / "session.db"))
    rows = []
    for (turn, att, pol, cat, sub, risk, disp, by, rev, judge, mon, det, reasons, text) in db.execute(
            "select turn_number, attempt, polar_level, category, subtype, risk_rating, disposition, decided_by, review_ms,"
            " fidelity_judge_json, monitor_json, fidelity_detector_json, blocked_reasons_json, llm_output "
            "from candidates order by created_at"):
        j, m, dt = (json.loads(x) if x else {} for x in (judge, mon, det))
        rows.append({"turn": turn, "attempt": att, "condition": f"{cat}{sub} at {pol:+d}", "risk": risk,
                     "judge_fidelity": j.get("fidelity"), "judge_category": j.get("matched_category"),
                     "detector_p_faithful": dt.get("p_faithful"),
                     "monitor_max": max((m.get("scores") or {}).values(), default=None),
                     "held_for": "; ".join(json.loads(reasons or "[]")), "decision": disp, "decided_by": by,
                     "review_s": round(rev / 1000, 1) if rev else None, "reply": text})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["turn"])
        w.writeheader()
        w.writerows(rows)
    log("\nturn  condition   risk    judge  monitor  decision    by        reply")
    for r in rows:
        log(f"{r['turn']:>2}.{r['attempt']:<2} {r['condition']:<11} {str(r['risk']):<7} {str(r['judge_fidelity']):>5}  "
            f"{str(r['monitor_max']):>7}  {str(r['decision']):<11} {str(r['decided_by']):<9} {(r['reply'] or '')[:60]}")
    log(f"\nper-candidate summary: {out / 'summary.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", required=True, choices=["reachy_mini", "furhat", "nao", "text"])
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True, help="new or empty folder for this rehearsal")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"), help="base configuration (default: config.yaml)")
    ap.add_argument("--port", type=int, default=8093, help="console port (the Reachy daemon uses 8000)")
    ap.add_argument("--video", action="store_true", help="record the console and the robot and compose a video")
    ap.add_argument("--no-expressions", action="store_true", help="turn off non-verbal cues (robot.expressions)")
    ap.add_argument("--no-sim", action="store_true", help="do not start a simulator (use a running robot or simulator)")
    ap.add_argument("--keep-sim", action="store_true", help="leave the simulator this script started running")
    args = ap.parse_args()

    sc = scn.load(args.scenario)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: choose a new --out (earlier rehearsals are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.scenario, out / "scenario.yaml")

    participant_audio(sc, out, args.video)
    cfg = make_config(args.config, sc, args, out)
    sim = console = None
    recorders, stop_file = [], out / "STOP"
    timeline = None
    try:
        if not args.no_sim:
            sim = start_simulator(args, cfg, out)
        log("starting the console")
        console = subprocess.Popen([sys.executable, str(ROOT / "main.py"), "--config", str(out / "config.yaml"),
                                    "--script", str(out / "participant_script.yaml")], cwd=str(ROOT),
                                   stdout=open(out / "console.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT)
        wait_http(f"http://127.0.0.1:{args.port}/api/status", 300, "the RAWR console")
        caps = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/status"))["capabilities"]
        log(f"robot: {caps['robot']}; speech: {caps['speech']}; Stop speech: {caps['interrupt']}")
        if args.video:
            recorders = start_recorders(args, cfg, out, stop_file)
        from operator_driver import OperatorDriver
        driver = OperatorDriver(f"http://127.0.0.1:{args.port}", str(out), record_video=args.video, log=log)
        try:
            timeline = driver.run(sc)
        finally:
            timeline = timeline or driver.T
            stop_file.touch()
            for p in recorders:
                try:
                    p.wait(90)
                except subprocess.TimeoutExpired:
                    kill_tree(p)
            timeline.update(robot=args.robot, db="session.db", capabilities=caps, robot_label=robot_label(args.robot, cfg, caps))
            if (out / "robot_stamp.json").exists():
                timeline["robot_video_start"] = json.load(open(out / "robot_stamp.json"))["robot_video_start"]
            json.dump(timeline, open(out / "timeline.json", "w"), indent=1)
    finally:
        kill_tree(console)
        if not args.keep_sim:
            kill_tree(sim)
    summary(out)

    if args.video:
        if args.robot == "reachy_mini":
            log("rendering the robot view (MuJoCo, offscreen; a few minutes)")
            subprocess.run([sys.executable, str(HERE / "robot_view.py"), "render", "--traj", str(out / "robot_traj.jsonl"),
                            "--out", str(out / "robot.mp4")], check=True)
        from compose import compose
        for cut in ("full", "short"):
            compose(out, sc, cut)


if __name__ == "__main__":
    main()
