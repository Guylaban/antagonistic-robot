"""Compose a rehearsal video: console (left) + robot (right) + caption panel (bottom), with explanatory pauses.

Usage: python compose.py RUN_DIR --scenario SCENARIO.yaml [--cut full|short|none] [--out video.mp4]

RUN_DIR is the output of rehearse.py --video. At each moment that has a caption with a pause, the
video freezes (both views and the audio) on the screenshot taken at that moment, the console is
dimmed except for the panels the caption describes (outlined), and the caption explains what
happened. A pause is moved to the next silence if someone is speaking, so no speech is cut; it then
freezes the live frame. Between pauses the caption of the latest step stays up.

--cut short keeps only the pauses of captions marked essential and drops captions marked
short: false (for videos with a length limit); --cut none plays the session in real time.
"""
import argparse
import json
import re
import sqlite3
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario as scn  # noqa: E402

FPS, SR = 25, 48000
W, H = 1920, 1080
CON = (24, 104, 1280, 720)          # x, y, w, h (the 1600x900 console page scaled by 0.8)
ROB = (1328, 104, 568, 710)
PAN = (24, 880, 1872, 176)          # caption panel
# The virtual Furhat's lips trail its own voice: +160 ms in two still-head tests (correlation of mouth
# opening with loudness, flat between 0 and +160 ms). Showing the face 120 ms earlier leaves about 40 ms.
FURHAT_LIP_LEAD = 0.12
ROBOT_LABELS = {
    "reachy_mini": ("Robot: Reachy Mini (MuJoCo simulation)", "speech-driven head and antenna motion; robot voice from RAWR"),
    "furhat": ("Robot: Furhat (virtual Furhat)", "Furhat's own voice with lip sync; condition-matched gestures"),
    "nao": ("Robot: NAO (mock robot)", "no robot view: the mock prints what the robot would say"),
    "text": ("No robot (text backend)", "replies are printed, not spoken"),
}


def font(bold, size):
    for name in (["segoeuib", "arialbd", "DejaVuSans-Bold"] if bold else ["segoeui", "arial", "DejaVuSans"]):
        for path in (f"C:/Windows/Fonts/{name}.ttf", name + ".ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


def wrap(text, f, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        test = (cur + " " + w_).strip()
        if f.getlength(test) <= width:
            cur = test
        else:
            lines.append(cur)
            cur = w_
    return lines + [cur]


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), "<i2").astype(np.float32) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr)), np.arange(len(x)), x).astype(np.float32)
    return x


def blend(frame, layer, x, y):
    h, w = layer.shape[:2]
    region = frame[y:y + h, x:x + w].astype(np.float32)
    a = layer[:, :, 3:4]
    frame[y:y + h, x:x + w] = (region * (1 - a) + layer[:, :, :3] * 255 * a).astype(np.uint8)


def reader(cmd, w, h):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    last = np.zeros((h, w, 3), np.uint8)
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        last = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        yield last
    while True:
        yield last


def console_lag(T, t0):
    """Seconds by which the console video starts after console_video_start (Playwright's first frame comes late).

    Measured from the review card: each Send/Temper click at a known wall time makes it vanish in the video.
    """
    pend = [m["rects"]["pending"] for m in T["moments"] if "pending" in m["rects"]]
    if not pend:
        return 0.0
    w, h = 400, 225
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", T["console_video_file"], "-vf", f"fps={FPS},scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    v = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)
    x, y, cw, ch = [int(c) // 4 for c in pend[0]]
    reg = v[:, y + 2:y + ch - 2, x + 2:x + cw - 2].astype(np.int16)
    vis = ((reg[..., 0] >= 246) & (reg[..., 1] >= 246) & (reg[..., 2] >= 220) & (reg[..., 2] < 241)).mean(axis=(1, 2)) > 0.25
    vanish = (np.flatnonzero(np.diff(vis.astype(int)) == -1) + 1) / FPS
    lags = []
    for a in T["actions"]:
        if ": Send" in a["action"] or ": Temper" in a["action"]:
            tb = a["t"] - t0
            near = vanish[(vanish > tb - 4) & (vanish < tb + 2)]
            if len(near):
                lags.append(tb - near[np.argmin(np.abs(near - tb))])
    return float(np.median(lags)) if lags else 0.0


def compose(run: Path, sc: dict, cut: str = "full", out: Path = None) -> Path:
    T = json.load(open(run / "timeline.json"))
    robot = {"reachy": "reachy_mini"}.get(T["robot"], T["robot"])
    out = out or run / f"rehearsal_{cut}.mp4"
    t0 = T["console_video_start"]
    dur = T["console_video_end"] - t0
    captions_def = sc.get("captions") or {}
    f_title, f_note, f_lab = font(True, 36), font(False, 19), font(True, 24)
    f_step, f_body, f_badge = font(True, 30), font(False, 25), font(True, 20)

    # ------------------------------------------------------------ audio on the session timeline (48 kHz mono)
    mix = np.zeros(int(dur * SR) + SR, np.float32)
    speech = []

    def place(x, start, gain=1.0):
        i = int(round(start * SR))
        if i < 0:
            x, i = x[-i:], 0
        x = x[:max(0, len(mix) - i)]
        mix[i:i + len(x)] += gain * x
        speech.append((start, start + len(x) / SR))

    db_path = run / T["db"] if T.get("db") else next(iter(sorted(run.glob("*.db")) + sorted(run.glob("data/*.db"))))
    db = sqlite3.connect(str(db_path))
    sid = db.execute("select session_id from sessions order by start_time desc limit 1").fetchone()[0]
    ts = lambda s: datetime.fromisoformat(s).timestamp()
    log = run / "robot_speech" / "speech_log.jsonl"
    if log.exists():                                 # Reachy Mini: the robot's audio exactly as played
        for l in open(log, encoding="utf-8"):
            r = json.loads(l)
            place(read_wav(log.parent / r["file"]), r["wall_start"] - t0)
    elif (run / "loopback.wav").exists():            # virtual Furhat: speaker loopback, only while it spoke
        st = json.load(open(run / "loopback_stamp.json"))
        lb = read_wav(run / "loopback.wav")
        sent = dict(db.execute("select turn_number, decided_at from candidates where session_id=? and "
                               "disposition in ('sent','auto_sent')", (sid,)))
        for turn, done in db.execute("select turn_number, timestamp from turns where session_id=?", (sid,)):
            if turn not in sent:
                continue
            a, b = ts(sent[turn]) - 1.0, ts(done) + 0.5
            seg = lb[max(0, int((a - st["audio_start"]) * SR)):max(0, int((b - st["audio_start"]) * SR))].copy()
            ramp = min(len(seg) // 2, int(0.05 * SR))
            if ramp:
                seg[:ramp] *= np.linspace(0, 1, ramp)
                seg[-ramp:] *= np.linspace(1, 0, ramp)
            place(seg, a - t0)
    voiced = (run / "participant_audio.json").exists()
    if voiced:                                       # each participant clip ends when RAWR received the line
        meta = {m["text"]: m for m in json.load(open(run / "participant_audio.json"))}
        for created, llm_ms, text in db.execute("select created_at, latency_llm_ms, user_transcript from candidates "
                                                "where session_id=? and attempt=1 order by turn_number", (sid,)):
            m = meta.get(text)
            if m:
                delivered = ts(created) - (llm_ms or 0) / 1000
                place(read_wav(run / m["file"]), delivered - t0 - m["seconds"] - 0.2, 0.9)
    speech.sort()

    # ------------------------------------------------------------ captions and pauses
    def reasons_of(m):
        if m.get("reasons"):
            return m["reasons"]
        nxt = [a["action"] for a in T["actions"] if a["t"] >= m["t"] and "(held:" in a["action"]][:1]
        return [nxt[0].split("(held:", 1)[1].rstrip(")")] if nxt else []

    events = []                                      # (time, key, caption, moment)
    intro = scn.caption_for(captions_def, "intro")
    if intro:
        events.append((0.0, "intro", intro, None))
    for m in T["moments"]:
        cap = scn.caption_for(captions_def, m["key"], reasons_of(m))
        if cap is None or (cut == "short" and captions_def[m["key"]].get("short") is False
                           and not cap["essential"]):
            continue
        events.append((m["t"] - t0, m["key"], cap, m))
    events.sort(key=lambda e: e[0])
    numbers = scn.number_captions([e[1] for e in events])
    for e in events:
        if e[1] in numbers:
            e[2]["title"] = f"{numbers[e[1]]} · {e[2]['title']}"
    pauses = []
    for t, key, cap, m in events:
        secs = cap["pause"] if cut == "full" or (cut == "short" and cap["essential"]) else 0
        if secs and m:
            tp = np.ceil(scn.defer_to_silence(t, speech) * FPS) / FPS     # on the frame grid
            pauses.append((tp, secs, cap, m, bool(m.get("screenshot")) and tp - t < 0.6))
            print(f"pause {key:16s} at {t:6.1f} s -> {tp:6.1f} s ({secs} s)")

    def panel(cap, paused):
        im = Image.new("RGBA", PAN[2:], (0, 0, 0, 0))
        g = ImageDraw.Draw(im)
        g.rounded_rectangle((0, 0, PAN[2] - 1, PAN[3] - 1), 14, fill=(30, 41, 59, 255),
                            outline=(245, 158, 11, 255) if paused else (51, 65, 85, 255), width=3)
        g.text((26, 16), cap["title"], font=f_step, fill=(255, 255, 255))
        for i, line in enumerate(wrap(cap["body"], f_body, PAN[2] - 60)[:3]):
            g.text((26, 62 + i * 34), line, font=f_body, fill=(226, 232, 240))
        if paused:
            txt = "PAUSED to explain"
            tw = f_badge.getlength(txt)
            g.rounded_rectangle((PAN[2] - tw - 64, 16, PAN[2] - 22, 50), 10, fill=(245, 158, 11, 255))
            g.text((PAN[2] - tw - 43, 19), txt, font=f_badge, fill=(17, 24, 39))
        return np.asarray(im).astype(np.float32) / 255

    def spotlight(rects, keys):
        im = Image.new("RGBA", CON[2:], (8, 12, 20, 125))
        g = ImageDraw.Draw(im)
        boxes = [b for b in (scn.visible_box(rects[k]) for k in keys if k in rects) if b]
        boxes = [[v * 0.8 for v in b] for b in boxes]
        for x, y, w, h in boxes:
            g.rectangle((x - 4, y - 4, x + w + 4, y + h + 4), fill=(0, 0, 0, 0))
        for x, y, w, h in boxes:
            g.rounded_rectangle((x - 5, y - 5, x + w + 5, y + h + 5), 8, outline=(245, 158, 11, 255), width=4)
        return np.asarray(im).astype(np.float32) / 255

    # ------------------------------------------------------------ static background
    bg = Image.new("RGB", (W, H), (17, 24, 39))
    d = ImageDraw.Draw(bg)
    d.text((24, 18), sc.get("title", "RAWR: operator-controlled antagonistic robot behavior"), font=f_title,
           fill=(255, 255, 255))
    note = ("Rehearsal in simulation. Scripted participant: its lines reach RAWR as text"
            + ("; the voice was added for the viewer." if voiced else ".")
            + (" Real time except the marked pauses." if pauses else " Real time."))
    d.text((24, 64), note, font=f_note, fill=(148, 163, 184))
    label = T.get("robot_label") or ROBOT_LABELS.get(robot, (f"Robot: {robot}", ""))
    d.text((CON[0], CON[1] + CON[3] + 8), "Operator console", font=f_lab, fill=(255, 255, 255))
    d.text((ROB[0], ROB[1] + ROB[3] + 8), label[0], font=f_lab, fill=(255, 255, 255))
    d.text((ROB[0], ROB[1] + ROB[3] + 40), label[1], font=f_note, fill=(203, 213, 225))
    has_view = (run / "robot.mp4").exists()
    if not has_view:
        d.rounded_rectangle((ROB[0], ROB[1], ROB[0] + ROB[2], ROB[1] + ROB[3]), 14, fill=(30, 41, 59))
        for i, line in enumerate(wrap("No robot view was recorded for this backend.", f_body, ROB[2] - 60)):
            d.text((ROB[0] + 30, ROB[1] + ROB[3] // 2 - 20 + i * 34), line, font=f_body, fill=(148, 163, 184))
    BG = np.asarray(bg).copy()

    # ------------------------------------------------------------ frames
    lag = console_lag(T, t0)
    print(f"console video starts {lag:.2f} s after the recorded start; delayed by that much")
    vf = f"fps={FPS},scale={CON[2]}:{CON[3]}" + (f",tpad=start_duration={lag:.3f}:start_mode=clone" if lag > 0 else "")
    con_frames = reader(["ffmpeg"] + (["-ss", f"{-lag:.3f}"] if lag < 0 else []) + ["-i", T["console_video_file"],
                        "-vf", vf if lag >= 0 else f"fps={FPS},scale={CON[2]}:{CON[3]}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                        CON[2], CON[3])
    if has_view:
        offset = t0 - T["robot_video_start"] + (FURHAT_LIP_LEAD if robot == "furhat" else 0)
        rob_frames = reader(["ffmpeg", "-ss", f"{max(0.0, offset):.3f}", "-i", str(run / "robot.mp4"), "-vf",
                             f"fps={FPS},scale=-2:{ROB[3]},crop={ROB[2]}:{ROB[3]}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                            ROB[2], ROB[3])
    video_only = run / "video_only.mp4"
    enc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(video_only)], stdin=subprocess.PIPE)
    panels = {}

    def get_panel(cap, paused):
        k = (cap["title"], paused)
        if k not in panels:
            panels[k] = panel(cap, paused)
        return panels[k]

    n = int(dur * FPS)
    pi, a_pos, audio = 0, 0, []
    for k in range(n):
        tb = k / FPS
        frame = BG.copy()
        frame[CON[1]:CON[1] + CON[3], CON[0]:CON[0] + CON[2]] = next(con_frames)
        if has_view:
            frame[ROB[1]:ROB[1] + ROB[3], ROB[0]:ROB[0] + ROB[2]] = next(rob_frames)
        current = [e[2] for e in events if e[0] <= tb + 1e-6]
        while pi < len(pauses) and pauses[pi][0] <= tb:
            tp, secs, cap, m, use_shot = pauses[pi]
            i = int(round(tp * SR))
            audio += [mix[a_pos:i], np.zeros(int(secs * SR), np.float32)]
            a_pos = i
            frozen = frame.copy()
            if use_shot:                      # the screenshot taken with the panel boxes: they match exactly
                path = Path(m["screenshot"])
                path = path if path.exists() else run / "moments" / path.name     # run folder moved
                shot = Image.open(path).convert("RGB").resize(CON[2:], Image.LANCZOS)
                frozen[CON[1]:CON[1] + CON[3], CON[0]:CON[0] + CON[2]] = np.asarray(shot)
            blend(frozen, spotlight(m.get("rects", {}), cap["highlight"]), CON[0], CON[1])
            blend(frozen, get_panel(cap, True), PAN[0], PAN[1])
            for _ in range(int(secs * FPS)):
                enc.stdin.write(frozen.tobytes())
            pi += 1
        if current:
            blend(frame, get_panel(current[-1], False), PAN[0], PAN[1])
        enc.stdin.write(frame.tobytes())
    audio.append(mix[a_pos:int(n / FPS * SR)])
    enc.stdin.close()
    enc.wait()

    a = np.concatenate(audio)
    peak = np.abs(a).max() if len(a) else 0
    if peak > 0.98:
        a *= 0.98 / peak
    with wave.open(str(run / "video_audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((a * 32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_only), "-i",
                    str(run / "video_audio.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    total = n / FPS + sum(p[1] for p in pauses)
    print(f"wrote {out}: {total:.1f} s ({len(pauses)} pauses)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--cut", choices=["full", "short", "none"], default="full")
    ap.add_argument("--out")
    a = ap.parse_args()
    compose(Path(a.run_dir), scn.load(a.scenario), a.cut, Path(a.out) if a.out else None)
