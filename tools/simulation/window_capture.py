"""Record one window (e.g. the Virtual Furhat) at a constant 25 fps on the wall clock. Windows 10 2004+ only.

Uses Windows Graphics Capture (pip install windows-capture), which receives every frame the window
presents even when other windows cover it (it must not be minimized). PrintWindow-style grabs were
too slow here (35-100 ms each, 10-14 fps), which made the Furhat's lip movements look jerky. The
title bar and borders are cropped off. Run nothing else on the GPU while recording: the virtual
Furhat renders at 30 fps only when the GPU is free.

Usage: python window_capture.py "Virtual Furhat" out.mp4 --stamp stamp.json --stop-file STOP [--height 900]
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np


def client_box(title, frame_w, frame_h):
    """Client area (no title bar or borders) inside the captured window image."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    ext = wt.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(ext), ctypes.sizeof(ext))   # extended frame bounds
    cr, pt = wt.RECT(), wt.POINT(0, 0)
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    sx, sy = frame_w / max(1, ext.right - ext.left), frame_h / max(1, ext.bottom - ext.top)
    return int((pt.x - ext.left) * sx), int((pt.y - ext.top) * sy), int(cr.right * sx), int(cr.bottom * sy)


def main():
    from windows_capture import Frame, InternalCaptureControl, WindowsCapture
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    title, out = sys.argv[1], sys.argv[2]
    args = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    fps, height = 25, int(args.get("--height", 900))
    latest = {"f": None, "n": 0}
    lock = threading.Lock()
    stop = threading.Event()
    cap = WindowsCapture(cursor_capture=False, draw_border=False, window_name=title)

    @cap.event
    def on_frame_arrived(frame: Frame, control: InternalCaptureControl):
        with lock:
            latest["f"] = np.ascontiguousarray(frame.frame_buffer[:, :, 2::-1])     # BGRA -> RGB
            latest["n"] += 1
        if stop.is_set():
            control.stop()

    @cap.event
    def on_closed():
        stop.set()

    cap.start_free_threaded()
    end = time.time() + 15
    while latest["f"] is None:
        if time.time() > end:
            sys.exit(f"no frames from window {title!r} (is it open and not minimized?)")
        time.sleep(0.01)
    cx, cy, w0, h0 = client_box(title, latest["f"].shape[1], latest["f"].shape[0])
    w0 = min(w0, latest["f"].shape[1] - cx) // 2 * 2
    h0 = min(h0, latest["f"].shape[0] - cy) // 2 * 2
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w0}x{h0}", "-r", str(fps), "-i", "-", "-vf", f"scale=-2:{height}",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out],
                            stdin=subprocess.PIPE)
    t0 = time.time()
    json.dump({"robot_video_start": t0}, open(args["--stamp"], "w"))
    n = 0
    while not os.path.exists(args["--stop-file"]):
        with lock:
            f = latest["f"][cy:cy + h0, cx:cx + w0]
        if f.shape[0] < h0 or f.shape[1] < w0:                 # window resized smaller: pad
            f = np.pad(f, ((0, h0 - f.shape[0]), (0, w0 - f.shape[1]), (0, 0)))
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
        n += 1
        time.sleep(max(0.0, t0 + n / fps - time.time()))
    stop.set()
    proc.stdin.close()
    proc.wait()
    print(f"recorded {n} frames ({n / fps:.1f} s); the window presented {latest['n'] / (n / fps):.1f} frames/s")


if __name__ == "__main__":
    main()
