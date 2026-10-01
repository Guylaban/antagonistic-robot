"""Scripted operator: drives the RAWR web console with Playwright, following a scenario's steps.

At each step marked with a moment it saves a screenshot together with the positions of the console
panels (for captions and highlights in the rehearsal video) and the reasons a pending reply is held.
"""
import json
import os
import time
import urllib.request

from scenario import ACTIONS

RECTS_JS = """() => {
  const card = s => { const e = document.querySelector(s); return e ? (e.closest('.card') || e) : null; };
  const els = {matrix: card('#polar'), policy: card('#modeTimed'), monitor_log: card('#log'), pending: document.querySelector('#pending'),
    safety: card('#sContent'), fidelity: card('#fJudge'), psych: card('#dims'), latency: card('#lat'), data: card('#csvBtn'),
    temper: document.querySelector('#temperBtn'), send: document.querySelector('#sendBtn'), apply: document.querySelector('#applyBtn'),
    discarded: [...document.querySelectorAll('#log .msg.discarded')].pop() || null,
    last_robot: [...document.querySelectorAll('#log .msg.r:not(.discarded)')].pop() || null};
  const out = {};
  for (const [k, e] of Object.entries(els)) { if (!e) continue; const r = e.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) out[k] = [r.x, r.y, r.width, r.height]; }
  return out; }"""


class OperatorDriver:
    def __init__(self, base_url: str, out_dir: str, record_video: bool = False, log=print):
        self.base, self.out, self.record_video, self.log = base_url.rstrip("/"), out_dir, record_video, log
        self.T = {"actions": [], "moments": [], "states": [], "participant_events": []}
        self.last_cid = None
        self.replies = 0

    # ---------------------------------------------------------------- console state
    def status(self) -> dict:
        return json.load(urllib.request.urlopen(self.base + "/api/status", timeout=5))

    def _wait(self, cond, timeout, what):
        end = time.time() + timeout
        while time.time() < end:
            v = cond()
            if v:
                return v
            time.sleep(0.15)
        raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")

    def wait_pending(self, timeout=180):
        def new():
            pid = self.status()["review"]["pending_candidate_id"]
            return pid if pid and pid != self.last_cid else None
        self.last_cid = self._wait(new, timeout, "the next reply in the review panel")
        return self.last_cid

    def wait_signals(self, cid, timeout=60) -> str:
        """'released' if the timer sent the reply, 'held' once it waits for the operator with its ratings in."""
        def done():
            r = self.status()["review"]
            if r["pending_candidate_id"] != cid:
                return "released"
            if not r["awaiting"] and not r["auto_release"]:
                return "held"
            return None
        return self._wait(done, timeout, "the reply's ratings")

    def wait_spoken(self, n, timeout=180):
        def done():
            s = self.status()
            return s["turn_count"] >= n and s["state"] == "listening" and not s["review"]["pending_candidate_id"]
        self._wait(done, timeout, f"reply {n} to be spoken")

    # ---------------------------------------------------------------- page actions
    def t(self):
        return time.time() - self.T["console_video_start"]

    def act(self, selector, label):
        self.T["actions"].append({"t": time.time(), "action": label})
        self.log(f"{self.t():6.1f}s  {label}")
        self.page.click(selector)

    def moment(self, key, reasons=None):
        t = time.time()
        shot = os.path.join(self.out, "moments", f"{key}.png")
        os.makedirs(os.path.dirname(shot), exist_ok=True)
        rects = self.page.evaluate(RECTS_JS)
        self.page.screenshot(path=shot)
        if self.page.evaluate(RECTS_JS) != rects:      # the layout moved during the screenshot: take both again
            rects = self.page.evaluate(RECTS_JS)
            self.page.screenshot(path=shot)
        self.T["moments"].append({"t": t, "key": key, "rects": rects, "screenshot": shot, "reasons": reasons or []})
        self.log(f"{t - self.T['console_video_start']:6.1f}s  moment {key}")

    def pause(self, seconds):
        self.page.wait_for_timeout(int(seconds * 1000))

    # ---------------------------------------------------------------- steps
    def step_matrix(self, m, apply=False, moment=None):
        p = self.page
        if "polar" in m:
            p.fill("#polar", str(m["polar"]))
            p.dispatch_event("#polar", "input")
        if "category" in m:
            p.click(f'[data-cat="{m["category"]}"]')
        if "subtype" in m:
            p.click(f'[data-sub="{m["subtype"]}"]')
        mods = m.get("modifiers") or {}
        if isinstance(mods, list):
            mods = {k: True for k in mods}
        for k, on in mods.items():
            (p.check if on else p.uncheck)(f'[data-mod="{k}"]')
        self.pause(0.6)
        if moment:
            self.moment(moment)
        if apply:
            desc = ", ".join(f"{k} {v}" for k, v in m.items())
            self.act("#applyBtn", f"Apply: {desc}")

    def step_start(self, s):
        self.page.fill("#pid", str(s.get("participant_id", "REHEARSAL")))
        self.act("#startBtn", "Start session")

    def step_reply(self, r, label, top=True):
        cid = self.wait_pending()
        if r.get("hold"):
            self.act("#holdBtn", f"{label}: Hold (stop the timer)")
        if r.get("on_arrival"):
            self.pause(0.5)
            self.moment(r["on_arrival"])
        state = self.wait_signals(cid)
        if state == "released":
            self.T["actions"].append({"t": time.time(), "action": f"{label}: released by the timer"})
            self.log(f"{self.t():6.1f}s  {label}: released by the timer")
        else:
            self.pause(0.6)
            reasons = self.status()["review"]["blocked_reasons"]
            if r.get("on_review"):
                self.moment(r["on_review"], reasons)
            self.pause(r.get("review_s", 2.5))
            action = r.get("action", "send")
            self.act(ACTIONS[action], f"{label}: {action.capitalize()} (held: {', '.join(reasons)})")
            if action != "send":
                done = {"temper": "tempered", "intensify": "intensified", "regenerate": "regenerated"}[action]
                self.step_reply(r.get("replacement") or {}, f"{label} {done}", top=False)
        if top:
            self.replies += 1
            if r.get("wait_spoken", True):
                self.wait_spoken(self.replies)

    def step_end(self, e):
        self.act("#endBtn", "End session")
        self.pause(1.2)
        self.page.evaluate("document.querySelector('#csvBtn').scrollIntoView({block: 'end', behavior: 'smooth'})")
        self.pause(0.8)
        if e.get("moment"):
            self.moment(e["moment"])
        self.pause(1.5)

    # ---------------------------------------------------------------- run
    def run(self, scenario: dict) -> dict:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception:
                browser = p.chromium.launch(channel="chrome", headless=True)    # installed Chrome instead
            opts = {"viewport": {"width": 1600, "height": 900}, "device_scale_factor": 1}
            if self.record_video:
                opts.update(record_video_dir=os.path.join(self.out, "console_video"),
                            record_video_size={"width": 1600, "height": 900})
            ctx = browser.new_context(**opts)
            self.page = ctx.new_page()
            self.T["console_video_start"] = time.time()

            def on_ws(ws):
                def frame(payload):
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        return
                    if ev.get("type") == "participant":
                        self.T["participant_events"].append({"t": time.time(), "turn": ev.get("turn_number"),
                                                             "text": ev.get("transcript")})
                    elif ev.get("type") == "state_change":
                        self.T["states"].append({"t": time.time(), "state": ev.get("state")})
                ws.on("framereceived", frame)
            self.page.on("websocket", on_ws)
            self.page.on("dialog", lambda d: d.accept())
            self.page.goto(self.base)
            self.pause(2)
            try:
                for i, step in enumerate(scenario["steps"]):
                    if "matrix" in step:
                        self.step_matrix(step["matrix"], step.get("apply", False), step.get("moment"))
                    elif "start" in step:
                        self.step_start(step["start"] or {})
                    elif "reply" in step:
                        self.step_reply(step["reply"] or {}, f"T{self.replies + 1}")
                    elif "end" in step:
                        self.step_end(step["end"] or {})
                    elif "wait" in step:
                        self.pause(float(step["wait"]))
            finally:
                self.T["console_video_end"] = time.time()
                video = self.page.video.path() if self.record_video else None
                ctx.close()
                browser.close()
                self.T["console_video_file"] = video
        return self.T
