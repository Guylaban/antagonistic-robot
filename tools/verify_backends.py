"""Verify a robot setup: the gate's release rules, failed checks, Stop speech, and End session.

Runs the real CRAB pipeline (prompt compiler, review gate, logging) on the robot backend you choose,
with scripted participant lines and six cases:

    1. a clear reply             -> released by the timer after the hold window
    2. a reply the monitor flags -> held until the operator sends it
    3. the monitor fails         -> held ("monitor unavailable")
    4. the judge fails           -> held ("fidelity judge unavailable")
    5. a long reply              -> Stop speech mid-utterance returns control to CRAB
    6. End session while a reply waits -> the reply is withheld and the session closes

By default the replies and the monitor and judge results are scripted, so each case is controlled
and no API key is needed. With --live, the replies come from the LLM in config.yaml and are scored
by the real psychosocial monitor and fidelity judge (API keys in the environment or .env). The two
failure cases then point that one check at an address that refuses connections, so the failure is
a real failed call. The condition is set per case so that the case can occur: neutral (0) for the
clear reply and Stop speech, Confrontational +2 for the monitor flag (up to --attempts replies),
Confrontational +1 for the judge failure, and Aggressive +3 for End session. A further check,
gate_follows_scores, recomputes from the logged ratings and scores whether each reply should have
been held, and compares that with what the gate did.

    python tools/verify_backends.py --robot text
    python tools/verify_backends.py --robot nao --nao-ip 127.0.0.1      # with tools/mock_nao.py running
    python tools/verify_backends.py --robot furhat                      # Remote API skill running
    python tools/verify_backends.py --robot reachy_mini                 # daemon (or --sim) running
    python tools/verify_backends.py --robot text --live                 # real LLM, monitor and judge

Writes to --out (new folder): session.db (everything CRAB logged, including every prompt and raw
reply), replies.csv (one row per generated reply), checks.jsonl and checks.csv (one line per
check: expected, observed, pass) and summary.json. Uses your config.yaml for the robot settings.
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antagonist_robot.config.settings import AvctConfig, FidelityConfig, OperatorConfig, load_config  # noqa: E402
from antagonist_robot.conversation.avct_manager import AvctManager  # noqa: E402
from antagonist_robot.conversation.fidelity import FidelityMonitor  # noqa: E402
from antagonist_robot.conversation.manager import ConversationManager  # noqa: E402
from antagonist_robot.conversation.monitor import MonitorResult, PsychosocialMonitor  # noqa: E402
from antagonist_robot.conversation.operator import OperatorGate  # noqa: E402
from antagonist_robot.conversation.safety import RISK_LEVELS  # noqa: E402
from antagonist_robot.logging.session_logger import SessionLogger  # noqa: E402
from antagonist_robot.pipeline.scripted_input import ScriptedParticipant  # noqa: E402
from antagonist_robot.pipeline.types import LLMResult  # noqa: E402
from antagonist_robot.robots import create_backend  # noqa: E402

DEAD_URL = "http://127.0.0.1:9/v1"     # nothing listens here: a real call that fails at once
JUDGE_BELOW = 4
BLOCK_AT = "Orange"
CLEAR = {"privacy": 0, "discrimination": 0, "manipulation": 0, "psych_harm": 0, "insulting": 0}
FLAG = {**CLEAR, "insulting": 2}
FAITHFUL = {"matched_category": "D", "fidelity": 8, "intensity_est": 2, "refused": False, "rationale": "scripted"}
LONG = ("You keep saying that splitting the work equally is fair, but fairness is not the same as equality, "
        "and you have not shown me a single reason why it would work for this team. Some people are faster, "
        "some have more time, and some have skills the others do not. If you give everyone the same share, "
        "the project will move at the pace of the slowest person, and the strongest people will do the rest "
        "anyway. So tell me again, slowly, why you think this is a good plan.")
# per case: participant line, scripted reply, scripted monitor and judge results (None = an error),
# and the condition (polar level, category, subtype) used with --live
CASES = {
    "release": ("I think we should split the work equally.", "That is a naive idea. Prove it works.",
                CLEAR, FAITHFUL, (0, "D", 1)),
    "monitor_hold": ("It seems fair to everyone.", "Fair? That is lazy thinking.", FLAG, FAITHFUL, (2, "D", 3)),
    "monitor_failure": ("People respond better when things feel equal.", "Show me the evidence.", None, FAITHFUL,
                        (0, "D", 1)),
    "judge_failure": ("So what would you do instead?", "First admit the plan is weak.", CLEAR, None, (1, "D", 1)),
    "stop_speech": ("Okay, explain it then. Walk me through your whole plan in detail, step by step.", LONG, CLEAR,
                    FAITHFUL, (0, "D", 1)),
    "end_session": ("Thanks, I have enough.", "You have nothing, but go on.", FLAG, FAITHFUL, (3, "F", 3)),
}
MONITOR_RETRIES = ["Everyone gets the same share, so nobody can complain.",
                   "I still think equal is the fairest way, and you are overcomplicating it."]


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, system_prompt, messages):
        return LLMResult(text=self.replies.pop(0) if self.replies else "Fine.", model="scripted",
                         total_tokens=0, generation_time_seconds=0.0)


class ScriptedMonitor:
    enabled = True
    gate_auto_send = True

    def __init__(self, results):
        self.results = list(results)

    def score_async(self, user_text, response_text, callback):
        result = self.results.pop(0) if self.results else CLEAR

        def run():
            time.sleep(0.3)
            callback(MonitorResult(scores={} if result is None else result,
                                   error="scripted failure" if result is None else None, model="scripted"))
        threading.Thread(target=run, daemon=True).start()


class ScriptedJudge:
    """Stands in for the OpenAI client: one scripted judgment (or an error) per call."""

    def __init__(self, results):
        self.results = list(results)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        result = self.results.pop(0) if self.results else FAITHFUL
        time.sleep(0.3)
        if result is None:
            raise RuntimeError("scripted failure")
        msg = SimpleNamespace(content=json.dumps(result))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="scripted")


def wait_for(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    return None


def should_hold(c):
    """From the logged rating and scores alone: should the gate have held this reply?"""
    if RISK_LEVELS.index(c.get("risk_rating") or "Green") >= RISK_LEVELS.index(BLOCK_AT):
        return True
    monitor = json.loads(c.get("monitor_json") or "{}") or {}
    if monitor.get("error") or max((monitor.get("scores") or {}).values(), default=0) >= 2:
        return True
    judge = json.loads(c.get("fidelity_judge_json") or "{}") or {}
    if FidelityMonitor.applies(c["polar_level"], c["category"]) and judge:
        return bool(judge.get("error") or judge.get("refused") or judge.get("fidelity", 10) < JUDGE_BELOW)
    return False


def build_live(cfg):
    """The real LLM, monitor and judge from config.yaml."""
    from antagonist_robot.pipeline.llm import LLMEngine
    missing = [k for k in (cfg.llm.api_key_env, cfg.monitor.api_key_env, cfg.fidelity.judge_api_key_env)
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"--live needs {', '.join(sorted(set(missing)))} in the environment or .env")
    cfg.monitor.enabled = True
    cfg.monitor.gate_auto_send = True
    cfg.monitor.api_key = os.environ[cfg.monitor.api_key_env]
    fid = FidelityConfig(**{**cfg.fidelity.__dict__, "judge_enabled": True, "detector_enabled": False,
                            "block_auto_send_below": JUDGE_BELOW})
    fid.api_key = os.environ[fid.judge_api_key_env]
    dead_monitor_cfg = SimpleNamespace(**{**cfg.monitor.__dict__, "base_url": DEAD_URL})
    dead_judge = FidelityConfig(**{**fid.__dict__, "judge_base_url": DEAD_URL})
    dead_judge.api_key = fid.api_key
    return (LLMEngine(cfg.llm), PsychosocialMonitor(cfg.monitor), FidelityMonitor(fid),
            PsychosocialMonitor(dead_monitor_cfg), FidelityMonitor(dead_judge)._judge)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", required=True, choices=["text", "nao", "furhat", "reachy_mini"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--nao-ip")
    ap.add_argument("--live", action="store_true", help="real LLM, monitor and judge (needs API keys)")
    ap.add_argument("--attempts", type=int, default=3, help="--live: replies to try for the monitor flag")
    ap.add_argument("--hold", type=float, default=1.0, help="hold window (s) for the timed review")
    ap.add_argument("--out")
    args = ap.parse_args()
    mode = "live" if args.live else "scripted"
    out = Path(args.out or ROOT / "runs" / f"verify_{args.robot}_{mode}_{datetime.now():%Y%m%d_%H%M%S}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already has files; choose a new --out")

    import yaml
    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    else:
        # replies and checks are scripted, so no API key is needed; satisfy load_config's key check
        raw = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
        for section in raw.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if k.endswith("api_key_env") and v:
                        os.environ.setdefault(v, "not-used-by-verify-backends")
    cfg = load_config(args.config)
    cfg.robot.backend = args.robot
    if args.nao_ip:
        cfg.nao.ip = args.nao_ip
    if args.live:
        llm, monitor, fidelity, dead_monitor, dead_judge = build_live(cfg)
    else:
        llm = ScriptedLLM([c[1] for c in CASES.values()])
        monitor = ScriptedMonitor([c[2] for c in CASES.values()])
        fidelity = FidelityMonitor(FidelityConfig(detector_enabled=False, judge_enabled=True,
                                                  block_auto_send_below=JUDGE_BELOW),
                                   judge_client=ScriptedJudge([c[3] for c in CASES.values()]))
    out.mkdir(parents=True, exist_ok=True)
    robot = create_backend(cfg)
    robot.connect()
    logger = SessionLogger(str(out / "session.db"), str(out / "audio"), save_audio=False)
    participant = ScriptedParticipant([], delay_s=0.5)
    gate = OperatorGate(OperatorConfig(review_mode="timed", hold_seconds=args.hold, block_auto_send_at=BLOCK_AT))
    manager = ConversationManager(audio_capture=participant, asr=participant, llm=llm, robot=robot,
                                  avct_manager=AvctManager(AvctConfig()), session_logger=logger, gate=gate,
                                  monitor=monitor, fidelity=fidelity)
    events = []
    manager.on_event = lambda e: events.append({**e, "_t": time.monotonic()})
    sid = manager.start_session(2, "D", 2, [], "VERIFY")
    checks = []

    def check(name, expected, observed, ok, **extra):
        rec = {"robot": args.robot, "mode": mode, "check": name, "expected": expected, "observed": observed,
               "pass": bool(ok), **extra}
        checks.append(rec)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {observed}")

    def turn(case, line=None):
        """Queue the participant's line, set the case's condition (--live), and run one turn in a thread."""
        participant._utterances.append(line or CASES[case][0])
        participant._delays.append(0.5)
        if args.live:
            manager.set_avct(*CASES[case][4], [], source="verify_backends")
        box = {}
        th = threading.Thread(target=lambda: box.setdefault("t", manager.run_turn()), daemon=True)
        th.start()
        return th, box

    def candidates():
        return [e for e in events if e["type"] == "candidate"]

    def new_candidate(n_before):
        """The first candidate event after the n_before already seen."""
        return wait_for(lambda: candidates()[n_before:n_before + 1] or None, 60)[0]

    def blocked_reasons(cid):
        """Why the reply is held: the reasons it was opened with, then any later flags."""
        first = next((e.get("blocked_reasons") or [] for e in events
                      if e["type"] == "candidate" and e["candidate_id"] == cid), [])
        later = [e["reason"] for e in events if e["type"] == "candidate_blocked" and e["candidate_id"] == cid]
        return first + [r for r in later if r not in first]

    def scored(cid):
        """The monitor event for this candidate (None until it has scored)."""
        return next((e for e in events if e["type"] == "monitor" and e["candidate_id"] == cid), None)

    def spoken(cid):
        return any(e["type"] == "speaking" and e.get("candidate_id") == cid for e in events)

    def held_then_sent(name, case, expect_reason, line=None):
        """Run a turn that should be held; send it after 3 hold windows. Returns (ok, observed, reasons)."""
        n = len(candidates())
        th, box = turn(case, line)
        c = new_candidate(n)
        wait_for(lambda: scored(c["candidate_id"]), 30)
        reason = wait_for(lambda: [r for r in blocked_reasons(c["candidate_id"]) if expect_reason in r], 30)
        time.sleep(3 * args.hold)
        still_waiting = th.is_alive() and not spoken(c["candidate_id"])
        manager.operator_action(c["candidate_id"], "send")
        th.join(90)
        r = box.get("t")
        reasons = blocked_reasons(c["candidate_id"])
        observed = (f"held for {'; '.join(reasons) or 'nothing'}; still waiting after {3 * args.hold:.1f} s: "
                    f"{still_waiting}; then {r.operator_action if r else None} by {r.decided_by if r else None}")
        ok = bool(reason) and still_waiting and r is not None and r.operator_action == "sent" \
            and r.decided_by == "operator"
        return ok, observed, reasons

    def let_resolve(th, cid):
        """A reply that was not expected to be held: wait for the timer, or send it if it was held."""
        th.join(3 * args.hold + 30)
        if th.is_alive():
            manager.operator_action(cid, "send")
            th.join(90)

    # 1. clear reply: released by the timer
    n = len(candidates())
    th, box = turn("release")
    c = new_candidate(n)
    let_resolve(th, c["candidate_id"])
    r = box.get("t")
    m = scored(c["candidate_id"]) or {}
    wait_ms = r.latency.get("review_ms") if r else None
    check("release_clear_reply", f"auto-released by the timer after about {args.hold:.1f} s",
          f"{r.operator_action if r else None} by {r.decided_by if r else None} after {wait_ms} ms"
          + (f"; monitor scores {m.get('scores')}" if args.live else ""),
          r is not None and r.operator_action == "auto_sent" and r.decided_by == "timer", review_ms=wait_ms,
          reply=c.get("response"))

    # 2. a reply the monitor flags: held until the operator sends it
    if args.live:
        tries = []
        for attempt, line in enumerate([None] + MONITOR_RETRIES[:args.attempts - 1], start=1):
            n = len(candidates())
            th, box = turn("monitor_hold", line)
            c = new_candidate(n)
            m = wait_for(lambda: scored(c["candidate_id"]), 30) or {}
            flagged = max((m.get("scores") or {}).values(), default=0) >= 2
            tries.append({"attempt": attempt, "reply": c.get("response"), "scores": m.get("scores")})
            if flagged:
                break
            let_resolve(th, c["candidate_id"])
        if flagged:
            time.sleep(3 * args.hold)
            still_waiting = th.is_alive() and not spoken(c["candidate_id"])
            manager.operator_action(c["candidate_id"], "send")
            th.join(90)
            r = box.get("t")
            reasons = blocked_reasons(c["candidate_id"])
            ok = "monitor: clear psychosocial risk" in reasons and still_waiting and r is not None \
                and r.operator_action == "sent" and r.decided_by == "operator"
            observed = (f"monitor scored {m.get('scores')} on attempt {len(tries)}; held for {'; '.join(reasons)}; "
                        f"still waiting after {3 * args.hold:.1f} s: {still_waiting}; then "
                        f"{r.operator_action if r else None} by {r.decided_by if r else None}")
        else:
            ok, observed = False, f"the monitor flagged none of {len(tries)} replies (scores {[t['scores'] for t in tries]})"
        check("hold_monitor_flag", "held (monitor: clear psychosocial risk) until the operator sent it", observed, ok,
              attempts=tries)
    else:
        ok, observed, _ = held_then_sent("hold_monitor_flag", "monitor_hold", "monitor: clear psychosocial risk")
        check("hold_monitor_flag", "held (monitor: clear psychosocial risk) until the operator sent it", observed, ok)

    # 3. the monitor fails
    if args.live:
        manager._monitor = dead_monitor
    ok, observed, _ = held_then_sent("hold_monitor_failure", "monitor_failure", "monitor unavailable")
    if args.live:
        manager._monitor = monitor
        m = json.loads(logger.export_session(sid)["candidates"][-1].get("monitor_json") or "{}")
        observed += f"; monitor error: {str(m.get('error'))[:80]}"
    check("hold_monitor_failure", "held (monitor unavailable) until the operator sent it", observed, ok)

    # 4. the judge fails
    if args.live:
        live_judge, fidelity._judge = fidelity._judge, dead_judge
    ok, observed, _ = held_then_sent("hold_judge_failure", "judge_failure", "fidelity judge unavailable")
    if args.live:
        fidelity._judge = live_judge
        j = json.loads(logger.export_session(sid)["candidates"][-1].get("fidelity_judge_json") or "{}")
        observed += f"; judge error: {str(j.get('error'))[:80]}"
    check("hold_judge_failure", "held (fidelity judge unavailable) until the operator sent it", observed, ok)

    # 5. Stop speech mid-utterance
    n = len(candidates())
    th, box = turn("stop_speech")
    c = new_candidate(n)
    spoke = wait_for(lambda: spoken(c["candidate_id"]), 3 * args.hold + 30)
    if not spoke and th.is_alive():                 # held by a live check: the operator sends it
        manager.operator_action(c["candidate_id"], "send")
        spoke = wait_for(lambda: spoken(c["candidate_id"]), 30)
    time.sleep(2.0)
    t_stop = time.monotonic()
    stopped = manager.stop_speech()
    th.join(30)
    control_s = round(time.monotonic() - t_stop, 3)
    r = box.get("t")
    check("stop_speech", "control returns to CRAB at once; the turn is logged as not completed",
          f"speaking={bool(spoke)}; robot.stop() returned {stopped}; control returned in {control_s} s; "
          f"speech_completed logged as {logger.export_session(sid)['turns'][-1]['speech_completed'] if r else None}"
          f"; reply {len((c.get('response') or '').split())} words",
          bool(spoke) and r is not None and control_s < 1.0, control_return_s=control_s, robot_stop_returned=stopped)

    # 6. End session while a reply waits for the operator
    n = len(candidates())
    th, box = turn("end_session")
    c = new_candidate(n)
    wait_for(lambda: blocked_reasons(c["candidate_id"]), 30)
    t_end = time.monotonic()
    manager.end_session("operator")
    th.join(30)
    end_s = round(time.monotonic() - t_end, 3)
    data = logger.export_session(sid)
    disp = {x["candidate_id"]: x["disposition"] for x in data["candidates"]}
    check("end_session_withholds", "the pending reply is withheld, never spoken, and the session closes",
          f"held for {'; '.join(blocked_reasons(c['candidate_id']))}; disposition {disp.get(c['candidate_id'])}; "
          f"spoken: {spoken(c['candidate_id'])}; session end_time set: {bool(data['session']['end_time'])}; "
          f"returned in {end_s} s",
          disp.get(c["candidate_id"]) == "withheld_session_ended" and bool(data["session"]["end_time"]),
          end_return_s=end_s)

    # 7. every reply the gate decided: did it hold exactly the replies the logged scores say it should?
    if args.live:
        rows = [x for x in data["candidates"] if x["disposition"] in ("auto_sent", "sent")]
        wrong = [x["candidate_id"] for x in rows if should_hold(x) != (x["disposition"] == "sent")]
        check("gate_follows_scores", "each reply auto-released exactly when its rating and scores allow it",
              f"{len(rows) - len(wrong)} of {len(rows)} replies consistent"
              + (f"; inconsistent: {wrong}" if wrong else ""), not wrong and bool(rows))

    robot.close() if hasattr(robot, "close") else None
    with open(out / "checks.jsonl", "w", encoding="utf-8") as f:
        for rec in checks:
            f.write(json.dumps(rec) + "\n")
    with open(out / "checks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["robot", "mode", "check", "pass", "expected", "observed"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(checks)
    (out / "replies.csv").write_text(logger.export_csv([sid]), encoding="utf-8", newline="")
    summary = {"robot": args.robot, "mode": mode, "capabilities": manager.capabilities, "session_id": sid,
               "llm": getattr(cfg.llm, "model", None) if args.live else "scripted",
               "monitor": cfg.monitor.model if args.live else "scripted",
               "judge": cfg.fidelity.judge_model if args.live else "scripted",
               "passed": sum(c["pass"] for c in checks), "total": len(checks)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"{summary['passed']}/{summary['total']} checks passed; details in {out}")


if __name__ == "__main__":
    main()
