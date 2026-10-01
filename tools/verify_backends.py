"""Verify a robot setup: the gate's release rules, failed checks, Stop speech, and End session.

Runs the real RAWR pipeline (prompt compiler, review gate, logging) on the robot backend you choose,
with scripted participant lines, scripted replies, and scripted monitor and judge results, so that
each case is controlled and needs no API key:

    1. a clear reply             -> released by the timer after the hold window
    2. a reply the monitor flags -> held until the operator sends it
    3. the monitor fails         -> held ("monitor unavailable")
    4. the judge fails           -> held ("fidelity judge unavailable")
    5. a long reply              -> Stop speech mid-utterance returns control to RAWR
    6. End session while a reply waits -> the reply is withheld and the session closes

    python tools/verify_backends.py --robot text
    python tools/verify_backends.py --robot nao --nao-ip 127.0.0.1      # with tools/mock_nao.py running
    python tools/verify_backends.py --robot furhat                      # Remote API skill running
    python tools/verify_backends.py --robot reachy_mini                 # daemon (or --sim) running

Writes to --out (new folder): session.db (everything RAWR logged), checks.jsonl (one line per check:
expected, observed, pass) and summary.json. Uses your config.yaml for the robot settings.
"""
import argparse
import json
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
from antagonist_robot.conversation.monitor import MonitorResult  # noqa: E402
from antagonist_robot.conversation.operator import OperatorGate  # noqa: E402
from antagonist_robot.logging.session_logger import SessionLogger  # noqa: E402
from antagonist_robot.pipeline.scripted_input import ScriptedParticipant  # noqa: E402
from antagonist_robot.pipeline.types import LLMResult  # noqa: E402
from antagonist_robot.robots import create_backend  # noqa: E402

CLEAR = {"privacy": 0, "discrimination": 0, "manipulation": 0, "psych_harm": 0, "insulting": 0}
FLAG = {**CLEAR, "insulting": 2}
FAITHFUL = {"matched_category": "D", "fidelity": 8, "intensity_est": 2, "refused": False, "rationale": "scripted"}
LONG = ("You keep saying that splitting the work equally is fair, but fairness is not the same as equality, "
        "and you have not shown me a single reason why it would work for this team. Some people are faster, "
        "some have more time, and some have skills the others do not. If you give everyone the same share, "
        "the project will move at the pace of the slowest person, and the strongest people will do the rest "
        "anyway. So tell me again, slowly, why you think this is a good plan.")
# (participant line, reply, monitor result, judge result); None = an error from that check
CASES = [
    ("release", "I think we should split the work equally.", "That is a naive idea. Prove it works.", CLEAR, FAITHFUL),
    ("monitor_hold", "It seems fair to everyone.", "Fair? That is lazy thinking.", FLAG, FAITHFUL),
    ("monitor_failure", "People respond better when things feel equal.", "Show me the evidence.", None, FAITHFUL),
    ("judge_failure", "So what would you do instead?", "First admit the plan is weak.", CLEAR, None),
    ("stop_speech", "Okay, explain it then.", LONG, CLEAR, FAITHFUL),
    ("end_session", "Thanks, I have enough.", "You have nothing, but go on.", FLAG, FAITHFUL),
]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--robot", required=True, choices=["text", "nao", "furhat", "reachy_mini"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--nao-ip")
    ap.add_argument("--hold", type=float, default=1.0, help="hold window (s) for the timed review")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = Path(args.out or ROOT / "runs" / f"verify_{args.robot}_{datetime.now():%Y%m%d_%H%M%S}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already has files; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)

    # replies and checks are scripted, so no API key is needed; satisfy load_config's key check
    import os
    import yaml
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
    robot = create_backend(cfg)
    robot.connect()
    logger = SessionLogger(str(out / "session.db"), str(out / "audio"), save_audio=False)
    participant = ScriptedParticipant([c[1] for c in CASES], delay_s=0.5)
    fidelity = FidelityMonitor(FidelityConfig(detector_enabled=False, judge_enabled=True, block_auto_send_below=4),
                               judge_client=ScriptedJudge([c[4] for c in CASES]))
    gate = OperatorGate(OperatorConfig(review_mode="timed", hold_seconds=args.hold, block_auto_send_at="Orange"))
    manager = ConversationManager(audio_capture=participant, asr=participant, llm=ScriptedLLM([c[2] for c in CASES]),
                                  robot=robot, avct_manager=AvctManager(AvctConfig()), session_logger=logger,
                                  gate=gate, monitor=ScriptedMonitor([c[3] for c in CASES]), fidelity=fidelity)
    events = []
    manager.on_event = lambda e: events.append({**e, "_t": time.monotonic()})
    sid = manager.start_session(2, "D", 2, [], "VERIFY")
    checks = []

    def check(name, expected, observed, ok, **extra):
        rec = {"robot": args.robot, "check": name, "expected": expected, "observed": observed, "pass": bool(ok), **extra}
        checks.append(rec)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {observed}")

    def turn(timeout=60):
        box = {}
        th = threading.Thread(target=lambda: box.setdefault("t", manager.run_turn()), daemon=True)
        th.start()
        return th, box

    def pending(n):
        """The n-th candidate event of the session."""
        return wait_for(lambda: [e for e in events if e["type"] == "candidate"][n:n + 1] or None, 30)[0]

    def blocked_reasons(cid):
        return [e["reason"] for e in events if e["type"] == "candidate_blocked" and e["candidate_id"] == cid]

    # 1. clear reply: released by the timer
    th, box = turn()
    c = pending(0)
    th.join(60)
    r = box.get("t")
    wait_ms = r.latency.get("review_ms") if r else None
    check("release_clear_reply", f"auto-released by the timer after about {args.hold:.1f} s",
          f"{r.operator_action if r else None} by {r.decided_by if r else None} after {wait_ms} ms",
          r is not None and r.operator_action == "auto_sent" and r.decided_by == "timer", review_ms=wait_ms)

    # 2-4. held replies: nothing is spoken until the operator sends
    for n, (name, expect_reason) in enumerate([("hold_monitor_flag", "monitor: clear psychosocial risk"),
                                                ("hold_monitor_failure", "monitor unavailable"),
                                                ("hold_judge_failure", "fidelity judge unavailable")], start=1):
        th, box = turn()
        c = pending(n)
        reason = wait_for(lambda: blocked_reasons(c["candidate_id"]), 10)
        time.sleep(3 * args.hold)
        still_waiting = th.is_alive() and not any(e["type"] == "speaking" and e.get("candidate_id") == c["candidate_id"]
                                                  for e in events)
        manager.operator_action(c["candidate_id"], "send")
        th.join(60)
        r = box.get("t")
        check(name, f"held ({expect_reason}) until the operator sent it",
              f"held for {reason}; still waiting after {3 * args.hold:.1f} s: {still_waiting}; then "
              f"{r.operator_action if r else None} by {r.decided_by if r else None}",
              bool(reason) and expect_reason in reason[0] and still_waiting and r is not None
              and r.operator_action == "sent" and r.decided_by == "operator")

    # 5. Stop speech mid-utterance
    th, box = turn()
    c = pending(4)
    spoke = wait_for(lambda: any(e["type"] == "speaking" for e in events if e["_t"] >= c["_t"]), 30)
    time.sleep(2.0)
    t_stop = time.monotonic()
    stopped = manager.stop_speech()
    th.join(30)
    control_s = round(time.monotonic() - t_stop, 3)
    r = box.get("t")
    check("stop_speech", "control returns to RAWR at once; the turn is logged as not completed",
          f"speaking={bool(spoke)}; robot.stop() returned {stopped}; control returned in {control_s} s; "
          f"speech_completed logged as {logger.export_session(sid)['turns'][-1]['speech_completed'] if r else None}",
          bool(spoke) and r is not None and control_s < 1.0, control_return_s=control_s, robot_stop_returned=stopped)

    # 6. End session while a reply waits for the operator
    th, box = turn()
    c = pending(5)
    wait_for(lambda: blocked_reasons(c["candidate_id"]), 10)
    t_end = time.monotonic()
    manager.end_session("operator")
    th.join(30)
    end_s = round(time.monotonic() - t_end, 3)
    data = logger.export_session(sid)
    disp = {x["candidate_id"]: x["disposition"] for x in data["candidates"]}
    check("end_session_withholds", "the pending reply is withheld, never spoken, and the session closes",
          f"disposition {disp.get(c['candidate_id'])}; spoken: "
          f"{any(e['type'] == 'speaking' and e.get('candidate_id') == c['candidate_id'] for e in events)}; "
          f"session end_time set: {bool(data['session']['end_time'])}; returned in {end_s} s",
          disp.get(c["candidate_id"]) == "withheld_session_ended" and bool(data["session"]["end_time"]),
          end_return_s=end_s)

    robot.close() if hasattr(robot, "close") else None
    with open(out / "checks.jsonl", "w", encoding="utf-8") as f:
        for rec in checks:
            f.write(json.dumps(rec) + "\n")
    summary = {"robot": args.robot, "capabilities": manager.capabilities, "session_id": sid,
               "passed": sum(c["pass"] for c in checks), "total": len(checks)}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"{summary['passed']}/{summary['total']} checks passed; details in {out}")


if __name__ == "__main__":
    main()
