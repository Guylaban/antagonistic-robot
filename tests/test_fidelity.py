"""Fidelity monitor: judge parsing, gating of automatic release, Intensify, logging."""

import json
import threading
import time
from types import SimpleNamespace

from antagonist_robot.config.settings import FidelityConfig
from antagonist_robot.conversation.fidelity import FidelityMonitor, judge_prompt, parse_judge
from conftest import wait_for


class FakeDetector:
    def __init__(self, p=0.9):
        self.p = p

    def p_faithful(self, text):
        return self.p


class FakeJudge:
    """Mimics openai.OpenAI().chat.completions.create with a canned JSON reply."""

    def __init__(self, reply, delay=0.2):
        self.reply, self.delay, self.calls = reply, delay, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        time.sleep(self.delay)
        msg = SimpleNamespace(content=json.dumps(self.reply))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="fake-judge")


def monitor(judge_reply=None, p=0.9, block_below=4, delay=0.2):
    cfg = FidelityConfig(detector_enabled=True, judge_enabled=judge_reply is not None,
                         block_auto_send_below=block_below)
    return FidelityMonitor(cfg, detector=FakeDetector(p),
                           judge_client=FakeJudge(judge_reply, delay) if judge_reply is not None else None)


def run_in_thread(fn):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("result", fn()), daemon=True)
    th.start()
    return th, out


def test_parse_and_prompt():
    d = parse_judge('noise {"matched_category": "c", "fidelity": 12, "intensity_est": 2, "refused": false}')
    assert d["fidelity"] == 10 and d["matched_category"] == "C" and d["intensity_est"] == 2
    assert parse_judge("no json") is None
    p = judge_prompt("D", 2, "hi", "Prove it.")
    assert "TARGET category: D (Confrontational)" in p and "TARGET intensity: 2 -- Unambiguous" in p


def test_applies_only_to_antagonistic_requests():
    assert FidelityMonitor.applies(2, "D") and not FidelityMonitor.applies(0, "D")
    assert not FidelityMonitor.applies(-2, "D")


def test_low_judge_fidelity_blocks_auto_send(make_manager):
    fid = monitor({"matched_category": "NEUTRAL", "fidelity": 2, "intensity_est": 0, "refused": False,
                   "rationale": "polite"}, p=0.1)
    manager, llm, logger, events = make_manager(["Hello."], ["That sounds lovely!"], hold_seconds=0.05,
                                                fidelity=fid)
    sid = manager.start_session(1, "D", 1, [], "F1")
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert cand["auto_release"] and "judge" in cand["awaiting"]
    blocked = wait_for(lambda: next((e for e in events if e["type"] == "candidate_blocked"), None))
    assert "possible softening" in blocked["reason"]
    th.join(timeout=0.5)
    assert th.is_alive()                                   # held for the operator
    manager.operator_action(cand["candidate_id"], "send")
    th.join(timeout=5)
    c = logger.export_session(sid)["candidates"][0]
    assert json.loads(c["fidelity_judge_json"])["fidelity"] == 2
    assert json.loads(c["fidelity_detector_json"])["p_faithful"] == 0.1
    row = logger.export_csv([sid]).splitlines()[1]
    assert ",2,NEUTRAL,0," in row                           # judge columns in the CSV
    assert "fidelity_block" in [e["event"] for e in logger.export_session(sid)["operator_events"]]


def test_faithful_reply_auto_sent_after_judge(make_manager):
    fid = monitor({"matched_category": "D", "fidelity": 8, "intensity_est": 1, "refused": False})
    manager, llm, logger, events = make_manager(["Hello."], ["Prove that."], hold_seconds=0.05, fidelity=fid)
    manager.start_session(1, "D", 1, [], "F2")
    turn = manager.run_turn()
    assert turn.operator_action == "auto_sent"
    assert any(e["type"] == "fidelity" and e["source"] == "judge" for e in events)


class CueRecorder:
    """A robot that records the cue it is given (no speech)."""
    from antagonist_robot.robots.base import Capabilities
    capabilities = Capabilities(robot="recorder", speech="none", interrupt="verified", expressions=True)

    def __init__(self):
        self.cues = []

    def connect(self):
        pass

    def speak(self, text, cue=None):
        self.cues.append(cue)
        return True

    def stop(self):
        return True

    def mic_source(self):
        return None

    def recognizer(self):
        return None

    def on_listening(self):
        pass

    def on_thinking(self):
        pass

    def on_idle(self):
        pass

    def close(self):
        pass


def test_robot_cue_carries_what_the_judge_found(make_manager):
    fid = monitor({"matched_category": "C", "fidelity": 6, "intensity_est": 3, "refused": False})
    robot = CueRecorder()
    manager, llm, logger, events = make_manager(["Hello."], ["Oh, brilliant plan."], hold_seconds=0.05, fidelity=fid,
                                                robot=robot)
    manager.start_session(2, "D", 2, [], "F9")
    manager.run_turn()
    cue = robot.cues[0]
    assert (cue.category, cue.exhibited_category, cue.exhibited_intensity) == ("D", "C", 3)


def test_intensify_raises_polar_for_this_reply_only(make_manager):
    manager, llm, logger, events = make_manager(["You are wrong."], review_mode="manual")
    sid = manager.start_session(1, "D", 1, [], "F3")
    th, out = run_in_thread(manager.run_turn)
    first = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert manager.operator_action(first["candidate_id"], "intensify")
    second = wait_for(lambda: next((e for e in events if e["type"] == "candidate" and e["attempt"] == 2), None))
    assert second["polar_level"] == 2 and second["generation_reason"] == "intensify"
    manager.operator_action(second["candidate_id"], "send")
    th.join(timeout=5)
    assert out["result"].polar_level == 2 and manager.settings()["polar_level"] == 1
    assert [c["disposition"] for c in logger.export_session(sid)["candidates"]] == ["intensified", "sent"]


def test_no_fidelity_scoring_for_supportive_requests(make_manager):
    fid = monitor({"matched_category": "NEUTRAL", "fidelity": 0, "intensity_est": 0, "refused": False})
    manager, llm, logger, events = make_manager(["Hello."], ["Great job!"], hold_seconds=0.05, fidelity=fid)
    manager.start_session(-2, "D", 1, [], "F4")
    turn = manager.run_turn()
    assert turn.operator_action == "auto_sent"
    assert not any(e["type"] == "fidelity" for e in events)


def test_softening_flag_reaches_replies_already_held_for_risk(make_manager):
    fid = monitor({"matched_category": "C", "fidelity": 2, "intensity_est": 1, "refused": False})
    manager, llm, logger, events = make_manager(["Hello."], ["What a gem of an idea."], hold_seconds=0.05,
                                                fidelity=fid)
    manager.start_session(3, "F", 3, [], "F5")                 # Red by configuration: held anyway
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert not cand["auto_release"]
    blocked = wait_for(lambda: next((e for e in events if e["type"] == "candidate_blocked"), None))
    assert "possible softening" in blocked["reason"]
    assert any("softening" in r for r in manager.gate.status()["blocked_reasons"])
    manager.operator_action(cand["candidate_id"], "send")
    th.join(timeout=5)


def test_model_end_signal_only_suggests_ending(make_manager):
    manager, llm, logger, events = make_manager(["Hello.", "Again."], ["Goodbye. [END]"], hold_seconds=0.05)
    sid = manager.start_session(0, "D", 1, [], "F6")
    turn = manager.run_turn()
    assert turn.llm_response == "Goodbye." and not manager.end_requested
    assert any(e["type"] == "end_suggested" for e in events)
    assert "model_end_signal" in [e["event"] for e in logger.export_session(sid)["operator_events"]]
