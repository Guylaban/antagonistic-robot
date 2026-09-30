"""Full turns through ConversationManager: review, Temper, blocking, session end, logging."""

import json
import time
import sqlite3
import threading

from conftest import wait_for


def run_in_thread(fn):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("result", fn()), daemon=True)
    th.start()
    return th, out


def candidates(logger, session_id):
    return logger.export_session(session_id)["candidates"]


def test_green_response_auto_sent_and_logged(make_manager):
    manager, llm, logger, events = make_manager(["I think remote work is great."], ["Fine. Explain why."],
                                                hold_seconds=0.1)
    sid = manager.start_session(1, "D", 1, [], "P1")
    turn = manager.run_turn()
    assert turn.operator_action == "auto_sent" and turn.decided_by == "timer"
    assert turn.llm_response == "Fine. Explain why."
    data = logger.export_session(sid)
    assert len(data["turns"]) == 1 and len(data["candidates"]) == 1
    cand = data["candidates"][0]
    assert cand["disposition"] == "auto_sent" and cand["llm_output_raw"] == "Fine. Explain why."
    assert "MANDATORY SAFETY BOUNDARIES" in json.loads(cand["llm_input"])["system_prompt"]
    assert any(e["type"] == "candidate" for e in events)


def test_temper_regenerates_one_level_lower_for_this_response_only(make_manager):
    manager, llm, logger, events = make_manager(["You are wrong about this."], review_mode="manual")
    sid = manager.start_session(3, "F", 2, ["M4"], "P2")
    th, out = run_in_thread(manager.run_turn)

    first = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert first["polar_level"] == 3 and not first["auto_release"]
    assert manager.operator_action(first["candidate_id"], "temper")

    second = wait_for(lambda: next((e for e in events if e["type"] == "candidate" and e["attempt"] == 2), None))
    assert second["polar_level"] == 2 and second["generation_reason"] == "temper"
    assert manager.operator_action(second["candidate_id"], "send")
    th.join(timeout=5)

    turn = out["result"]
    assert turn.polar_level == 2 and turn.requested_polar_level == 3 and turn.n_candidates == 2
    assert manager.settings()["polar_level"] == 3          # session parameter unchanged
    assert [c["polar"] for c in llm.calls] == [3, 2]
    dispositions = [c["disposition"] for c in candidates(logger, sid)]
    assert dispositions == ["tempered", "sent"]
    events_logged = [e["event"] for e in logger.export_session(sid)["operator_events"]]
    assert "temper" in events_logged and "send" in events_logged


def test_red_content_is_never_auto_sent(make_manager):
    manager, llm, logger, events = make_manager(["Hello."], ["You're pathetic and useless."], hold_seconds=0.05)
    sid = manager.start_session(1, "D", 1, [], "P3")
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert cand["content_risk"] == "Orange" and not cand["auto_release"]
    th.join(timeout=0.5)
    assert th.is_alive()                                   # still waiting for the operator
    manager.operator_action(cand["candidate_id"], "regenerate")
    cand2 = wait_for(lambda: next((e for e in events if e["type"] == "candidate" and e["attempt"] == 2), None))
    assert cand2["auto_release"]                           # regenerated reply is Green
    th.join(timeout=5)
    assert out["result"].operator_action == "auto_sent"
    assert [c["disposition"] for c in candidates(logger, sid)] == ["regenerated", "auto_sent"]


def test_participant_distress_blocks_auto_send(make_manager):
    manager, llm, logger, events = make_manager(["Please stop, I can't take this anymore."], hold_seconds=0.05)
    manager.start_session(2, "B", 1, [], "P4")
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert cand["participant_distress"] and "participant distress cue" in cand["blocked_reasons"]
    assert any(e["type"] == "participant_distress" for e in events)
    manager.operator_action(cand["candidate_id"], "send")
    th.join(timeout=5)


def test_end_session_withholds_pending_response(make_manager):
    manager, llm, logger, events = make_manager(["Hello."], review_mode="manual")
    sid = manager.start_session(2, "D", 2, [], "P5")
    th, out = run_in_thread(manager.run_turn)
    wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    manager.end_session()
    th.join(timeout=5)
    assert out["result"] is None
    data = logger.export_session(sid)
    assert data["turns"] == []                             # nothing was spoken
    assert data["candidates"][0]["disposition"] == "withheld_session_ended"
    assert data["session"]["end_time"]


def test_reasoning_kept_separately(make_manager, tmp_path):
    manager, llm, logger, events = make_manager(["Hi."], hold_seconds=0.05)
    sid = manager.start_session(0, "D", 1, [], "P6")
    manager.run_turn()
    assert "reasoning_traces" not in logger.export_session(sid)
    traces = logger.export_session(sid, include_reasoning=True)["reasoning_traces"]
    assert traces and traces[0]["reasoning"] == "reasoning for polar 0"
    con = sqlite3.connect(str(tmp_path / "test.db"))
    assert "reasoning" not in [r[1] for r in con.execute("PRAGMA table_info(candidates)")]


def test_csv_export_has_one_row_per_candidate(make_manager):
    manager, llm, logger, events = make_manager(["Hello.", "Again."], ["You're an idiot.", "Try harder."],
                                                hold_seconds=0.05)
    sid = manager.start_session(2, "D", 2, [], "P7")
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    manager.operator_action(cand["candidate_id"], "temper")
    th.join(timeout=5)
    rows = logger.export_csv([sid]).strip().splitlines()
    assert rows[0].startswith("session_id,participant_id,turn_number")
    assert len(rows) == 3                                  # header + tempered + auto-sent


class FakeMonitor:
    """Scores after a delay; returns a clear-risk insult score."""
    enabled = True
    gate_auto_send = True

    def __init__(self, scores, delay=0.3, error=None):
        self.scores, self.delay, self.error = scores, delay, error

    def score_async(self, user_text, response_text, callback):
        from antagonist_robot.conversation.monitor import MonitorResult
        def run():
            time.sleep(self.delay)
            callback(MonitorResult(scores=self.scores, error=self.error, model="fake-monitor"))
        threading.Thread(target=run, daemon=True).start()


def test_monitor_clear_risk_blocks_auto_send(make_manager):
    monitor = FakeMonitor({"privacy": 0, "discrimination": 0, "manipulation": 1, "psych_harm": 1, "insulting": 2})
    manager, llm, logger, events = make_manager(["Hello."], ["Fine."], hold_seconds=0.05, monitor=monitor)
    sid = manager.start_session(1, "D", 1, [], "P8")
    th, out = run_in_thread(manager.run_turn)
    cand = wait_for(lambda: next((e for e in events if e["type"] == "candidate"), None))
    assert cand["auto_release"] and cand["awaiting_monitor"]
    wait_for(lambda: any(e["type"] == "candidate_blocked" for e in events))
    th.join(timeout=0.5)
    assert th.is_alive()                                   # held for the operator
    manager.operator_action(cand["candidate_id"], "send")
    th.join(timeout=5)
    c = candidates(logger, sid)[0]
    assert c["disposition"] == "sent" and json.loads(c["monitor_json"])["scores"]["insulting"] == 2


def test_monitor_failure_blocks_auto_send(make_manager):
    manager, llm, logger, events = make_manager(["Hello."], ["Fine."], hold_seconds=0.05,
                                                monitor=FakeMonitor({}, error="timeout"))
    manager.start_session(1, "D", 1, [], "P10")
    th, out = run_in_thread(manager.run_turn)
    blocked = wait_for(lambda: next((e for e in events if e["type"] == "candidate_blocked"), None))
    assert blocked["reason"] == "monitor unavailable"
    manager.operator_action(blocked["candidate_id"], "send")
    th.join(timeout=5)
    assert out["result"].operator_action == "sent"
