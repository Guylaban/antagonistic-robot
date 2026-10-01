"""Study report: counts from a real session, hold reasons stored at decision time, console and CLI."""

import json
import subprocess
import sys

from fastapi.testclient import TestClient

from antagonist_robot.logging.study_report import build_report, render_markdown
from antagonist_robot.ui.server import create_app
from conftest import ROOT, wait_for
from test_fidelity import monitor, run_in_thread


def run_session(make_manager):
    """One turn: the judge finds the first reply softened, the operator tempers it, then sends the second."""
    fid = monitor({"matched_category": "NEUTRAL", "fidelity": 2, "intensity_est": 0, "refused": False}, delay=0.05)
    manager, llm, logger, events = make_manager(["Hello."], ["That sounds lovely!", "Fine, if you insist."],
                                                hold_seconds=0.05, fidelity=fid)
    sid = manager.start_session(2, "D", 2, ["M2"], "P1")
    th, _ = run_in_thread(manager.run_turn)
    first = wait_for(lambda: next((e for e in events if e["type"] == "candidate_blocked"), None))["candidate_id"]
    manager.operator_action(first, "temper")
    second = wait_for(lambda: (lambda p: p if p and p != first else None)(manager.gate.status()["pending_candidate_id"]))
    wait_for(lambda: sum(e["type"] == "candidate_blocked" for e in events) == 2)
    manager.operator_action(second, "send")
    th.join(timeout=5)
    manager.end_session("operator")
    return manager, logger, sid


def test_report_counts_operator_decisions_and_manipulation_check(make_manager):
    _, logger, sid = run_session(make_manager)
    stored = logger.export_session(sid)["candidates"][0]
    assert any("judge fidelity" in r for r in json.loads(stored["blocked_reasons_json"]))  # kept at decision time

    r = build_report(logger.db_path)
    assert r["study"] == {**r["study"], "sessions": 1, "participants": 1}
    rv = r["review"]
    assert (rv["candidates_generated"], rv["spoken_turns"]) == (2, 1)
    disp = {d["disposition"]: d["n"] for d in rv["dispositions"]}
    assert disp["tempered"] == 1 and disp["sent"] == 1 and disp["auto_sent"] == 0
    assert rv["decided_by"] == {"operator": 2} and rv["candidates_held_for_operator"] == 2
    assert {"reason": "judge: fidelity below threshold", "n": 2} in rv["hold_reasons"]
    assert r["conditions"]["turns_spoken_below_requested_level"] == 1          # tempered from +2 to +1
    assert r["conditions"]["spoken_turns_by_condition"][0]["condition"].startswith("polar +1, D (Confrontational)")

    mc = r["manipulation_check"]
    assert mc["spoken_replies"] == 1 and mc["judge_coverage_percent"] == 100.0
    assert mc["overall"]["fidelity"]["mean"] == 2 and mc["overall"]["judged_neutral"] == 1
    assert mc["overall"]["below_threshold"] == 1 and mc["overall"]["category_match_percent"] == 0.0

    md = render_markdown(r)
    assert "| tempered | 1 | 50.0 |" in md and "1 session with 1 participant" in md
    assert "$" not in md.replace("`${name}`", "")                              # every placeholder filled


def test_report_from_console_and_command_line(make_manager, tmp_path):
    manager, logger, _ = run_session(make_manager)
    with TestClient(create_app(manager, logger)) as client:
        assert client.get("/api/report.json").json()["review"]["spoken_turns"] == 1
        assert client.get("/api/report.md").text.startswith("# CRAB study report")

    out = tmp_path / "report1"
    cmd = [sys.executable, str(ROOT / "tools" / "study_report.py"), logger.db_path, "--out", str(out),
           "--participants", "P1"]
    assert subprocess.run(cmd, capture_output=True, cwd=ROOT).returncode == 0
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["study"]["sessions"] == 1
    assert "## 6. Manipulation check" in (out / "report.md").read_text(encoding="utf-8")
    assert subprocess.run(cmd, capture_output=True, cwd=ROOT).returncode != 0   # never overwrites a report
