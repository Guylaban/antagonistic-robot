"""Rehearsal scenarios (tools/simulation): validation, captions, and pause placement."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "simulation"))
import scenario as scn  # noqa: E402

DEMO = ROOT / "tools" / "simulation" / "scenarios" / "demo.yaml"


def minimal(**over):
    sc = {"participant": {"lines": ["Hello there.", "Okay."]},
          "steps": [{"start": {}}, {"reply": {"action": "send"}}, {"end": {}}]}
    sc.update(over)
    return sc


def test_demo_scenario_is_valid():
    sc = scn.load(str(DEMO))
    replies = [s for s in sc["steps"] if "reply" in s]
    assert len(replies) == len(sc["participant"]["lines"]) == 5
    def reply_moments(r):
        own = {v for k, v in r.items() if k.startswith("on_")}
        return own | (reply_moments(r["replacement"]) if r.get("replacement") else set())

    moments = {s.get("moment") for s in sc["steps"]} | {(s.get("end") or {}).get("moment") for s in sc["steps"]}
    moments |= set().union(*(reply_moments(s["reply"]) for s in replies))
    assert set(sc["captions"]) - {"intro"} <= moments                 # every caption has a moment


@pytest.mark.parametrize("bad, msg", [
    (minimal(steps=[{"reply": {}}]), "start step"),
    (minimal(steps=[{"start": {}}, {"reply": {"action": "shout"}}]), "action must be"),
    (minimal(steps=[{"start": {}}] + [{"reply": {}}] * 3), "participant lines"),
    (minimal(steps=[{"start": {}, "end": {}}]), "exactly one"),
    (minimal(captions={"x": {"highlight": ["nowhere"]}}), "unknown highlight"),
    (minimal(participant={"lines": []}), "participant.lines"),
])
def test_invalid_scenarios_are_rejected(bad, msg):
    with pytest.raises(scn.ScenarioError, match=msg):
        scn.validate(bad)


def test_replacement_actions_are_checked():
    sc = minimal(steps=[{"start": {}}, {"reply": {"action": "temper", "replacement": {"action": "nope"}}}])
    with pytest.raises(scn.ScenarioError, match="replacement"):
        scn.validate(sc)


def test_caption_switches_on_the_hold_reason():
    caps = yaml.safe_load(DEMO.read_text(encoding="utf-8"))["captions"]
    plain = scn.caption_for(caps, "t5_held", ["monitor: clear psychosocial risk"])
    assert plain["pause"] == 0 and "monitor holds" in plain["body"]
    soft = scn.caption_for(caps, "t5_held", ["judge fidelity 0/10 below 4 (exhibits NEUTRAL): possible softening"])
    assert soft["essential"] and soft["pause"] > 0
    assert "neutral (fidelity 0/10)" in soft["body"] and soft["highlight"] == ["pending", "fidelity"]
    assert scn.caption_for(caps, "no_such_moment") is None


def test_captions_are_numbered_in_display_order():
    assert scn.number_captions(["intro", "setup", "t1_held", "setup", "end"]) == {"setup": 1, "t1_held": 2, "end": 3}


def test_boxes_outside_the_console_are_dropped():
    assert scn.visible_box((100, 100, 200, 50)) == (100, 100, 200, 50)
    assert scn.visible_box((1400, 880, 200, 100)) is None              # scrolled below the visible page
    assert scn.visible_box((-10, 10, 200, 50)) == (0, 10, 190, 50)       # slightly outside: clipped


def test_pauses_wait_for_silence():
    speech = [(10.0, 14.0), (14.2, 16.0)]
    assert scn.defer_to_silence(5.0, speech) == 5.0
    assert scn.defer_to_silence(11.0, speech) == pytest.approx(16.3)   # after both clips, not inside the gap
