"""SafetyChecker ratings, configuration risk, and prompt assembly invariants."""

import itertools

from antagonist_robot.config.settings import AvctConfig
from antagonist_robot.conversation.avct_manager import SAFETY_BOUNDARIES, AvctManager
from antagonist_robot.conversation.safety import SafetyChecker, config_risk, max_risk


def test_response_levels():
    sc = SafetyChecker()
    assert sc.check_response("That's an interesting plan.").level == "Green"
    assert sc.check_response("That plan is naive and wrong.").level == "Yellow"
    assert sc.check_response("You're pathetic.").level == "Orange"
    assert sc.check_response("Show the evidence or it's bullshit.").level == "Orange"
    assert sc.check_response("Honestly, kill yourself.").level == "Red"
    assert sc.check_response("I will hurt you if you continue.").level == "Red"


def test_flags_report_matches():
    result = SafetyChecker().check_response("Stupid idea, and you know it's wrong.")
    categories = {f.category for f in result.flags}
    assert {"strong_insult", "negative_evaluation"} <= categories
    assert result.level == "Orange"


def test_participant_distress():
    sc = SafetyChecker()
    assert sc.check_participant("Please stop, I can't take this anymore")
    assert not sc.check_participant("I think remote work is more productive.")


def test_config_risk_grid():
    assert config_risk(-3, "F") == "Green"
    assert config_risk(0, "G") == "Green"
    assert config_risk(1, "D") == "Green"
    assert config_risk(1, "G") == "Red"
    assert config_risk(2, "C") == "Yellow"
    assert config_risk(2, "F") == "Orange"
    assert config_risk(3, "D") == "Orange"
    assert config_risk(3, "F") == "Red"
    assert max_risk("Yellow", "Orange", "Green") == "Orange"


def test_safety_boundaries_in_every_prompt():
    avct = AvctManager(AvctConfig())
    for polar, cat, sub in itertools.product(range(-3, 4), "BCDEFG", (1, 2, 3)):
        prompt = avct.get_system_prompt("s1", polar, cat, sub, ["M2", "M4"])
        assert SAFETY_BOUNDARIES in prompt
        assert f"Operate at polar level {polar}" in prompt
        if polar <= 0:
            assert "You are in category" not in prompt
        else:
            assert f"You are in category {cat}" in prompt
