"""Release policy of the operator review gate."""

import threading
import time

from antagonist_robot.config.settings import OperatorConfig
from antagonist_robot.conversation.operator import OperatorGate


def gate(**kw):
    cfg = dict(review_mode="timed", hold_seconds=0.2, block_auto_send_at="Orange")
    cfg.update(kw)
    return OperatorGate(OperatorConfig(**cfg))


def test_timed_auto_release():
    g = gate()
    terms = g.open(1, "Yellow", distress=False)
    assert terms["auto_release"]
    d = g.wait(lambda: True)
    assert d.action == "auto_send" and d.decided_by == "timer" and d.wait_ms >= 150


def test_orange_and_distress_and_manual_block_auto_release():
    assert not gate().open(1, "Orange", distress=False)["auto_release"]
    assert not gate().open(1, "Green", distress=True)["auto_release"]
    assert not gate(review_mode="manual").open(1, "Green", distress=False)["auto_release"]


def _act_later(g, cid, action, delay=0.1):
    threading.Thread(target=lambda: (time.sleep(delay), g.act(cid, action)), daemon=True).start()


def test_blocked_waits_for_operator():
    g = gate()
    g.open(7, "Red", distress=False)
    _act_later(g, 7, "send", 0.3)
    d = g.wait(lambda: True)
    assert d.action == "send" and d.decided_by == "operator" and d.wait_ms >= 250


def test_hold_cancels_timer_and_stale_ids_are_rejected():
    g = gate(hold_seconds=0.1)
    g.open(3, "Green", distress=False)
    assert g.act(3, "hold")
    assert not g.act(99, "send")          # stale id
    _act_later(g, 3, "temper", 0.3)
    d = g.wait(lambda: True)
    assert d.action == "temper"


def test_session_end_releases_waiter():
    g = gate()
    g.open(5, "Red", distress=False)
    threading.Thread(target=lambda: (time.sleep(0.1), g.end()), daemon=True).start()
    d = g.wait(lambda: True)
    assert d.action == "end" and d.decided_by == "system"


def test_auto_release_waits_for_monitor():
    g = gate(hold_seconds=0.05)
    assert g.open(4, "Green", distress=False, await_monitor=True)["awaiting_monitor"]
    threading.Thread(target=lambda: (time.sleep(0.4), g.monitor_done(4)), daemon=True).start()
    d = g.wait(lambda: True)
    assert d.action == "auto_send" and d.wait_ms >= 350     # not released at 50 ms


def test_escalate_cancels_auto_release():
    g = gate(hold_seconds=0.5)
    g.open(2, "Green", distress=False)
    assert g.escalate(2, "monitor")
    _act_later(g, 2, "send", 0.7)
    d = g.wait(lambda: True)
    assert d.action == "send" and d.decided_by == "operator"
