"""Operator review gate between response generation and robot speech.

Every candidate response is held here before the robot may speak it. The
operator sees it in the console and can act on it; if they do nothing, the
gate applies the configured release policy:

- review_mode "timed": the response is released automatically after
  hold_seconds, unless its risk is at or above block_auto_send_at, the
  participant showed distress cues, or the operator pressed Hold. Those
  responses wait for an explicit operator action. Background signals that
  gate release (the psychosocial monitor, the fidelity judge) make the
  timer also wait for their scores; a flag or a failure from any of them
  blocks automatic release.
- review_mode "manual": nothing is released without an explicit action.

Operator actions:
    send        release the response now
    temper      discard it and regenerate one polar level lower
                (this response only; session parameters are unchanged)
    intensify   discard it and regenerate one polar level higher, up to +3
                (this response only; for replies that under-deliver)
    regenerate  discard it and regenerate with the same parameters
    hold        cancel the automatic release for this response
    end         the session is ending; nothing is released

The gate is thread-safe: wait() blocks the conversation thread while
act() is called from the web server thread.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from antagonist_robot.config.settings import OperatorConfig
from antagonist_robot.conversation.safety import risk_index

ACTIONS = ("send", "temper", "intensify", "regenerate", "hold", "end")


@dataclass
class GateDecision:
    """Outcome of one review."""
    action: str            # send | auto_send | temper | intensify | regenerate | end
    decided_by: str        # "operator" | "timer" | "system"
    wait_ms: int           # time the response was held
    held_for: list = field(default_factory=list)  # every reason automatic release was cancelled


class OperatorGate:
    """Holds one candidate response at a time until it is released or replaced."""

    def __init__(self, config: OperatorConfig):
        self.review_mode = config.review_mode
        self.hold_seconds = float(config.hold_seconds)
        self.block_auto_send_at = config.block_auto_send_at
        self._cond = threading.Condition()
        self._candidate_id: Optional[int] = None
        self._action: Optional[str] = None
        self._auto_allowed = False
        self._deadline: Optional[float] = None
        self._blocked_reasons: list = []
        self._awaiting: set = set()

    # --- policy ---------------------------------------------------------

    def auto_release_allowed(self, risk_level: str, distress: bool) -> bool:
        """Whether a response with this rating may be released by the timer."""
        if self.review_mode != "timed":
            return False
        if distress:
            return False
        return risk_index(risk_level) < risk_index(self.block_auto_send_at)

    def set_policy(self, review_mode: Optional[str] = None, hold_seconds: Optional[float] = None) -> None:
        """Change the release policy live (applies from the next candidate)."""
        with self._cond:
            if review_mode in ("timed", "manual"):
                self.review_mode = review_mode
            if hold_seconds is not None:
                self.hold_seconds = max(0.0, float(hold_seconds))

    # --- conversation-thread side ----------------------------------------

    def open(self, candidate_id: int, risk_level: str, distress: bool, await_monitor: bool = False,
             await_signals: Iterable[str] = ()) -> dict:
        """Register a new pending candidate and return its release terms.

        Automatic release additionally waits until signal_done() has been
        called for every name in await_signals ("monitor" if await_monitor).
        """
        with self._cond:
            self._candidate_id = candidate_id
            self._action = None
            self._auto_allowed = self.auto_release_allowed(risk_level, distress)
            signals = set(await_signals) | ({"monitor"} if await_monitor else set())
            self._awaiting = signals if self._auto_allowed else set()
            self._deadline = time.monotonic() + self.hold_seconds if self._auto_allowed else None
            self._blocked_reasons = []
            if not self._auto_allowed:
                if self.review_mode == "manual":
                    self._blocked_reasons.append("manual review mode")
                if distress:
                    self._blocked_reasons.append("participant distress cue")
                if risk_index(risk_level) >= risk_index(self.block_auto_send_at):
                    self._blocked_reasons.append(f"risk {risk_level}")
            return {
                "auto_release": self._auto_allowed,
                "hold_seconds": self.hold_seconds if self._auto_allowed else None,
                "awaiting_monitor": "monitor" in self._awaiting,
                "awaiting": sorted(self._awaiting),
                "blocked_reasons": list(self._blocked_reasons),
            }

    def wait(self, is_active: Callable[[], bool]) -> GateDecision:
        """Block until the pending candidate is acted on, released, or the session ends."""
        start = time.monotonic()
        with self._cond:
            while True:
                if not is_active():
                    return self._close("end", "system", start)
                if self._action is not None:
                    by = "system" if self._action == "end" else "operator"
                    return self._close(self._action, by, start)
                if self._auto_allowed and self._deadline is not None:
                    remaining = self._deadline - time.monotonic()
                    if remaining <= 0 and not self._awaiting:
                        return self._close("auto_send", "timer", start)
                    # hold window still running, or over but still waiting for a signal
                    self._cond.wait(timeout=min(max(remaining, 0.0), 0.1) or 0.1)
                else:
                    self._cond.wait(timeout=0.1)

    def _close(self, action: str, by: str, start: float) -> GateDecision:
        held_for = list(self._blocked_reasons)
        self._candidate_id = None
        self._action = None
        self._deadline = None
        self._awaiting = set()
        return GateDecision(action=action, decided_by=by, wait_ms=round((time.monotonic() - start) * 1000),
                            held_for=held_for)

    # --- web-server / signal side -------------------------------------------

    def act(self, candidate_id: Optional[int], action: str) -> bool:
        """Apply an operator action to the pending candidate.

        Returns False if there is no pending candidate or the id is stale
        (the operator clicked on a response that was already replaced).
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}; expected one of {ACTIONS}")
        with self._cond:
            if self._candidate_id is None:
                return False
            if candidate_id is not None and candidate_id != self._candidate_id:
                return False
            if action == "hold":
                self._auto_allowed = False
                self._deadline = None
                self._blocked_reasons.append("held by operator")
            else:
                self._action = action
            self._cond.notify_all()
            return True

    def escalate(self, candidate_id: int, reason: str) -> bool:
        """A background signal flagged the pending candidate: cancel automatic release
        and add the reason, also when the candidate was already held for another reason,
        so the operator sees every flag before deciding."""
        with self._cond:
            if candidate_id != self._candidate_id:
                return False
            self._auto_allowed = False
            self._deadline = None
            if reason not in self._blocked_reasons:
                self._blocked_reasons.append(reason)
            self._cond.notify_all()
            return True

    def signal_done(self, candidate_id: int, name: str) -> None:
        """A background signal has scored this candidate."""
        with self._cond:
            if candidate_id == self._candidate_id:
                self._awaiting.discard(name)
                self._cond.notify_all()

    def monitor_done(self, candidate_id: int) -> None:
        """The psychosocial monitor has scored this candidate."""
        self.signal_done(candidate_id, "monitor")

    def end(self) -> None:
        """Release any waiter because the session is ending."""
        with self._cond:
            if self._candidate_id is not None:
                self._action = "end"
            self._cond.notify_all()

    @property
    def pending_id(self) -> Optional[int]:
        return self._candidate_id

    def status(self) -> dict:
        """Current pending state for the console."""
        with self._cond:
            remaining = None
            if self._deadline is not None:
                remaining = max(0.0, round(self._deadline - time.monotonic(), 2))
            return {
                "pending_candidate_id": self._candidate_id,
                "auto_release": self._auto_allowed,
                "awaiting_monitor": "monitor" in self._awaiting,
                "awaiting": sorted(self._awaiting),
                "seconds_remaining": remaining,
                "blocked_reasons": list(self._blocked_reasons),
                "review_mode": self.review_mode,
                "hold_seconds": self.hold_seconds,
            }
