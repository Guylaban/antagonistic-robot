"""Conversation manager: orchestrates the turn-based conversation loop.

One turn:
    capture -> ASR -> prompt compilation -> LLM -> safety rating
    -> operator review gate -> robot speech -> logging

Each step completes before the next starts. No generated response
reaches the robot without passing the operator gate (operator.py): it is
held for review, and the operator can send it, temper it (regenerate one
polar level lower, for this response only), regenerate it, hold it, or
end the session. Every generated response is logged, including the ones
that were never spoken.
"""

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from antagonist_robot.conversation.avct_manager import AvctManager
from antagonist_robot.conversation.fidelity import FidelityMonitor
from antagonist_robot.conversation.history import ConversationHistory
from antagonist_robot.conversation.monitor import MonitorResult, PsychosocialMonitor
from antagonist_robot.conversation.operator import OperatorGate
from antagonist_robot.conversation.safety import SafetyChecker, config_risk, max_risk
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.pipeline.llm import LLMEngine
from antagonist_robot.pipeline.types import LLMResult, TurnResult
from antagonist_robot.robots.base import RobotBackend, SpeechCue

log = logging.getLogger(__name__)

_END_PATTERN = re.compile(r'\[end\]', re.IGNORECASE)

FALLBACK_RESPONSE = "I see. Go on."


def extract_end_signal(text: str) -> tuple[str, bool]:
    """Check for [END] sentinel token and return cleaned text.

    Returns:
        (cleaned_text, end_detected): The text with [END] stripped and
        whether the end signal was found. Case-insensitive.
    """
    if _END_PATTERN.search(text):
        cleaned = _END_PATTERN.sub('', text).strip()
        return cleaned, True
    return text, False


class SystemState:
    """Observable state constants for the UI."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    REVIEW = "review"
    SPEAKING = "speaking"


class ConversationManager:
    """Orchestrates the turn-based voice conversation loop."""

    def __init__(
        self,
        audio_capture,
        asr,
        llm: LLMEngine,
        robot: RobotBackend,
        avct_manager: AvctManager,
        session_logger: SessionLogger,
        gate: OperatorGate,
        safety: Optional[SafetyChecker] = None,
        monitor: Optional[PsychosocialMonitor] = None,
        fidelity: Optional[FidelityMonitor] = None,
        config_snapshot: Optional[dict] = None,
        model_can_end_session: bool = False,
    ):
        self._capture = audio_capture
        self._asr = asr
        self._llm = llm
        self._robot = robot
        self._avct = avct_manager
        self._logger = session_logger
        self._gate = gate
        self._judged: dict = {}          # candidate id -> fidelity judge result (for non-verbal cues)
        self._safety = safety or SafetyChecker()
        self._monitor = monitor
        self._fidelity = fidelity
        self._config_snapshot = config_snapshot
        self._model_can_end = model_can_end_session

        self._history = ConversationHistory()
        self._session_id: Optional[str] = None
        self._participant_id: str = ""
        self._turn_count: int = 0
        self._state: str = SystemState.IDLE
        self._running: bool = False
        self._session_start_time: Optional[float] = None

        # Behavioral parameters (the RAWR parameter matrix)
        self._polar_level: int = self._avct.default_polar_level
        self._category: str = self._avct.default_category
        self._subtype: int = self._avct.default_subtype
        self._modifiers: list = []
        self._end_requested: bool = False

        self.on_state_change: Optional[Callable[[str], None]] = None
        self.on_event: Optional[Callable[[dict], None]] = None

    # --- properties ---------------------------------------------------------

    @property
    def state(self) -> str: return self._state
    @property
    def session_id(self) -> Optional[str]: return self._session_id
    @property
    def turn_count(self) -> int: return self._turn_count
    @property
    def is_running(self) -> bool: return self._running
    @property
    def polar_level(self) -> int: return self._polar_level
    @property
    def end_requested(self) -> bool: return self._end_requested
    @property
    def gate(self) -> OperatorGate: return self._gate
    @property
    def capabilities(self) -> dict:
        caps = self._robot.capabilities.as_dict()
        caps["fidelity"] = {
            "detector": bool(self._fidelity and self._fidelity.detector_enabled),
            "judge": bool(self._fidelity and self._fidelity.judge_enabled),
            "block_below": self._fidelity.block_below if self._fidelity else None,
        }
        caps["monitor"] = bool(self._monitor and self._monitor.enabled)
        return caps
    @property
    def elapsed_seconds(self) -> float:
        if self._session_start_time is None:
            return 0.0
        return time.monotonic() - self._session_start_time

    def settings(self) -> dict:
        return {
            "polar_level": self._polar_level, "category": self._category,
            "subtype": self._subtype, "modifiers": list(self._modifiers),
        }

    # --- operator controls ----------------------------------------------------

    def set_avct(self, polar_level: int, category: str, subtype: int, modifiers: list,
                 source: Optional[str] = None) -> None:
        """Set the behavioral parameters; they apply from the next generated response."""
        if "M1" in modifiers and "M6" in modifiers:
            log.warning("M1 (Interrupting) and M6 (Silent Treatment) are contradictory; applying M6")
            modifiers = [m for m in modifiers if m != "M1"]
        self._polar_level = max(-3, min(3, int(polar_level)))
        self._category = category
        self._subtype = max(1, min(3, int(subtype)))
        self._modifiers = list(modifiers)
        if source and self._session_id:
            self._logger.log_event(self._session_id, "settings_change", self._turn_count,
                                   payload={"source": source, **self.settings()})

    def operator_action(self, candidate_id: Optional[int], action: str) -> bool:
        """Apply an operator action (send, temper, intensify, regenerate, hold) to the pending response."""
        ok = self._gate.act(candidate_id, action)
        if ok and action == "hold":
            self._logger.log_event(self._session_id, "hold", self._turn_count, candidate_id)
            self._emit({"type": "candidate_blocked", "candidate_id": candidate_id, "reason": "held by operator"})
        return ok

    def set_review_policy(self, review_mode: Optional[str] = None, hold_seconds: Optional[float] = None) -> dict:
        """Change the release policy live; applies from the next candidate."""
        self._gate.set_policy(review_mode, hold_seconds)
        status = {"review_mode": self._gate.review_mode, "hold_seconds": self._gate.hold_seconds}
        self._logger.log_event(self._session_id, "review_policy", self._turn_count, payload=status)
        return status

    def stop_speech(self) -> bool:
        """Emergency stop: interrupt the robot's current utterance; the session continues."""
        stopped = self._robot.stop()
        self._logger.log_event(self._session_id, "stop_speech", self._turn_count,
                               payload={"stopped": stopped, "interrupt": self._robot.capabilities.interrupt})
        return stopped

    # --- internals -------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # the UI must never break a turn
                log.exception("on_event handler failed")

    def _is_current(self, session_id: Optional[str]) -> bool:
        """True while session_id is still the running session.

        A turn started in a previous session must not keep recording, or
        speak and log into the session that replaced it.
        """
        return self._running and self._session_id == session_id

    def _on_monitor(self, candidate_id: int, session_id: str, turn_number: int, result: MonitorResult) -> None:
        self._logger.update_candidate(candidate_id, monitor_json=result.as_dict())
        self._emit({"type": "monitor", "candidate_id": candidate_id, "scores": result.scores,
                    "rationale": result.rationale, "error": result.error, "latency_ms": result.latency_ms})
        reason = None
        if result.error and self._monitor.gate_auto_send:
            reason = "monitor unavailable"
        elif result.max_score >= 2:
            reason = "monitor: clear psychosocial risk"
        if reason and self._gate.escalate(candidate_id, reason):
            self._logger.log_event(session_id, "monitor_block", turn_number, candidate_id,
                                   {"scores": result.scores, "error": result.error})
            self._emit({"type": "candidate_blocked", "candidate_id": candidate_id, "reason": reason})
        self._gate.monitor_done(candidate_id)

    def _on_fidelity(self, candidate_id: int, session_id: str, turn_number: int, source: str, result: dict) -> None:
        self._logger.update_candidate(candidate_id, **{f"fidelity_{source}_json": result})
        self._emit({"type": "fidelity", "candidate_id": candidate_id, "source": source,
                    **{k: v for k, v in result.items() if k != "raw"}})
        if source == "judge":
            if not result.get("error"):
                self._judged[candidate_id] = result
            reason = self._fidelity.below_threshold(result)
            if reason and self._gate.escalate(candidate_id, reason):
                self._logger.log_event(session_id, "fidelity_block", turn_number, candidate_id,
                                       {k: v for k, v in result.items() if k != "raw"})
                self._emit({"type": "candidate_blocked", "candidate_id": candidate_id, "reason": reason})
            self._gate.signal_done(candidate_id, "judge")

    # --- session lifecycle ---------------------------------------------------------

    def start_session(self, polar_level: int, category: str, subtype: int, modifiers: list, participant_id: str) -> str:
        self._session_id = str(uuid.uuid4())[:8]
        self._end_requested = False
        self._judged = {}
        self.set_avct(polar_level, category, subtype, modifiers)
        self._participant_id = participant_id
        self._turn_count = 0
        self._history.clear()
        self._running = True
        self._session_start_time = time.monotonic()
        self._set_state(SystemState.IDLE)

        self._logger.create_session(
            session_id=self._session_id,
            participant_id=participant_id,
            polar_level=self._polar_level,
            category=self._category,
            subtype=self._subtype,
            modifiers=self._modifiers,
            config_snapshot=self._config_snapshot,
        )
        self._logger.log_event(self._session_id, "session_start", 0,
                               payload={**self.settings(), "capabilities": self.capabilities})
        return self._session_id

    def end_session(self, reason: str = "operator") -> dict:
        was_speaking = self._state == SystemState.SPEAKING
        self._running = False
        self._gate.end()
        if was_speaking:
            self._robot.stop()  # cut the robot off mid-utterance
        self._set_state(SystemState.IDLE)
        self._robot.on_idle()

        summary = {
            "session_id": self._session_id,
            "participant_id": self._participant_id,
            "total_turns": self._turn_count,
            "polar_level": self._polar_level,
            "duration_seconds": round(self.elapsed_seconds, 1),
        }
        if self._session_id:
            self._logger.log_event(self._session_id, "session_end", self._turn_count,
                                   payload={"reason": reason, "interrupted_speech": was_speaking})
            self._logger.end_session(self._session_id)
        return summary

    def stop(self) -> None:
        self._running = False
        self._gate.end()

    # --- one turn -------------------------------------------------------------------------

    def run_turn(self) -> Optional[TurnResult]:
        session_id = self._session_id
        latency: dict[str, int] = {}

        # 1-2. Capture and transcribe; keep listening if Whisper hears nothing
        while True:
            self._set_state(SystemState.LISTENING)
            self._robot.on_listening()
            t0 = time.monotonic()
            audio = self._capture.record_utterance(is_active=lambda: self._is_current(session_id))
            if audio is None:
                self._set_state(SystemState.IDLE)
                return None
            latency["vad_ms"] = round((time.monotonic() - t0) * 1000)

            self._set_state(SystemState.PROCESSING)
            t1 = time.monotonic()
            asr_result = self._asr.transcribe(audio)
            latency["asr_ms"] = round((time.monotonic() - t1) * 1000)
            if asr_result.text.strip():
                break
            log.info("Empty transcript, listening again")

        self._turn_count += 1
        turn_number = self._turn_count
        transcript = asr_result.text
        self._robot.on_thinking()
        self._emit({"type": "participant", "turn_number": turn_number, "transcript": transcript})

        distress = self._safety.check_participant(transcript)
        if distress:
            self._logger.log_event(session_id, "participant_distress", turn_number, payload={"cues": distress})
            self._emit({"type": "participant_distress", "turn_number": turn_number, "cues": distress})

        # Parameters in force when the participant finished speaking
        requested_polar = self._polar_level
        polar, category, subtype, modifiers = self._polar_level, self._category, self._subtype, list(self._modifiers)
        self._history.add_user_message(transcript)
        messages = self._history.get_messages()

        # 3-5. Generate, rate, and review until a response is released
        attempt, reason = 0, "initial"
        llm_ms_total, review_ms_total = 0, 0
        while True:
            attempt += 1
            system_prompt = self._avct.get_system_prompt(session_id, polar, category, subtype, modifiers)
            t2 = time.monotonic()
            try:
                llm_result = self._llm.generate(system_prompt, messages)
            except Exception as e:
                log.error("LLM error, using fallback line: %s", e)
                llm_result = LLMResult(text=FALLBACK_RESPONSE, model="fallback", total_tokens=0,
                                       generation_time_seconds=time.monotonic() - t2)
            llm_ms = round((time.monotonic() - t2) * 1000)
            llm_ms_total += llm_ms

            response_text, end_detected = extract_end_signal(llm_result.text)
            content = self._safety.check_response(response_text)
            cfg_risk = config_risk(polar, category)
            risk = max_risk(content.level, cfg_risk)
            llm_input = {"system_prompt": system_prompt, "messages": messages}

            candidate_id = self._logger.log_candidate({
                "session_id": session_id, "turn_number": turn_number, "attempt": attempt,
                "user_transcript": transcript, "requested_polar_level": requested_polar,
                "polar_level": polar, "category": category, "subtype": subtype, "modifiers": modifiers,
                "generation_reason": reason, "llm_input": llm_input, "llm_output_raw": llm_result.text,
                "llm_output": response_text, "llm_model": llm_result.model,
                "tokens_used": llm_result.total_tokens, "latency_llm_ms": llm_ms,
                "end_signal": end_detected, "content_risk": content.level,
                "content_flags": content.as_dict()["flags"], "config_risk": cfg_risk, "risk_rating": risk,
                "participant_distress": distress,
            }, reasoning=llm_result.reasoning)

            monitoring = self._monitor is not None and self._monitor.enabled
            scoring = self._fidelity is not None and self._fidelity.enabled and self._fidelity.applies(polar, category)
            awaiting = []
            if monitoring and self._monitor.gate_auto_send:
                awaiting.append("monitor")
            if scoring and self._fidelity.gates(polar, category):
                awaiting.append("judge")
            terms = self._gate.open(candidate_id, risk, bool(distress), await_signals=awaiting)
            self._logger.update_candidate(candidate_id, auto_release=int(terms["auto_release"]),
                                          blocked_reasons_json=terms["blocked_reasons"])
            if not self._is_current(session_id):
                self._gate.end()
            self._set_state(SystemState.REVIEW)
            self._emit({
                "type": "candidate", "candidate_id": candidate_id, "turn_number": turn_number,
                "attempt": attempt, "generation_reason": reason, "transcript": transcript,
                "response": response_text, "polar_level": polar, "requested_polar_level": requested_polar,
                "category": category, "subtype": subtype, "modifiers": modifiers,
                "content_risk": content.level, "content_flags": content.as_dict()["flags"],
                "config_risk": cfg_risk, "risk_rating": risk, "participant_distress": distress,
                "llm_ms": llm_ms, "end_signal": end_detected, **terms,
            })
            if monitoring:
                self._monitor.score_async(
                    transcript, response_text,
                    lambda r, cid=candidate_id: self._on_monitor(cid, session_id, turn_number, r),
                )
            if scoring:
                self._fidelity.score_async(
                    transcript, response_text, polar, category, subtype,
                    lambda src, r, cid=candidate_id: self._on_fidelity(cid, session_id, turn_number, src, r),
                )

            decision = self._gate.wait(lambda: self._is_current(session_id))
            review_ms_total += decision.wait_ms

            if decision.action in ("temper", "intensify", "regenerate"):
                disposition = {"temper": "tempered", "intensify": "intensified",
                               "regenerate": "regenerated"}[decision.action]
                new_polar = {"temper": max(-3, polar - 1), "intensify": min(3, polar + 1),
                             "regenerate": polar}[decision.action]
                self._logger.decide_candidate(candidate_id, disposition, decision.decided_by, decision.wait_ms)
                self._logger.log_event(session_id, decision.action, turn_number, candidate_id,
                                       {"from_polar": polar, "to_polar": new_polar})
                self._emit({"type": "candidate_replaced", "candidate_id": candidate_id,
                            "action": decision.action, "new_polar_level": new_polar})
                polar, reason = new_polar, decision.action
                self._set_state(SystemState.PROCESSING)
                continue

            if decision.action == "end" or not self._is_current(session_id):
                self._logger.decide_candidate(candidate_id, "withheld_session_ended", decision.decided_by,
                                              decision.wait_ms)
                return None

            disposition = "sent" if decision.action == "send" else "auto_sent"
            self._logger.decide_candidate(candidate_id, disposition, decision.decided_by, decision.wait_ms)
            self._logger.log_event(session_id, decision.action, turn_number, candidate_id,
                                   {"decided_by": decision.decided_by, "review_ms": decision.wait_ms})
            break

        latency["llm_ms"] = llm_ms_total
        latency["review_ms"] = review_ms_total
        if end_detected:
            # Ending the session is the operator's decision unless configured otherwise
            self._logger.log_event(session_id, "model_end_signal", turn_number, candidate_id,
                                   {"ends_session": self._model_can_end})
            if self._model_can_end:
                self._end_requested = True
            else:
                self._emit({"type": "end_suggested", "turn_number": turn_number, "candidate_id": candidate_id})

        # 6. Robot speech through the configured backend; the [END] token never reaches the robot
        self._set_state(SystemState.SPEAKING)
        self._emit({"type": "speaking", "candidate_id": candidate_id, "response": response_text})
        t3 = time.monotonic()
        judged = self._judged.pop(candidate_id, None) or {}
        cue = SpeechCue(polar, category, subtype, modifiers,
                        exhibited_category=judged.get("matched_category"),
                        exhibited_intensity=judged.get("intensity_est"))
        completed = self._robot.speak(response_text, cue)
        latency["speech_ms"] = round((time.monotonic() - t3) * 1000)
        latency["total_ms"] = round((time.monotonic() - t0) * 1000)
        self._history.add_assistant_message(response_text)

        turn_result = TurnResult(
            turn_number=turn_number,
            user_audio=audio if getattr(audio, "samples", np.empty(0)).size else None,
            transcript=transcript,
            llm_response=response_text,
            polar_level=polar,
            category=category,
            subtype=subtype,
            modifiers=modifiers,
            risk_rating=risk,
            latency=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
            requested_polar_level=requested_polar,
            content_risk=content.level,
            config_risk=cfg_risk,
            candidate_id=candidate_id,
            n_candidates=attempt,
            operator_action=disposition,
            decided_by=decision.decided_by,
            participant_distress=distress,
        )
        self._logger.log_turn(
            session_id=session_id, turn=turn_result, asr_result=asr_result,
            llm_model=llm_result.model, tokens_used=llm_result.total_tokens,
            llm_input=llm_input, speech_completed=completed,
        )

        if self._is_current(session_id):
            self._set_state(SystemState.IDLE)
        return turn_result
