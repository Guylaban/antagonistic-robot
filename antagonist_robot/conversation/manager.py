"""Conversation manager — orchestrates the turn-based conversation loop.

Wires together all pipeline components and runs sequential turns:
capture -> ASR -> LLM -> TTS -> audio output. Each step completes
before the next starts.
"""

import re
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from antagonist_robot.conversation.history import ConversationHistory
from antagonist_robot.conversation.avct_manager import AvctManager
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.nao.base import NAOAdapter
from antagonist_robot.pipeline.asr import ASREngine
from antagonist_robot.pipeline.audio_capture import AudioCapture
from antagonist_robot.pipeline.audio_output import AudioOutputBase, NAOAudioOutput
from antagonist_robot.pipeline.llm import LLMEngine
from antagonist_robot.pipeline.tts import TTSBase
from antagonist_robot.pipeline.types import TurnResult, LLMResult

_END_PATTERN = re.compile(r'\[end\]', re.IGNORECASE)

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
    SPEAKING = "speaking"

class ConversationManager:
    """Orchestrates the turn-based voice conversation loop."""
    def __init__(
        self,
        audio_capture: AudioCapture,
        asr: ASREngine,
        llm: LLMEngine,
        tts: TTSBase,
        audio_output: AudioOutputBase,
        avct_manager: AvctManager,
        session_logger: SessionLogger,
        nao_adapter: NAOAdapter,
    ):
        self._capture = audio_capture
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._output = audio_output
        self._avct = avct_manager
        self._logger = session_logger
        self._nao = nao_adapter

        self._history = ConversationHistory()
        self._session_id: Optional[str] = None
        self._participant_id: str = ""
        self._turn_count: int = 0
        self._state: str = SystemState.IDLE
        self._running: bool = False
        self._session_start_time: Optional[float] = None
        
        # AVCT Properties
        self._polar_level: int = self._avct.default_polar_level
        self._category: str = self._avct.default_category
        self._subtype: int = self._avct.default_subtype
        self._modifiers: list = []
        self._end_requested: bool = False

        self.on_state_change: Optional[Callable[[str], None]] = None

    @property
    def state(self) -> str: return self._state
    @property
    def session_id(self) -> Optional[str]: return self._session_id
    @property
    def turn_count(self) -> int: return self._turn_count
    @property
    def is_running(self) -> bool: return self._running
    @property
    def polar_level(self) -> int:
        return self._polar_level
    @property
    def end_requested(self) -> bool:
        return self._end_requested
    @property
    def elapsed_seconds(self) -> float:
        if self._session_start_time is None: return 0.0
        return time.monotonic() - self._session_start_time

    # Provide backwards compatibility getter/setter for older UI code
    @property
    def hostility_level(self) -> int: return self._polar_level
    @hostility_level.setter
    def hostility_level(self, value: int) -> None:
        self.set_avct(value, self._category, self._subtype, self._modifiers)

    def set_avct(self, polar_level: int, category: str, subtype: int, modifiers: list) -> None:
        self._polar_level = max(-3, min(3, polar_level))
        self._category = category
        self._subtype = subtype
        self._modifiers = modifiers

    def _set_state(self, state: str) -> None:
        self._state = state
        if self.on_state_change: self.on_state_change(state)

    def start_session(self, polar_level: int, category: str, subtype: int, modifiers: list, participant_id: str) -> str:
        self._session_id = str(uuid.uuid4())[:8]
        self._end_requested = False
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
            config_snapshot=None
        )
        return self._session_id

    def _is_current(self, session_id: Optional[str]) -> bool:
        """True while session_id is still the running session.

        A turn started in a previous session must not keep recording, or
        speak and log into the session that replaced it.
        """
        return self._running and self._session_id == session_id

    def run_turn(self) -> Optional[TurnResult]:
        session_id = self._session_id
        latency: dict[str, int] = {}

        # 1-2. Capture and transcribe; keep listening if Whisper hears nothing
        while True:
            self._set_state(SystemState.LISTENING)
            self._nao.on_listening()
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
            logging.getLogger(__name__).info("Empty transcript, listening again")

        self._turn_count += 1

        # 3. LLM Generate
        system_prompt = self._avct.get_system_prompt(
            session_id, self._polar_level, self._category, self._subtype, self._modifiers
        )
        self._history.add_user_message(asr_result.text)

        t2 = time.monotonic()
        try:
            llm_result = self._llm.generate(system_prompt, self._history.get_messages())
        except Exception as e:
            logging.getLogger(__name__).warning("LLM error: %s", e)
            llm_result = LLMResult(text="I see. Go on.", model="fallback", total_tokens=0, generation_time_seconds=time.monotonic() - t2)
        latency["llm_ms"] = round((time.monotonic() - t2) * 1000)

        # Snapshot the conversation history as sent to the LLM (before assistant response is added)
        conversation_history = self._history.get_messages()

        # 3b. Check for LLM end signal and clean the response text
        response_text, end_detected = extract_end_signal(llm_result.text)
        if end_detected:
            self._end_requested = True

        risk_rating = self._avct.get_risk_rating(self._polar_level, self._category, self._subtype, self._modifiers)

        if not self._is_current(session_id):
            return None

        # 4. Synthesize (using cleaned text — [END] token never reaches TTS)
        self._set_state(SystemState.SPEAKING)
        tts_result = None

        if isinstance(self._output, NAOAudioOutput) and self._output.use_builtin_tts:
            t3 = time.monotonic()
            self._output.speak_text(response_text)
            latency["tts_ms"] = round((time.monotonic() - t3) * 1000)
        else:
            t3 = time.monotonic()
            tts_result = self._tts.synthesize(response_text)
            latency["tts_ms"] = round((time.monotonic() - t3) * 1000)
            self._output.play_audio(tts_result)

        latency["total_ms"] = round((time.monotonic() - t0) * 1000)

        self._nao.on_response(response_text, self._polar_level)

        self._history.add_assistant_message(response_text)

        timestamp = datetime.now(timezone.utc).isoformat()
        turn_result = TurnResult(
            turn_number=self._turn_count,
            user_audio=audio,
            transcript=asr_result.text,
            llm_response=response_text,
            tts_result=tts_result,
            polar_level=self._polar_level,
            category=self._category,
            subtype=self._subtype,
            modifiers=self._modifiers,
            risk_rating=risk_rating,
            latency=latency,
            timestamp=timestamp,
        )

        self._logger.log_turn(
            session_id=session_id,
            turn=turn_result,
            asr_result=asr_result,
            llm_result=llm_result,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )

        self._set_state(SystemState.IDLE)
        return turn_result

    def end_session(self) -> dict:
        self._running = False
        self._set_state(SystemState.IDLE)
        self._nao.on_idle()

        summary = {
            "session_id": self._session_id,
            "participant_id": self._participant_id,
            "total_turns": self._turn_count,
            "polar_level": self._polar_level,
            "duration_seconds": round(self.elapsed_seconds, 1),
        }
        self._logger.end_session(self._session_id)
        return summary

    def stop(self) -> None:
        self._running = False
