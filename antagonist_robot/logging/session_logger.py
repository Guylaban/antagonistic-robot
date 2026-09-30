"""SQLite-based session logger with WAV file saving.

Tables:
    sessions          one row per session (participant, initial parameters, config snapshot)
    turns             one row per spoken robot turn (what the participant heard)
    candidates        one row per generated response, including every response the
                      operator tempered, regenerated, or withheld, with its full LLM
                      input, raw output, ratings, monitor scores, and disposition
    reasoning_traces  provider reasoning traces, keyed by candidate_id (kept apart
                      from the responses; excluded from exports unless asked for)
    operator_events   every operator action (parameter changes, send, temper,
                      regenerate, hold, review-policy changes, session end)

Participant audio is saved as WAV under data/audio/SESSION_ID/.
"""

import csv
import io
import json
import sqlite3
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from antagonist_robot.pipeline.types import ASRResult, TurnResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionLogger:
    """SQLite-based logger for research data collection."""

    def __init__(self, db_path: str, audio_dir: str, save_audio: bool = True):
        self._db_path = db_path
        self._audio_dir = audio_dir
        self._save_audio = save_audio

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()  # the conversation, monitor, and server threads all write
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they do not exist and migrate older databases."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                polar_level INTEGER NOT NULL,
                category TEXT NOT NULL,
                subtype INTEGER NOT NULL,
                modifiers_json TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                config_snapshot TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                user_audio_path TEXT,
                user_transcript TEXT,
                transcript_confidence REAL,
                llm_input TEXT,
                llm_output TEXT,
                llm_model TEXT,
                tokens_used INTEGER,
                tts_voice TEXT,
                tts_audio_path TEXT,

                hostility_level INTEGER,
                polar_level INTEGER,
                category TEXT,
                subtype INTEGER,
                modifiers_json TEXT,
                risk_rating TEXT,

                latency_vad_ms INTEGER,
                latency_asr_ms INTEGER,
                latency_llm_ms INTEGER,
                latency_tts_ms INTEGER,
                latency_total_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                turn_number INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                user_transcript TEXT,
                requested_polar_level INTEGER,
                polar_level INTEGER,
                category TEXT,
                subtype INTEGER,
                modifiers_json TEXT,
                generation_reason TEXT,
                llm_input TEXT,
                llm_output_raw TEXT,
                llm_output TEXT,
                llm_model TEXT,
                tokens_used INTEGER,
                latency_llm_ms INTEGER,
                end_signal INTEGER,
                content_risk TEXT,
                content_flags_json TEXT,
                config_risk TEXT,
                risk_rating TEXT,
                participant_distress_json TEXT,
                auto_release INTEGER,
                blocked_reasons_json TEXT,
                monitor_json TEXT,
                fidelity_detector_json TEXT,
                fidelity_judge_json TEXT,
                disposition TEXT,
                decided_by TEXT,
                decided_at TEXT,
                review_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS reasoning_traces (
                candidate_id INTEGER PRIMARY KEY REFERENCES candidates(candidate_id),
                session_id TEXT NOT NULL,
                reasoning TEXT
            );

            CREATE TABLE IF NOT EXISTS operator_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                turn_number INTEGER,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                candidate_id INTEGER,
                payload_json TEXT
            );
        """)

        migrations = [
            "ALTER TABLE sessions ADD COLUMN polar_level INTEGER DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN category TEXT DEFAULT 'D'",
            "ALTER TABLE sessions ADD COLUMN subtype INTEGER DEFAULT 1",
            "ALTER TABLE sessions ADD COLUMN modifiers_json TEXT DEFAULT '[]'",
            "ALTER TABLE sessions ADD COLUMN config_snapshot TEXT",
            "ALTER TABLE turns ADD COLUMN polar_level INTEGER DEFAULT 0",
            "ALTER TABLE turns ADD COLUMN category TEXT DEFAULT 'D'",
            "ALTER TABLE turns ADD COLUMN subtype INTEGER DEFAULT 1",
            "ALTER TABLE turns ADD COLUMN modifiers_json TEXT DEFAULT '[]'",
            "ALTER TABLE turns ADD COLUMN risk_rating TEXT DEFAULT 'UNKNOWN'",
            # operator console additions
            "ALTER TABLE turns ADD COLUMN requested_polar_level INTEGER",
            "ALTER TABLE turns ADD COLUMN content_risk TEXT",
            "ALTER TABLE turns ADD COLUMN config_risk TEXT",
            "ALTER TABLE turns ADD COLUMN candidate_id INTEGER",
            "ALTER TABLE turns ADD COLUMN n_candidates INTEGER",
            "ALTER TABLE turns ADD COLUMN operator_action TEXT",
            "ALTER TABLE turns ADD COLUMN decided_by TEXT",
            "ALTER TABLE turns ADD COLUMN participant_distress_json TEXT",
            "ALTER TABLE turns ADD COLUMN latency_review_ms INTEGER",
            "ALTER TABLE turns ADD COLUMN speech_completed INTEGER",
            # fidelity monitor
            "ALTER TABLE candidates ADD COLUMN fidelity_detector_json TEXT",
            "ALTER TABLE candidates ADD COLUMN fidelity_judge_json TEXT",
        ]
        for query in migrations:
            try:
                self._conn.execute(query)
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.commit()

    # --- sessions -----------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        participant_id: str,
        polar_level: int,
        category: str,
        subtype: int,
        modifiers: list,
        config_snapshot: Optional[dict] = None,
    ) -> None:
        """Create a new session record in the database."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, participant_id, polar_level, category, subtype, modifiers_json, "
                "start_time, config_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, participant_id, polar_level, category, subtype,
                    json.dumps(modifiers), _now(),
                    json.dumps(config_snapshot) if config_snapshot else None,
                ),
            )
            self._conn.commit()

        if self._save_audio:
            (Path(self._audio_dir) / session_id).mkdir(parents=True, exist_ok=True)

    def end_session(self, session_id: str) -> None:
        """Set the end time on a session record."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET end_time = ? WHERE session_id = ?", (_now(), session_id)
            )
            self._conn.commit()

    # --- candidates -----------------------------------------------------------

    def log_candidate(self, record: dict, reasoning: Optional[str] = None) -> int:
        """Insert a generated response as soon as it exists; return its candidate_id.

        Written before the operator decides, so a response that is never
        spoken (tempered, regenerated, withheld at session end, or lost to a
        crash) is still on disk.
        """
        cols = [
            "session_id", "turn_number", "attempt", "user_transcript", "requested_polar_level",
            "polar_level", "category", "subtype", "generation_reason", "llm_output_raw",
            "llm_output", "llm_model", "tokens_used", "latency_llm_ms", "content_risk",
            "config_risk", "risk_rating", "auto_release",
        ]
        values = [record.get(c) for c in cols]
        cols += ["created_at", "modifiers_json", "llm_input", "end_signal", "content_flags_json",
                 "participant_distress_json", "blocked_reasons_json", "disposition"]
        values += [
            _now(), json.dumps(record.get("modifiers", [])), json.dumps(record.get("llm_input")),
            int(bool(record.get("end_signal"))), json.dumps(record.get("content_flags", [])),
            json.dumps(record.get("participant_distress", [])),
            json.dumps(record.get("blocked_reasons", [])), "pending",
        ]
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO candidates ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", values
            )
            candidate_id = cur.lastrowid
            if reasoning:
                self._conn.execute(
                    "INSERT INTO reasoning_traces (candidate_id, session_id, reasoning) VALUES (?, ?, ?)",
                    (candidate_id, record.get("session_id"), reasoning),
                )
            self._conn.commit()
        return candidate_id

    def update_candidate(self, candidate_id: int, **fields) -> None:
        """Update columns of a candidate row (e.g. release terms, disposition)."""
        if not fields:
            return
        encoded = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in fields.items()}
        with self._lock:
            self._conn.execute(
                f"UPDATE candidates SET {', '.join(f'{k} = ?' for k in encoded)} WHERE candidate_id = ?",
                list(encoded.values()) + [candidate_id],
            )
            self._conn.commit()

    def decide_candidate(self, candidate_id: int, disposition: str, decided_by: str, review_ms: int) -> None:
        """Record what happened to a candidate."""
        self.update_candidate(
            candidate_id, disposition=disposition, decided_by=decided_by,
            decided_at=_now(), review_ms=review_ms,
        )

    # --- operator events -----------------------------------------------------

    def log_event(
        self,
        session_id: Optional[str],
        event: str,
        turn_number: Optional[int] = None,
        candidate_id: Optional[int] = None,
        payload: Optional[dict] = None,
    ) -> None:
        """Record an operator action or system event."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO operator_events (session_id, turn_number, timestamp, event, candidate_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_number, _now(), event, candidate_id, json.dumps(payload or {})),
            )
            self._conn.commit()

    # --- turns ------------------------------------------------------------------

    def log_turn(
        self,
        session_id: str,
        turn: TurnResult,
        asr_result: ASRResult,
        llm_model: str,
        tokens_used: int,
        llm_input: dict,
        speech_completed: bool = True,
    ) -> None:
        """Log a spoken turn and save the participant's audio."""
        user_audio_path = None
        if self._save_audio and turn.user_audio is not None:
            user_audio_path = str(Path(self._audio_dir) / session_id / f"turn_{turn.turn_number:03d}_user.wav")
            self._save_wav(user_audio_path, turn.user_audio.samples, turn.user_audio.sample_rate)

        row = {
            "session_id": session_id, "turn_number": turn.turn_number, "timestamp": turn.timestamp,
            "user_audio_path": user_audio_path, "user_transcript": turn.transcript,
            "transcript_confidence": asr_result.confidence, "llm_input": json.dumps(llm_input),
            "llm_output": turn.llm_response, "llm_model": llm_model, "tokens_used": tokens_used,
            "polar_level": turn.polar_level, "requested_polar_level": turn.requested_polar_level,
            "category": turn.category, "subtype": turn.subtype, "modifiers_json": json.dumps(turn.modifiers),
            "risk_rating": turn.risk_rating, "content_risk": turn.content_risk, "config_risk": turn.config_risk,
            "candidate_id": turn.candidate_id, "n_candidates": turn.n_candidates,
            "operator_action": turn.operator_action, "decided_by": turn.decided_by,
            "participant_distress_json": json.dumps(turn.participant_distress),
            "latency_vad_ms": turn.latency.get("vad_ms"), "latency_asr_ms": turn.latency.get("asr_ms"),
            "latency_llm_ms": turn.latency.get("llm_ms"), "latency_review_ms": turn.latency.get("review_ms"),
            # latency_tts_ms keeps its column name for older databases; it now holds the
            # time the robot spent speaking (built-in NAOqi TTS, text to acknowledgement).
            "latency_tts_ms": turn.latency.get("speech_ms"), "latency_total_ms": turn.latency.get("total_ms"),
            "speech_completed": int(bool(speech_completed)),
        }
        with self._lock:
            self._conn.execute(
                f"INSERT INTO turns ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                list(row.values()),
            )
            self._conn.commit()

    # --- reading and export ----------------------------------------------------

    def get_sessions(self) -> list:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM sessions ORDER BY start_time DESC")
            return [dict(row) for row in cursor.fetchall()]

    def _rows(self, query: str, args: tuple) -> list:
        with self._lock:
            return [dict(r) for r in self._conn.execute(query, args).fetchall()]

    def export_session(self, session_id: str, include_reasoning: bool = False) -> dict:
        """Session record with its turns, all candidates, and operator events."""
        sessions = self._rows("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        if not sessions:
            return {"session": None, "turns": [], "candidates": [], "operator_events": []}
        data = {
            "session": sessions[0],
            "turns": self._rows("SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number", (session_id,)),
            "candidates": self._rows(
                "SELECT * FROM candidates WHERE session_id = ? ORDER BY candidate_id", (session_id,)
            ),
            "operator_events": self._rows(
                "SELECT * FROM operator_events WHERE session_id = ? ORDER BY event_id", (session_id,)
            ),
        }
        if include_reasoning:
            data["reasoning_traces"] = self._rows(
                "SELECT * FROM reasoning_traces WHERE session_id = ? ORDER BY candidate_id", (session_id,)
            )
        return data

    CSV_COLUMNS = [
        "session_id", "participant_id", "turn_number", "attempt", "candidate_id", "created_at",
        "user_transcript", "requested_polar_level", "polar_level", "category", "subtype", "modifiers",
        "generation_reason", "llm_output", "llm_model", "tokens_used", "latency_llm_ms",
        "content_risk", "content_flags", "config_risk", "risk_rating", "participant_distress",
        "auto_release", "blocked_reasons", "monitor_privacy", "monitor_discrimination",
        "monitor_manipulation", "monitor_psych_harm", "monitor_insulting", "monitor_rationale",
        "judge_fidelity", "judge_category", "judge_intensity", "judge_refused", "judge_rationale",
        "detector_p_faithful",
        "disposition", "decided_by", "review_ms", "spoken",
    ]

    def export_csv(self, session_ids: Optional[list] = None) -> str:
        """One tidy CSV row per generated candidate (spoken or not), across sessions."""
        where, args = "", ()
        if session_ids:
            where = f"WHERE c.session_id IN ({', '.join('?' * len(session_ids))})"
            args = tuple(session_ids)
        rows = self._rows(
            "SELECT c.*, s.participant_id FROM candidates c JOIN sessions s USING (session_id) "
            f"{where} ORDER BY c.session_id, c.candidate_id",
            args,
        )
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=self.CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            monitor = json.loads(r.get("monitor_json") or "{}") or {}
            judge = json.loads(r.get("fidelity_judge_json") or "{}") or {}
            detector = json.loads(r.get("fidelity_detector_json") or "{}") or {}
            scores = monitor.get("scores") or {}
            flags = json.loads(r.get("content_flags_json") or "[]")
            r.update({
                "modifiers": " ".join(json.loads(r.get("modifiers_json") or "[]")),
                "content_flags": "; ".join(f"{f['category']}:{f['match']}" for f in flags),
                "participant_distress": "; ".join(json.loads(r.get("participant_distress_json") or "[]")),
                "blocked_reasons": "; ".join(json.loads(r.get("blocked_reasons_json") or "[]")),
                "monitor_rationale": monitor.get("rationale", ""),
                "spoken": int(r.get("disposition") in ("sent", "auto_sent")),
                "judge_fidelity": judge.get("fidelity"), "judge_category": judge.get("matched_category"),
                "judge_intensity": judge.get("intensity_est"), "judge_refused": judge.get("refused"),
                "judge_rationale": judge.get("rationale"), "detector_p_faithful": detector.get("p_faithful"),
                **{f"monitor_{k}": scores.get(k) for k in
                   ("privacy", "discrimination", "manipulation", "psych_harm", "insulting")},
            })
            writer.writerow(r)
        return out.getvalue()

    def _save_wav(self, path: str, samples: np.ndarray, sample_rate: int) -> None:
        int16_samples = (samples * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_samples.tobytes())

    def close(self) -> None:
        self._conn.close()
