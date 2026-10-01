"""FastAPI server for the operator console: REST API + WebSocket.

Serves the operator console (ui/static/index.html) and provides endpoints
for session control, live parameter changes, operator review actions,
emergency speech stop, and data export. The conversation loop runs in a
background thread; its events are pushed to the console over WebSocket.
"""

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from antagonist_robot.conversation.manager import ConversationManager
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.logging.study_report import build_report, render_markdown

logger = logging.getLogger(__name__)

CONSOLE = Path(__file__).parent / "static" / "index.html"


class SessionStartRequest(BaseModel):
    """Request body for POST /api/session/start."""
    participant_id: str
    polar_level: int = 0
    category: str = "D"
    subtype: int = 1
    modifiers: list = []


class SettingsUpdateRequest(BaseModel):
    """Request body for POST /api/settings."""
    polar_level: Optional[int] = None
    category: Optional[str] = None
    subtype: Optional[int] = None
    modifiers: Optional[list] = None


class OperatorActionRequest(BaseModel):
    """Request body for POST /api/operator/action."""
    action: str                      # send | temper | intensify | regenerate | hold
    candidate_id: Optional[int] = None


class ReviewPolicyRequest(BaseModel):
    """Request body for POST /api/operator/policy."""
    review_mode: Optional[str] = None    # timed | manual
    hold_seconds: Optional[float] = None


class WebSocketManager:
    """Manages WebSocket connections and thread-safe broadcasting."""

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add(self, ws: WebSocket) -> None:
        with self._lock:
            self._clients.append(ws)

    def remove(self, ws: WebSocket) -> None:
        with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    def broadcast(self, event: dict) -> None:
        """Push an event to all clients. Safe to call from any thread."""
        message = json.dumps(event)
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)
            except Exception:
                self.remove(ws)


def create_app(manager: ConversationManager, session_logger: SessionLogger) -> FastAPI:
    """Create the FastAPI app with injected dependencies."""
    app = FastAPI(title="RAWR operator console")
    ws_manager = WebSocketManager()

    _conversation_thread: dict = {"thread": None, "generation": 0}
    _thread_lock = threading.Lock()

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    async def on_startup():
        ws_manager.set_event_loop(asyncio.get_event_loop())

    def on_state_change(state: str):
        ws_manager.broadcast({
            "type": "state_change", "state": state, "turn_count": manager.turn_count,
            "elapsed_seconds": round(manager.elapsed_seconds, 1),
        })

    manager.on_state_change = on_state_change
    manager.on_event = ws_manager.broadcast

    @app.get("/")
    async def serve_console():
        return FileResponse(str(CONSOLE))

    # --- status and settings -------------------------------------------------

    @app.get("/api/status")
    async def get_status():
        return {
            "state": manager.state,
            "session_id": manager.session_id,
            "is_running": manager.is_running,
            "turn_count": manager.turn_count,
            "elapsed_seconds": round(manager.elapsed_seconds, 1),
            **manager.settings(),
            "review": manager.gate.status(),
            "capabilities": manager.capabilities,
        }

    @app.get("/api/settings")
    async def get_settings():
        return manager.settings()

    @app.post("/api/settings")
    async def update_settings(req: SettingsUpdateRequest):
        """Change behavioral parameters; they apply from the next generated response."""
        cur = manager.settings()
        manager.set_avct(
            req.polar_level if req.polar_level is not None else cur["polar_level"],
            req.category if req.category is not None else cur["category"],
            req.subtype if req.subtype is not None else cur["subtype"],
            req.modifiers if req.modifiers is not None else cur["modifiers"],
            source="operator",
        )
        ws_manager.broadcast({"type": "settings", **manager.settings()})   # keeps every open console in sync
        return manager.settings()

    # --- operator review --------------------------------------------------------

    @app.get("/api/operator/pending")
    async def get_pending():
        return manager.gate.status()

    @app.post("/api/operator/action")
    async def operator_action(req: OperatorActionRequest):
        if req.action not in ("send", "temper", "intensify", "regenerate", "hold"):
            return JSONResponse(status_code=400, content={"error": f"unknown action {req.action!r}"})
        ok = manager.operator_action(req.candidate_id, req.action)
        if not ok:
            return JSONResponse(status_code=409, content={"error": "no pending response with that id"})
        return {"ok": True, "action": req.action, "candidate_id": req.candidate_id}

    @app.post("/api/operator/policy")
    async def review_policy(req: ReviewPolicyRequest):
        if req.review_mode is not None and req.review_mode not in ("timed", "manual"):
            return JSONResponse(status_code=400, content={"error": "review_mode must be timed or manual"})
        return manager.set_review_policy(req.review_mode, req.hold_seconds)

    @app.post("/api/robot/stop")
    async def stop_robot_speech():
        """Emergency stop: interrupt the robot's current utterance."""
        stopped = await asyncio.to_thread(manager.stop_speech)
        ws_manager.broadcast({"type": "speech_stopped", "stopped": stopped})
        return {"stopped": stopped}

    # --- session control -------------------------------------------------------

    @app.post("/api/session/start")
    async def start_session(req: SessionStartRequest):
        """Start a session and run the conversation loop in a background thread."""
        with _thread_lock:
            if manager.is_running:
                manager.end_session(reason="superseded")

            old_thread = _conversation_thread["thread"]
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=0.5)

            _conversation_thread["generation"] += 1
            my_generation = _conversation_thread["generation"]

            session_id = manager.start_session(
                req.polar_level, req.category, req.subtype, req.modifiers, req.participant_id
            )
            ws_manager.broadcast({"type": "session_started", "session_id": session_id,
                                  "participant_id": req.participant_id, **manager.settings()})

            def conversation_loop():
                consecutive_errors = 0
                while manager.is_running:
                    if _conversation_thread["generation"] != my_generation:
                        return
                    try:
                        turn_result = manager.run_turn()
                        if turn_result is None or _conversation_thread["generation"] != my_generation:
                            return
                        consecutive_errors = 0
                        ws_manager.broadcast({
                            "type": "turn_complete",
                            "turn_number": turn_result.turn_number,
                            "candidate_id": turn_result.candidate_id,
                            "transcript": turn_result.transcript,
                            "response": turn_result.llm_response,
                            "polar_level": turn_result.polar_level,
                            "requested_polar_level": turn_result.requested_polar_level,
                            "category": turn_result.category,
                            "subtype": turn_result.subtype,
                            "modifiers": turn_result.modifiers,
                            "risk_rating": turn_result.risk_rating,
                            "operator_action": turn_result.operator_action,
                            "n_candidates": turn_result.n_candidates,
                            "latency": turn_result.latency,
                            "timestamp": turn_result.timestamp,
                        })
                        if manager.end_requested:
                            summary = manager.end_session(reason="robot_initiated")
                            ws_manager.broadcast({"type": "session_ended", "reason": "robot_initiated", **summary})
                            break
                    except Exception as e:
                        if _conversation_thread["generation"] != my_generation:
                            return
                        consecutive_errors += 1
                        logger.error("Conversation loop error: %s", e, exc_info=True)
                        ws_manager.broadcast({"type": "error", "message": str(e)})
                        if consecutive_errors >= 3:
                            summary = manager.end_session(reason="errors")
                            ws_manager.broadcast({"type": "session_ended", "reason": "Too many consecutive errors",
                                                  **summary})
                            break

            thread = threading.Thread(target=conversation_loop, daemon=True, name="conversation-loop")
            _conversation_thread["thread"] = thread
            thread.start()

        return {"session_id": session_id, "status": "started"}

    @app.post("/api/session/stop")
    async def stop_session():
        """End the current session; interrupts robot speech and withholds any pending response."""
        with _thread_lock:
            _conversation_thread["generation"] += 1
            if not manager.is_running:
                manager.stop()
                return {"status": "no active session"}
            summary = await asyncio.to_thread(manager.end_session, "operator")
            ws_manager.broadcast({"type": "session_ended", "reason": "operator", **summary})
            return summary

    # --- data --------------------------------------------------------------------

    @app.get("/api/sessions")
    async def list_sessions():
        return session_logger.get_sessions()

    @app.get("/api/sessions/{session_id}/export")
    async def export_session(session_id: str):
        """Session as JSON: session, turns, all candidates, operator events (no reasoning traces)."""
        return JSONResponse(
            session_logger.export_session(session_id),
            headers={"Content-Disposition": f"attachment; filename={session_id}.json"},
        )

    @app.get("/api/sessions/{session_id}/export.csv")
    async def export_session_csv(session_id: str):
        """One CSV row per generated response (spoken or not)."""
        return Response(
            session_logger.export_csv([session_id]), media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={session_id}.csv"},
        )

    @app.get("/api/export.csv")
    async def export_all_csv():
        """Merged CSV of every session in the database."""
        return Response(
            session_logger.export_csv(), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=rawr_all_sessions.csv"},
        )

    @app.get("/api/report.json")
    async def study_report_json():
        """Study report over every session in the database (see docs/reporting_template.md)."""
        report = await asyncio.to_thread(build_report, session_logger.db_path)
        return JSONResponse(report, headers={"Content-Disposition": "attachment; filename=rawr_study_report.json"})

    @app.get("/api/report.md")
    async def study_report_md():
        report = await asyncio.to_thread(build_report, session_logger.db_path)
        return Response(render_markdown(report), media_type="text/markdown",
                        headers={"Content-Disposition": "attachment; filename=rawr_study_report.md"})

    # --- WebSocket -------------------------------------------------------------------

    @app.websocket("/ws/conversation")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        ws_manager.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            ws_manager.remove(websocket)

    return app
