"""FastAPI server with REST API + WebSocket for real-time updates.

Serves the static frontend and provides endpoints for session control,
settings management, and real-time conversation updates via WebSocket.
"""

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from antagonist_robot.conversation.manager import ConversationManager
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.pipeline.tts import TTSBase

logger = logging.getLogger(__name__)


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
    tts_voice: Optional[str] = None


class WebSocketManager:
    """Manages WebSocket connections and broadcasting.

    Thread-safe: broadcasts can be called from the conversation
    background thread, and messages are dispatched via the asyncio
    event loop.
    """

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for cross-thread broadcasting."""
        self._loop = loop

    def add(self, ws: WebSocket) -> None:
        """Register a WebSocket client."""
        with self._lock:
            self._clients.append(ws)

    def remove(self, ws: WebSocket) -> None:
        """Unregister a WebSocket client."""
        with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    def broadcast(self, event: dict) -> None:
        """Push an event to all connected WebSocket clients.

        Safe to call from any thread. Uses the stored event loop
        to schedule the async send.
        """
        message = json.dumps(event)
        with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(message), self._loop
                    )
                else:
                    asyncio.run(ws.send_text(message))
            except Exception:
                self.remove(ws)


def create_app(
    manager: ConversationManager,
    tts_engine: TTSBase,
    session_logger: SessionLogger,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    """Factory function that creates the FastAPI app with injected dependencies."""
    app = FastAPI(title="Antagonistic Robot")
    ws_manager = WebSocketManager()

    # Track the conversation thread so we can prevent duplicates
    _conversation_thread: dict = {"thread": None, "generation": 0}
    _thread_lock = threading.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def on_startup():
        ws_manager.set_event_loop(asyncio.get_event_loop())

    if static_dir and (static_dir / "index.html").exists():
        logger.info("Serving React UI from %s", static_dir)
    else:
        logger.info("React build not found; serving fallback UI (build with: cd webui && npm run build)")

    if static_dir and (static_dir / "static").exists():
        app.mount("/static", StaticFiles(directory=str(static_dir / "static")), name="static")

    # --- Static files ---

    fallback_index = Path(__file__).parent / "static" / "index.html"

    @app.get("/")
    async def serve_index():
        """Serve the React build if present, otherwise the bundled fallback UI."""
        if static_dir:
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
        if fallback_index.exists():
            return FileResponse(str(fallback_index))
        return JSONResponse(
            {"error": "Frontend not found"},
            status_code=404,
        )

    # --- REST API ---

    @app.get("/api/status")
    async def get_status():
        """Return the current system state."""
        return {
            "state": manager.state,
            "session_id": manager.session_id,
            "turn_count": manager.turn_count,
            "elapsed_seconds": round(manager.elapsed_seconds, 1),
            "polar_level": manager.polar_level,
        }

    @app.post("/api/session/start")
    async def start_session(req: SessionStartRequest):
        """Start a new conversation session.

        Launches the conversation loop in a background thread.
        """
        with _thread_lock:
            # End any previous session so its end_time is recorded
            if manager.is_running:
                manager.end_session()

            # Wait briefly for old thread to notice the stop
            old_thread = _conversation_thread["thread"]
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=0.5)

            # Bump the generation counter — old threads check this to self-terminate
            _conversation_thread["generation"] += 1
            my_generation = _conversation_thread["generation"]

            session_id = manager.start_session(
                req.polar_level, req.category, req.subtype, req.modifiers, req.participant_id
            )

            # Set up state change callback for WebSocket broadcasting
            def on_state_change(state: str):
                ws_manager.broadcast({
                    "type": "state_change",
                    "state": state,
                    "turn_count": manager.turn_count,
                    "elapsed_seconds": round(manager.elapsed_seconds, 1),
                })

            manager.on_state_change = on_state_change

            # Run conversation loop in background thread
            def conversation_loop():
                consecutive_errors = 0
                while manager.is_running:
                    # Check if a newer session has started — if so, exit silently
                    if _conversation_thread["generation"] != my_generation:
                        logger.info("Old conversation thread exiting (superseded)")
                        return

                    try:
                        turn_result = manager.run_turn()
                        if turn_result is None:
                            logger.info("Old thread aborted mid-turn")
                            return

                        # Check generation again after the blocking turn completes
                        if _conversation_thread["generation"] != my_generation:
                            logger.info("Old conversation thread exiting after turn (superseded)")
                            return

                        consecutive_errors = 0
                        ws_manager.broadcast({
                            "type": "turn_complete",
                            "turn_number": turn_result.turn_number,
                            "transcript": turn_result.transcript,
                            "response": turn_result.llm_response,
                            "polar_level": turn_result.polar_level,
                            "category": turn_result.category,
                            "subtype": turn_result.subtype,
                            "risk_rating": turn_result.risk_rating,
                            "latency": turn_result.latency,
                            "timestamp": turn_result.timestamp,
                        })

                        # Check if the LLM requested session end
                        if manager.end_requested:
                            logger.info("LLM requested session end via [END] token")
                            summary = manager.end_session()
                            ws_manager.broadcast({
                                "type": "session_ended",
                                "reason": "robot_initiated",
                                **summary,
                            })
                            break
                    except Exception as e:
                        if _conversation_thread["generation"] != my_generation:
                            return
                        consecutive_errors += 1
                        logger.error("Conversation loop error: %s", e, exc_info=True)
                        ws_manager.broadcast({
                            "type": "error",
                            "message": str(e),
                        })
                        if consecutive_errors >= 3:
                            logger.error("Too many consecutive errors, stopping session")
                            manager.stop()
                            ws_manager.broadcast({
                                "type": "session_ended",
                                "reason": "Too many consecutive errors",
                            })
                            break

            thread = threading.Thread(
                target=conversation_loop, daemon=True, name="conversation-loop"
            )
            _conversation_thread["thread"] = thread
            thread.start()

        return {"session_id": session_id, "status": "started"}

    @app.post("/api/session/stop")
    async def stop_session():
        """End the current session."""
        with _thread_lock:
            # Bump generation so old thread self-terminates
            _conversation_thread["generation"] += 1

            if not manager.is_running:
                manager.stop()
                return {"status": "no active session"}

            summary = manager.end_session()
            manager.stop()
            ws_manager.broadcast({"type": "session_ended", **summary})
            return summary

    @app.get("/api/session/current")
    async def get_current_session():
        """Return current session info."""
        return {
            "session_id": manager.session_id,
            "turn_count": manager.turn_count,
            "elapsed_seconds": round(manager.elapsed_seconds, 1),
            "polar_level": manager.polar_level,
            "is_running": manager.is_running,
        }

    @app.get("/api/settings")
    async def get_settings():
        """Return current settings."""
        return {
            "polar_level": getattr(manager, "_polar_level", 0),
            "category": getattr(manager, "_category", "D"),
            "subtype": getattr(manager, "_subtype", 1),
            "modifiers": getattr(manager, "_modifiers", []),
            "tts_voice": "onyx"
        }

    @app.post("/api/settings")
    async def update_settings(req: SettingsUpdateRequest):
        """Update runtime settings."""
        try:
            if req.polar_level is not None or req.category is not None or req.subtype is not None or req.modifiers is not None:
                curr_polar = req.polar_level if req.polar_level is not None else manager._polar_level
                curr_cat = req.category if req.category is not None else manager._category
                curr_sub = req.subtype if req.subtype is not None else manager._subtype
                curr_mod = req.modifiers if req.modifiers is not None else manager._modifiers
                manager.set_avct(curr_polar, curr_cat, curr_sub, curr_mod)
            return {"polar_level": manager._polar_level, "category": manager._category, "subtype": manager._subtype, "modifiers": manager._modifiers}
        except Exception as e:
            import traceback
            return JSONResponse(status_code=500, content={"error": traceback.format_exc()})

    @app.get("/api/voices")
    async def get_voices():
        """Return the list of available TTS voices."""
        voices = tts_engine.list_voices()
        return [
            {"name": v.name, "gender": v.gender, "locale": v.locale}
            for v in voices
        ]

    @app.get("/api/sessions")
    async def list_sessions():
        """List all past sessions from the database."""
        return session_logger.get_sessions()

    @app.get("/api/sessions/{session_id}/export")
    async def export_session(session_id: str):
        """Export session data as a JSON download."""
        data = session_logger.export_session(session_id)
        return JSONResponse(
            data,
            headers={
                "Content-Disposition": f"attachment; filename={session_id}.json"
            },
        )

    # --- WebSocket ---

    @app.websocket("/ws/conversation")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time conversation updates."""
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
