"""Operator console API end to end: start, review action, live settings, export, stop."""

from fastapi.testclient import TestClient

from antagonist_robot.ui.server import create_app
from conftest import wait_for


def test_console_session_flow(make_manager):
    manager, llm, logger, _ = make_manager(["I think we should split the work equally.", "It seems fair."],
                                           ["That is simplistic.", "Fair is not effective."], review_mode="manual")
    with TestClient(create_app(manager, logger)) as client:
        assert "RAWR" in client.get("/").text

        sid = client.post("/api/session/start", json={
            "participant_id": "P9", "polar_level": 2, "category": "D", "subtype": 2, "modifiers": ["M2", "M4"],
        }).json()["session_id"]

        pending = wait_for(lambda: client.get("/api/operator/pending").json()["pending_candidate_id"])
        assert client.post("/api/operator/action", json={"action": "send", "candidate_id": 999}).status_code == 409
        assert client.post("/api/operator/action", json={"action": "temper", "candidate_id": pending}).json()["ok"]

        # live parameter change applies to the next generated response
        assert client.post("/api/settings", json={"category": "C"}).json()["category"] == "C"
        second = wait_for(lambda: (lambda p: p if p and p != pending else None)(
            client.get("/api/operator/pending").json()["pending_candidate_id"]))
        client.post("/api/operator/action", json={"action": "send", "candidate_id": second})
        wait_for(lambda: client.get("/api/status").json()["turn_count"] == 2)

        assert client.post("/api/operator/policy", json={"review_mode": "timed", "hold_seconds": 1}).json() == {
            "review_mode": "timed", "hold_seconds": 1.0}
        assert client.post("/api/session/stop").json()["session_id"] == sid

        export = client.get(f"/api/sessions/{sid}/export").json()
        assert [c["disposition"] for c in export["candidates"]][:2] == ["tempered", "sent"]
        events = [e["event"] for e in export["operator_events"]]
        for expected in ("session_start", "temper", "settings_change", "send", "review_policy", "session_end"):
            assert expected in events
        csv_text = client.get(f"/api/sessions/{sid}/export.csv").text
        assert csv_text.splitlines()[0].startswith("session_id,participant_id")
