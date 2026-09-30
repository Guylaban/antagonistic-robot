"""RAWR (Robotic Antagonism Workbench for Research): main entry point.

Initializes all components, verifies the robot's speaker server, and starts
the operator console (or a terminal console with --no-ui).

Usage:
    python main.py                          # operator console on http://localhost:8000
    python main.py --no-ui                  # terminal console
    python main.py --config my.yaml         # custom config file
    python main.py --nao-ip 127.0.0.1       # override nao.ip (e.g. tools/mock_nao.py)
    python main.py --script examples/demo_script.yaml
                                            # scripted participant instead of mic + ASR
"""

import argparse
import dataclasses
import logging
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    """Parse arguments, load config, initialize all components, and start."""
    parser = argparse.ArgumentParser(
        description="RAWR: operator-controlled antagonistic robot behavior for HRI research"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file (default: config.yaml)")
    parser.add_argument("--no-ui", action="store_true", help="Run a terminal console instead of the web console")
    parser.add_argument("--nao-ip", help="Override nao.ip from the config")
    parser.add_argument("--script", help="YAML file of participant utterances; replaces microphone and ASR")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    from antagonist_robot.config.settings import load_config
    config = load_config(args.config)
    if args.nao_ip:
        config.nao.ip = args.nao_ip

    print("=" * 58)
    print("  RAWR: Robotic Antagonism Workbench for Research")
    print("=" * 58)

    # Participant input: microphone + ASR, or a scripted participant
    if args.script:
        from antagonist_robot.pipeline.scripted_input import ScriptedParticipant, load_script
        participant = ScriptedParticipant(load_script(args.script))
        capture = asr = participant
        print(f"  Input: scripted participant ({args.script})")
    else:
        from antagonist_robot.pipeline.audio_capture import AudioCapture
        from antagonist_robot.pipeline.asr import ASREngine
        print(f"  Loading ASR model ({config.asr.model_size})...")
        capture = AudioCapture(config.audio)
        asr = ASREngine(config.asr)

    from antagonist_robot.pipeline.llm import LLMEngine
    print(f"  LLM: {config.llm.provider_name} ({config.llm.model})")
    llm = LLMEngine(config.llm)

    # Robot: speech is produced by the robot's built-in TTS via nao_speaker_server.py
    from antagonist_robot.nao.real import RealNAO
    from antagonist_robot.pipeline.audio_output import NAOAudioOutput

    print(f"  Robot: {config.nao.ip}:{config.nao.port}")
    audio_output = NAOAudioOutput(ip=config.nao.ip, port=config.nao.port)
    nao_adapter = RealNAO(config.nao.ip, config.nao.port, config.nao.naoqi_port, config.nao.password)
    nao_adapter.connect()
    if not nao_adapter.is_connected():
        print(
            f"\n  ERROR: NAO speaker server not reachable at {config.nao.ip}:{config.nao.port}.\n"
            f"  Run: python deploy_nao.py   (starts the speaker server on the robot)\n"
            f"  If that fails, see 'Troubleshooting the NAO connection' in README.md."
        )
        sys.exit(1)

    from antagonist_robot.logging.session_logger import SessionLogger
    session_logger = SessionLogger(
        db_path=config.logging.db_path, audio_dir=config.logging.audio_dir, save_audio=config.logging.save_audio,
    )

    from antagonist_robot.conversation.avct_manager import AvctManager
    from antagonist_robot.conversation.manager import ConversationManager
    from antagonist_robot.conversation.monitor import PsychosocialMonitor
    from antagonist_robot.conversation.operator import OperatorGate
    from antagonist_robot.conversation.safety import SafetyChecker

    gate = OperatorGate(config.operator)
    monitor = PsychosocialMonitor(config.monitor)
    print(f"  Review: {config.operator.review_mode}, hold {config.operator.hold_seconds}s, "
          f"explicit Send required at {config.operator.block_auto_send_at}+")
    print(f"  Psychosocial monitor: {'on (' + config.monitor.model + ')' if monitor.enabled else 'off'}")

    manager = ConversationManager(
        audio_capture=capture,
        asr=asr,
        llm=llm,
        audio_output=audio_output,
        avct_manager=AvctManager(config.avct),
        session_logger=session_logger,
        nao_adapter=nao_adapter,
        gate=gate,
        safety=SafetyChecker(),
        monitor=monitor,
        config_snapshot=_config_snapshot(config, args),
    )

    if args.no_ui:
        print("=" * 58)
        _run_terminal_mode(manager)
    else:
        print(f"  Operator console: http://localhost:{config.server.port}")
        print("=" * 58)
        import uvicorn
        from antagonist_robot.ui.server import create_app
        uvicorn.run(create_app(manager, session_logger), host=config.server.host, port=config.server.port)


def _config_snapshot(config, args) -> dict:
    """Serializable copy of the configuration (API keys removed) for the session record."""
    import os
    snap = dataclasses.asdict(config)
    root = str(snap.pop("project_root"))
    for section in ("llm", "monitor"):
        snap[section].pop("api_key", None)
    # store paths relative to the project so session records carry no local user paths
    for key in ("db_path", "audio_dir"):
        snap["logging"][key] = os.path.relpath(snap["logging"][key], root).replace("\\", "/")
    snap["cli"] = {"script": args.script, "nao_ip_override": args.nao_ip}
    return snap


def _run_terminal_mode(manager):
    """Terminal console: every held response is shown, and blocked ones ask for a decision."""
    participant_id = input("  Participant ID: ").strip() or "anonymous"
    polar_level = max(-3, min(3, int(input("  Polar level (-3 to +3): ").strip() or "0")))
    category = input("  Category (B-G): ").strip().upper() or "D"
    subtype = max(1, min(3, int(input("  Intensity class (1-3): ").strip() or "1")))

    def on_event(event: dict):
        if event.get("type") != "candidate":
            return
        print(f"\n  [{event['risk_rating']}] polar {event['polar_level']:+d}: {event['response']}")
        if event["auto_release"]:
            print(f"  (auto-send in {event['hold_seconds']}s)")
            return
        print(f"  Blocked: {', '.join(event['blocked_reasons'])}")
        choice = ""
        while choice not in ("s", "t", "r"):
            choice = input("  [s]end / [t]emper / [r]egenerate: ").strip().lower()
        manager.operator_action(event["candidate_id"], {"s": "send", "t": "temper", "r": "regenerate"}[choice])

    manager.on_event = on_event
    session_id = manager.start_session(polar_level, category, subtype, [], participant_id)
    print(f"\n  Session {session_id} started at polar {polar_level:+d}, category {category}{subtype}.")
    print("  Speak into the microphone. Press Ctrl+C to end.\n")

    try:
        while manager.is_running:
            result = manager.run_turn()
            if result is None:
                break
            print(f"\n--- Turn {result.turn_number} ({result.operator_action}) ---")
            print(f"  Participant: {result.transcript}")
            print(f"  Robot:       {result.llm_response}")
            if manager.end_requested:
                break
    except KeyboardInterrupt:
        pass
    finally:
        summary = manager.end_session()
        print(f"\n  Session ended. {summary['total_turns']} turns in {summary['duration_seconds']}s.")


if __name__ == "__main__":
    main()
