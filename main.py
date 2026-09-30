"""Antagonistic Robot — Main entry point.

Initializes all components, starts the web server (or runs in headless
terminal mode with --no-ui), and orchestrates the conversation system.

Usage:
    python main.py                    # Start with web UI
    python main.py --no-ui            # Headless terminal mode
    python main.py --config my.yaml   # Custom config file
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    """Parse arguments, load config, initialize all components, and start."""
    parser = argparse.ArgumentParser(
        description="Antagonistic Robot — Hostile voice conversation system for HRI research"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run in headless terminal mode without the web server",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load config
    from antagonist_robot.config.settings import load_config

    config = load_config(args.config)

    # Print startup banner
    print("=" * 54)
    print("  Antagonistic Robot — Hostile Voice Conversation System")
    print("=" * 54)

    # Initialize pipeline components
    from antagonist_robot.pipeline.audio_capture import AudioCapture
    from antagonist_robot.pipeline.asr import ASREngine
    from antagonist_robot.pipeline.llm import LLMEngine
    from antagonist_robot.pipeline.tts import OpenAITTSEngine
    from antagonist_robot.pipeline.audio_output import NAOAudioOutput

    print(f"  Loading ASR model ({config.asr.model_size})...")
    capture = AudioCapture(config.audio)
    asr = ASREngine(config.asr)

    print(f"  LLM: {config.llm.provider_name} ({config.llm.model})")
    llm = LLMEngine(config.llm)

    print(f"  TTS: {config.tts.engine} ({config.tts.default_voice})")
    tts = OpenAITTSEngine(config.tts)

    # Audio output + NAO adapter
    from antagonist_robot.nao.real import RealNAO

    print(f"  NAO: {config.nao.ip}:{config.nao.port}")
    audio_output = NAOAudioOutput(
        ip=config.nao.ip,
        port=config.nao.port,
        use_builtin_tts=config.nao.use_builtin_tts,
    )
    nao_adapter = RealNAO(
        config.nao.ip, config.nao.port, config.nao.naoqi_port, config.nao.password
    )

    nao_adapter.connect()
    if not nao_adapter.is_connected():
        print(
            f"\n  ERROR: NAO speaker server not reachable at "
            f"{config.nao.ip}:{config.nao.port}.\n"
            f"  Run: python deploy_nao.py   (starts the speaker server on the robot)\n"
            f"  If that fails, see 'Troubleshooting the NAO connection' in README.md."
        )
        sys.exit(1)

    # Logger
    from antagonist_robot.logging.session_logger import SessionLogger

    session_logger = SessionLogger(
        db_path=config.logging.db_path,
        audio_dir=config.logging.audio_dir,
        save_audio=config.logging.save_audio,
    )

    # AVCT manager
    from antagonist_robot.conversation.avct_manager import AvctManager
    avct = AvctManager(config.avct)

    # Conversation manager
    from antagonist_robot.conversation.manager import ConversationManager

    manager = ConversationManager(
        audio_capture=capture,
        asr=asr,
        llm=llm,
        tts=tts,
        audio_output=audio_output,
        avct_manager=avct,
        session_logger=session_logger,
        nao_adapter=nao_adapter,
    )

    if args.no_ui:
        print("=" * 54)
        _run_terminal_mode(manager)
    else:
        print(f"  Web UI: http://{config.server.host}:{config.server.port}")
        print("=" * 54)
        _run_web_mode(manager, tts, session_logger, config)


def _run_web_mode(manager, tts, session_logger, config):
    """Start the FastAPI web server with uvicorn."""
    import uvicorn
    from antagonist_robot.ui.server import create_app

    static_dir = Path(__file__).parent / "webui" / "build"
    app = create_app(manager, tts, session_logger, static_dir)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


def _run_terminal_mode(manager):
    """Simple terminal interface for headless operation."""
    participant_id = input("  Participant ID: ").strip() or "anonymous"
    polar_str = input("  Polar level (-3 to +3): ").strip() or "0"
    polar_level = max(-3, min(3, int(polar_str)))
    category = input("  Category (B-G): ").strip().upper() or "D"
    subtype_str = input("  Subtype (1-3): ").strip() or "1"
    subtype = max(1, min(3, int(subtype_str)))

    session_id = manager.start_session(polar_level, category, subtype, [], participant_id)
    print(f"\n  Session {session_id} started at polar {polar_level}, Cat {category}{subtype}.")
    print("  Speak into your microphone. Press Ctrl+C to end.\n")

    try:
        while manager.is_running:
            result = manager.run_turn()
            print(f"\n--- Turn {result.turn_number} ---")
            print(f"  You:   {result.transcript}")
            print(f"  Agent: {result.llm_response}")
            latency = result.latency
            print(
                f"  Latency: VAD={latency.get('vad_ms')}ms "
                f"ASR={latency.get('asr_ms')}ms "
                f"LLM={latency.get('llm_ms')}ms "
                f"TTS={latency.get('tts_ms')}ms "
                f"Total={latency.get('total_ms')}ms"
            )
    except KeyboardInterrupt:
        pass
    finally:
        summary = manager.end_session()
        print(
            f"\n  Session ended. {summary['total_turns']} turns "
            f"in {summary['duration_seconds']}s."
        )


if __name__ == "__main__":
    main()
