# Antagonistic Robot (AVCT-LLM) — Build and Architecture Guide

> **Updated March 2026** — Reflects the AVCT-LLM Control Interface refactor.

## What This System Is

Antagonistic Robot is a turn-based voice conversation system for HRI (Human-Robot Interaction) research. A participant speaks into the laptop's microphone, the system transcribes their speech, sends it to an LLM (Grok, grok-4.20-0309-non-reasoning) that responds with behavior controlled by the **AVCT matrix**, converts the response to speech (OpenAI TTS), and plays it through the NAO robot's speakers.

The AVCT matrix replaces the old 1-5 hostility scale with a multi-dimensional parametric control system.

---

## Project Structure

```
NAO_LLM/
├── main.py                      # Entry point — initializes components, starts server
├── config.yaml                  # All configuration (LLM, TTS, NAO, AVCT defaults)
├── nao_speaker_server.py        # Runs on the NAO robot to receive TTS commands
├── requirements.txt             # Python dependencies
├── .env                         # API keys (GROK_API_KEY, OPENAI_API_KEY)
│
├── antagonist_robot/                    # Main Python package
│   ├── config/settings.py       # Loads & validates config.yaml
│   ├── conversation/
│   │   ├── avct_manager.py      # AVCT prompt assembly (7-slot system prompts)
│   │   ├── history.py           # Conversation history management
│   │   └── manager.py           # ConversationManager — orchestrates each turn
│   ├── pipeline/
│   │   ├── audio_capture.py     # Microphone input with Silero VAD
│   │   ├── audio_output.py      # NAO audio playback
│   │   ├── asr.py               # faster-whisper speech recognition
│   │   ├── llm.py               # OpenAI-compatible LLM client (Grok)
│   │   ├── tts.py               # OpenAI TTS (gpt-4o-mini-tts)
│   │   └── types.py             # Shared dataclasses (TurnResult, AudioData, etc.)
│   ├── logging/session_logger.py # SQLite session & turn logging
│   ├── nao/
│   │   ├── base.py              # Abstract NAO adapter interface
│   │   └── real.py              # Real NAO adapter (TCP)
│   └── ui/server.py             # FastAPI REST API + WebSocket server
│
├── webui/                       # React frontend
│   ├── src/App.js               # AVCT Control Panel + Turn Preview Monitor
│   └── build/                   # Compiled static assets served by FastAPI
│
├── data/                        # SQLite DB + saved audio (gitignored)
└── logs/                        # Session log files
```

---

## AVCT Control Matrix

The AVCT (Adaptive Voice Conversation Tuning) system replaces the old hostility scale with four dimensions:

| Dimension | Range | Description |
|---|---|---|
| **Polar Level** | -3 to +3 | -3 = Maximum Support, 0 = Neutral, +3 = Maximum Hostile |
| **Category** | B through G | Behavioral category (B=Dismissive, C=Sarcastic, D=Confrontational, E=Passive-Aggressive, F=Aggressive, G=Extreme) |
| **Subtype** | 1, 2, 3 | Intensity variant within each category |
| **Modifiers** | M1–M6 | Overlay behaviors (M1=Interrupting, M2=Gaslighting, M3=Deflecting, M4=Condescending, M5=Threatening, M6=Silent Treatment) |

### Risk Rating

The system automatically calculates a risk rating based on the current parameters:
- **Green** — Safe parameters (low polar, mild categories)
- **Amber** — Elevated parameters (moderate polar, mid categories)
- **Red** — High-risk parameters (high polar, extreme categories)

### Safety Boundaries

All LLM prompts include **mandatory, non-removable safety boundaries**:
- Never encourage self-harm or suicide
- Never make threats of physical violence
- Never use slurs based on race, gender, sexuality, religion, or disability
- Never provide harmful instructions
- If the user appears distressed, break character and provide support resources

---

## Key Architecture Decisions

1. **Sequential pipeline** — Audio capture → ASR → LLM → TTS → Audio output. Each step completes before the next. No async in the pipeline.
2. **Dynamic prompt assembly** — `AvctManager` builds 7-slot system prompts from the matrix parameters. No static prompt files.
3. **FastAPI + React** — The backend serves a compiled React app and provides REST + WebSocket endpoints.
4. **Config-driven** — All settings in `config.yaml`. API keys via `.env` environment variables.
5. **SQLite logging** — Every turn logs transcript, LLM output, AVCT parameters, risk rating, and latency breakdown.

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
GROK_API_KEY=your_grok_key
OPENAI_API_KEY=your_openai_key

# Start the server (web UI mode)
python main.py

# Start headless (terminal mode)
python main.py --no-ui
```

The web UI is accessible at `http://localhost:8000`.

---

## Configuration (config.yaml)

| Section | Key Settings |
|---|---|
| `llm` | `model: grok-4.20-0309-non-reasoning`, `base_url: https://api.x.ai/v1`, `max_tokens: 256`, `temperature: 0.9` |
| `tts` | `engine: openai`, `model: gpt-4o-mini-tts`, `default_voice: onyx` |
| `audio` | `sample_rate: 16000`, `silence_threshold_ms: 700` |
| `nao` | `mode: real`, `ip`, `port` |
| `avct` | `default_polar_level: 2`, `default_category: D`, `default_subtype: 2` |

---

## General Guidelines

- All code uses type hints and docstrings
- Dataclasses for all inter-component data types
- Every class and public method has a docstring
- Safety boundaries in prompts are **non-negotiable** and present in every prompt path
- Favor simplicity — this is a research tool, not a consumer product
