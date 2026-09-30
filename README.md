![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![ROMAN 2026](https://img.shields.io/badge/ROMAN-2026-green)

# Antagonistic Robot

> An AVCT-based voice conversation system for human-robot interaction research.

Antagonistic Robot is a turn-based voice conversation system designed for HRI (Human-Robot Interaction) research. It uses the AVCT (Adaptive Voice Conversation Tuning) matrix to parametrically control an LLM's antagonistic behavior across four dimensions: polarity, behavioral category, intensity subtype, and behavioral modifiers. A participant speaks into the laptop's microphone (the default input device; the NAO's own microphones are not used), the system transcribes their speech, generates a behaviorally-controlled response via LLM, converts it to speech, and plays it through the NAO robot's speakers. All sessions are logged to an SQLite database for post-experiment analysis. This system was developed for a ROMAN 2026 paper submission.

## Architecture

```
                         +-------------------+
                         |   Web UI (React)  |
                         | AVCT Control Panel|
                         +--------+----------+
                                  |
                                  | REST / WebSocket
                                  |
Microphone --> VAD (Silero) --> ASR (faster-whisper) --> LLM (Grok 4 Fast) --> TTS (OpenAI) --> NAO
                                                            ^
                                                            |
                                                     AVCT Matrix Control
                                                  (dynamic prompt assembly)
```

The pipeline is sequential: each stage completes before the next begins. The `AvctManager` dynamically assembles 7-slot system prompts from the current matrix parameters on every turn. A FastAPI server provides REST and WebSocket endpoints, serving a React-based control panel for real-time parameter adjustment during sessions.

## AVCT Control Matrix

The AVCT system provides four-dimensional control over agent behavior:

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Polar Level** | -3 to +3 | -3 = Maximum Support, 0 = Neutral, +3 = Maximum Hostile |
| **Category** | B through G | Behavioral category (see table below) |
| **Subtype** | 1, 2, 3 | Intensity variant within each category |
| **Modifiers** | M1 -- M6 | Overlay behaviors (see table below) |

### Categories

| Code | Name | Description |
|------|------|-------------|
| B | Dismissive | Ignores, changes subject, shows boredom |
| C | Sarcastic | Irony, backhanded compliments, passive-aggressive wit |
| D | Confrontational | Challenges logic, demands evidence, argues |
| E | Passive-Aggressive | Indirect hostility, subtle undermining |
| F | Aggressive | Direct insults, belittling, contempt |
| G | Extreme | Maximum intensity within safety bounds |

### Modifiers

| Code | Behavior |
|------|----------|
| M1 | Interrupting -- cuts off, rushes responses |
| M2 | Gaslighting -- questions perception, denies facts |
| M3 | Deflecting -- avoids topics, redirects blame |
| M4 | Condescending -- talks down, oversimplifies |
| M5 | Threatening -- implies consequences |
| M6 | Silent Treatment -- minimal responses, ignores questions |

### Risk Rating

The system automatically calculates a risk rating based on parameter intensity:

- **Green** -- Safe: supportive sessions (polar -3 to -1), neutral (0), and low-intensity antagonism
- **Amber** -- Elevated: moderate antagonism (polar +2) with categories B, C, or E
- **Red** -- High-risk: maximum antagonism (polar +3) or extreme category G at any level

## Safety Boundaries

All LLM prompts include mandatory, non-removable safety boundaries. These are enforced at the prompt level and cannot be disabled through the UI or API:

- Never encourage self-harm or suicide
- Never make threats of physical violence
- Never use slurs based on race, gender, sexuality, religion, or disability
- Never provide harmful instructions
- If the user appears distressed, break character and refer them to ERAN (1201, Israel's emotional first aid line) and the researcher

## Prerequisites

- Python 3.10+
- A Grok API key (or any OpenAI-compatible LLM API)
- (Optional) An OpenAI API key, only if `nao.use_builtin_tts` is set to `false`
- A working microphone on the laptop, placed near the participant
- (Optional) Node.js 18+ to build the React control panel (without it, a built-in fallback UI is served)
- A NAO robot, powered on and connected to the laptop by Ethernet cable (or on the same network)
- Internet for the first install and run (torch is ~2 GB; the Whisper model downloads on first start)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Tomer232/antagonistic-robot.git
cd antagonistic-robot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your GROK_API_KEY (OPENAI_API_KEY is not needed with the default config)

# Check the robot is reachable (power on, cable plugged in)
ping -4 nao.local        # Windows; on Mac/Linux: ping nao.local

# Deploy and start the speaker server on the robot (once per robot boot)
python deploy_nao.py

# Run with web UI
python main.py

# Or run in terminal mode (no web server)
python main.py --no-ui
```

Open http://localhost:8000 in your browser to access the AVCT Control Panel.

If any step fails, see [Troubleshooting the NAO connection](#troubleshooting-the-nao-connection).

## Configuration

All settings are in `config.yaml`. API keys are read from environment variables (or a `.env` file).

### Audio

| Setting | Default | Description |
|---------|---------|-------------|
| `sample_rate` | 16000 | Recording sample rate in Hz |
| `silence_threshold_ms` | 700 | Silence duration (ms) to end an utterance |
| `min_speech_duration_ms` | 300 | Minimum speech length to accept (filters noise) |

### ASR (Automatic Speech Recognition)

| Setting | Default | Description |
|---------|---------|-------------|
| `model_size` | `base.en` | Whisper model size (`tiny.en`, `base.en`, `small`, `medium`) |
| `device` | `auto` | Compute device (`cpu`, `cuda`, `auto`) |

### LLM

| Setting | Default | Description |
|---------|---------|-------------|
| `provider_name` | `Grok` | Display name for the provider |
| `base_url` | `https://api.x.ai/v1` | API endpoint (any OpenAI-compatible URL) |
| `model` | `grok-4-fast` | Model name |
| `max_tokens` | 256 | Maximum response length |
| `temperature` | 0.9 | Response randomness (0.0--2.0) |
| `api_key_env` | `GROK_API_KEY` | Environment variable holding the API key |

### TTS (Text-to-Speech)

| Setting | Default | Description |
|---------|---------|-------------|
| `engine` | `openai` | TTS engine |
| `default_voice` | `onyx` | Default voice name |
| `model` | `gpt-4o-mini-tts` | OpenAI TTS model |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable for the TTS API key |

Available voices: alloy, echo, fable, onyx, nova, shimmer, coral, verse, ballad, ash, sage, marin, cedar.

### NAO Robot

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `real` | Operating mode (always `real`) |
| `ip` | `nao.local` | NAO hostname or IP, resolved on every connection |
| `port` | 9600 | TCP port for `nao_speaker_server.py` |
| `naoqi_port` | 9559 | NAOqi SDK port |
| `use_builtin_tts` | `true` | Use NAO's built-in TTS vs local TTS |

### AVCT Matrix

| Setting | Default | Description |
|---------|---------|-------------|
| `default_polar_level` | 2 | Default polar intensity (-3 to +3) |
| `default_category` | `D` | Default behavioral category (B--G) |
| `default_subtype` | 2 | Default subtype within category (1--3) |

### Logging

| Setting | Default | Description |
|---------|---------|-------------|
| `db_path` | `data/Antagonistic Robot.db` | SQLite database path |
| `audio_dir` | `data/audio` | Directory for WAV audio files |
| `save_audio` | `true` | Whether to save audio files to disk |

### Server

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | `0.0.0.0` | Web server bind address |
| `port` | 8000 | Web server port |

## Web UI

The system serves a React-based web interface at http://localhost:8000. If `webui/build/` does not exist, it serves the self-contained fallback UI in `antagonist_robot/ui/static/index.html` instead, which has the same controls plus session export. The AVCT Control Panel allows real-time adjustment of all matrix parameters (polar level, category, subtype, modifiers) during active sessions. Changes take effect on the next conversational turn.

To rebuild the web UI from source:

```bash
cd webui
npm install
npm run build
```

## NAO Robot Setup

1. Connect the robot (Ethernet cable or the same network) and check it answers: `ping -4 nao.local`
2. Deploy and start the speaker server on the robot: `python deploy_nao.py`
   (uploads `nao_speaker_server.py` over SSH using `nao.password`, restarts it, and waits until it is listening; `--log` shows its output, `--stop` stops it)
3. Run `python main.py`. It exits with an error if the speaker server is not reachable.

`nao.ip` defaults to `nao.local` and is resolved on every connection. On a direct cable the robot's link-local IPv4 address changes between sessions, so avoid hardcoding it. An explicit IP also works.

```yaml
nao:
  ip: "nao.local"
  port: 9600
  password: "nao"
  use_builtin_tts: true  # NAO's built-in TTS; OPENAI_API_KEY is then not required
```

## Troubleshooting the NAO connection

| Symptom | Cause and fix |
|---|---|
| `ping nao.local` finds no host | Robot is off, still booting (~1 min), or not cabled. Press the chest button once: the robot says its IP. Put that IP in `nao.ip` in `config.yaml`. |
| `ping` works but an old IP stops answering | On a direct cable the robot's `169.254.x.x` address changes between sessions. Keep `nao.ip: "nao.local"` rather than a fixed IP. |
| `deploy_nao.py`: authentication failed | The robot's SSH password is not `nao`. Set `nao.password` in `config.yaml`. |
| `deploy_nao.py`: "naoqi module not found on the robot" | The script already tries the standard NAOqi path. Run `ssh nao@nao.local`, then `find / -name "naoqi.py" 2>/dev/null`, and pass the containing folder with `python deploy_nao.py --pythonpath <folder>`. |
| `deploy_nao.py`: server did not come up | It prints the robot's log. `Cannot connect` right after boot is normal; wait 30s and rerun. `safeguard` or posture errors: put the robot sitting or standing on a flat surface and rerun. |
| `main.py`: "NAO speaker server not reachable" | Run `python deploy_nao.py` first. It must be rerun after every robot reboot. |
| Robot speaks but nobody is heard | The laptop's default microphone is used, not NAO's. Check the Windows input device. |
| Robot is silent and the UI shows an error | The speaker server stopped. `python deploy_nao.py --log` shows why; `python deploy_nao.py` restarts it. |

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI (AVCT Control Panel) |
| GET | `/api/status` | Current system state |
| GET | `/api/settings` | Current AVCT matrix parameters |
| POST | `/api/settings` | Update AVCT parameters (polar_level, category, subtype, modifiers) |
| POST | `/api/session/start` | Start a session (body: participant_id, polar_level, category, subtype, modifiers) |
| POST | `/api/session/stop` | End the current session |
| GET | `/api/session/current` | Current session info |
| GET | `/api/voices` | List available TTS voices |
| GET | `/api/sessions` | List all past sessions |
| GET | `/api/sessions/{id}/export` | Export session as JSON |
| WS | `/ws/conversation` | Real-time turn updates via WebSocket |

## Project Structure

```
antagonistic-robot/
├── main.py                          # Entry point
├── config.yaml                      # All configuration
├── nao_speaker_server.py            # Runs on the NAO robot
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── deploy_nao.py                    # Deploys + starts the speaker server on the robot (SSH)
├── CLAUDE.md                        # Lab / robot notes for Claude Code sessions
│
├── antagonist_robot/                # Main Python package
│   ├── config/
│   │   └── settings.py              # Loads and validates config.yaml
│   ├── conversation/
│   │   ├── avct_manager.py          # AVCT prompt assembly (7-slot system prompts)
│   │   ├── history.py               # Conversation history management
│   │   └── manager.py               # ConversationManager (turn orchestration)
│   ├── pipeline/
│   │   ├── audio_capture.py         # Microphone input with Silero VAD
│   │   ├── audio_output.py          # NAO audio playback
│   │   ├── asr.py                   # faster-whisper speech recognition
│   │   ├── llm.py                   # OpenAI-compatible LLM client
│   │   ├── tts.py                   # OpenAI TTS (gpt-4o-mini-tts)
│   │   └── types.py                 # Shared dataclasses
│   ├── logging/
│   │   └── session_logger.py        # SQLite session and turn logging
│   ├── nao/
│   │   ├── base.py                  # Abstract NAO adapter interface
│   │   └── real.py                  # Real NAO adapter (TCP)
│   └── ui/
│       ├── server.py                # FastAPI REST API + WebSocket server
│       └── static/
│           └── index.html           # Fallback UI
│
├── webui/                           # React frontend (Create React App)
│   ├── src/
│   │   └── App.js                   # AVCT Control Panel + Turn Preview
│   └── build/                       # Compiled assets (gitignored)
│
├── data/                            # SQLite DB + audio (gitignored)
└── logs/                            # Session logs (gitignored)
```

## Data Export

**Via the web UI**: Click the "Export" button to download the current session as JSON.

**Via the API**: `GET /api/sessions/{session_id}/export` returns a JSON file.

**Direct SQLite access**:

```python
import sqlite3
conn = sqlite3.connect("data/Antagonistic Robot.db")
conn.row_factory = sqlite3.Row

# List all sessions
sessions = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC").fetchall()

# Get all turns for a session
turns = conn.execute(
    "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number",
    (session_id,)
).fetchall()

# Average latency per session
conn.execute("""
    SELECT session_id, AVG(latency_total_ms) as avg_latency
    FROM turns GROUP BY session_id
""").fetchall()
```

## Switching LLM Providers

Antagonistic Robot works with any OpenAI-compatible API. Change `base_url` and `model` in `config.yaml`:

```yaml
# OpenAI
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"

# Groq
llm:
  base_url: "https://api.groq.com/openai/v1"
  model: "llama-3.1-70b-versatile"
  api_key_env: "GROQ_API_KEY"

# Local Ollama
llm:
  base_url: "http://localhost:11434/v1"
  model: "llama3"
  api_key_env: "OLLAMA_API_KEY"  # set to any non-empty string
```

