"""Configuration loader and validation for Antagonistic Robot.

Loads config.yaml, validates all fields using dataclasses, and resolves
API keys from environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000
    silence_threshold_ms: int = 700
    min_speech_duration_ms: int = 300


@dataclass
class ASRConfig:
    """Automatic speech recognition settings."""
    model_size: str = "base.en"
    device: str = "auto"


@dataclass
class LLMConfig:
    """LLM provider settings. Provider-agnostic via OpenAI-compatible API."""
    provider_name: str = "Grok"
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.20-0309-non-reasoning"
    max_tokens: int = 256
    temperature: float = 0.9
    api_key_env: str = "GROK_API_KEY"
    stream: bool = False
    api_key: str = field(default="", repr=False)


@dataclass
class NAOConfig:
    """NAO robot connection settings."""
    ip: str = "nao.local"
    port: int = 9600
    naoqi_port: int = 9559
    password: str = "nao"


@dataclass
class AvctConfig:
    """AVCT parameter configuration via Polar Scale and Categories."""
    default_polar_level: int = 2
    default_category: str = "D"
    default_subtype: int = 2


@dataclass
class OperatorConfig:
    """Operator review gate: how candidate responses are released to the robot."""
    review_mode: str = "timed"          # "timed" or "manual"
    hold_seconds: float = 3.0           # review window before automatic release
    block_auto_send_at: str = "Orange"  # responses rated at or above this need an explicit Send


@dataclass
class MonitorConfig:
    """Optional psychosocial risk monitor (DialogGuard dimensions)."""
    enabled: bool = False
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.20-0309-non-reasoning"
    api_key_env: str = "GROK_API_KEY"
    timeout_s: float = 20.0
    gate_auto_send: bool = True         # auto-release waits for scores; clear risk or failure blocks it
    api_key: str = field(default="", repr=False)


@dataclass
class LoggingConfig:
    """Session logging and data storage settings."""
    db_path: str = "data/Antagonistic Robot.db"
    audio_dir: str = "data/audio"
    save_audio: bool = True


@dataclass
class ServerConfig:
    """Web UI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    """Top-level application configuration."""
    audio: AudioConfig
    asr: ASRConfig
    llm: LLMConfig
    nao: NAOConfig
    avct: AvctConfig
    operator: OperatorConfig
    monitor: MonitorConfig
    logging: LoggingConfig
    server: ServerConfig
    project_root: Path = field(default_factory=lambda: Path.cwd())


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and validate config from YAML file.

    Resolves the LLM API key from the environment variable named in config.
    Raises clear errors if required fields are missing or the config file
    is not found.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Fully validated AppConfig with resolved API keys.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project_root = path.parent.resolve()

    # Build each sub-config, filtering out unexpected keys
    audio = _build_dataclass(AudioConfig, raw.get("audio", {}))
    asr = _build_dataclass(ASRConfig, raw.get("asr", {}))
    llm = _build_dataclass(LLMConfig, raw.get("llm", {}))
    nao = _build_dataclass(NAOConfig, raw.get("nao", {}))
    avct_cfg = _build_dataclass(AvctConfig, raw.get("avct", {}))
    operator = _build_dataclass(OperatorConfig, raw.get("operator", {}))
    monitor = _build_dataclass(MonitorConfig, raw.get("monitor", {}))
    logging_cfg = _build_dataclass(LoggingConfig, raw.get("logging", {}))
    server = _build_dataclass(ServerConfig, raw.get("server", {}))

    # Resolve LLM API key from environment
    llm.api_key = os.environ.get(llm.api_key_env, "")
    if not llm.api_key:
        raise ValueError(
            f"LLM API key environment variable '{llm.api_key_env}' is not set. "
            f"Set it with: export {llm.api_key_env}=your-key-here"
        )

    if operator.review_mode not in ("timed", "manual"):
        raise ValueError(f"operator.review_mode must be 'timed' or 'manual', got {operator.review_mode!r}")

    # Resolve the monitor's API key (only needed when the monitor is enabled)
    if monitor.enabled:
        monitor.api_key = os.environ.get(monitor.api_key_env, "")
        if not monitor.api_key:
            raise ValueError(
                f"monitor.enabled is true but '{monitor.api_key_env}' is not set. "
                f"Set it, or set monitor.enabled to false."
            )

    # Resolve relative paths to absolute
    logging_cfg.db_path = str(project_root / logging_cfg.db_path)
    logging_cfg.audio_dir = str(project_root / logging_cfg.audio_dir)
    return AppConfig(
        audio=audio,
        asr=asr,
        llm=llm,
        nao=nao,
        avct=avct_cfg,
        operator=operator,
        monitor=monitor,
        logging=logging_cfg,
        server=server,
        project_root=project_root,
    )


def _build_dataclass(cls, data: dict):
    """Build a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)
