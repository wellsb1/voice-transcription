"""Configuration loading for faster-whisper backend."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transcribe_shared.config import load_yaml, apply_env_overrides, filter_to_dataclass, resolve_config_path


@dataclass
class Config:
    """Faster-whisper backend configuration."""

    # Device identification
    device_name: str = "whisper-device"

    # Audio settings
    audio_device: Optional[str] = None
    sample_rate: int = 16000

    # Transcript filtering - words to always ignore (e.g., keyboard click sounds)
    ignore_words: list[str] = field(default_factory=list)

    # VAD-aware chunking settings
    vad_threshold: float = 0.85
    silence_duration: float = 0.5
    min_chunk_duration: float = 2.0
    max_chunk_duration: float = 30.0
    min_audio_energy: float = 0.01
    min_speech_duration: float = 0.4

    # Whisper model settings
    whisper_model: str = "medium.en"
    whisper_compute_type: str = "int8"

    # Speaker identification
    speaker_similarity_threshold: float = 0.75
    speaker_inactivity_timeout: float = 1800.0

    # Queue settings
    queue_dir: str = "./data/queue-whisper"


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration with override hierarchy:
    1. Defaults (from Config dataclass)
    2. YAML config file (with ${VAR:-default} interpolation)
    3. Environment variables (UPPERCASE_KEY format)

    Config path resolution: CLI arg > TRANSCRIBE_WHISPER_CONFIG env var > ./config-backend-whisper.yaml
    """
    config_path = resolve_config_path(
        cli_path=config_path,
        env_var="TRANSCRIBE_WHISPER_CONFIG",
        default_path=Path("config-backend-whisper.yaml"),
    )

    config = load_yaml(config_path)
    config = apply_env_overrides(config, Config)
    config = filter_to_dataclass(config, Config)

    return Config(**config)
