"""Configuration loading for M4 transcription."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class Config:
    """M4 transcription configuration."""

    # ASR settings
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Diarization settings
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    huggingface_token: Optional[str] = None

    # Batch parameters
    min_batch_duration: float = 30.0  # Minimum batch for pyannote context
    max_batch_duration: float = 60.0  # Force batch during monologues
    silence_duration: float = 0.5  # Silence threshold for batch boundary

    # Speaker tracking
    speaker_similarity_threshold: float = 0.75  # Cosine similarity for matching
    speaker_inactivity_timeout: float = 1800.0  # Reset after 30 min silence

    # Audio settings
    audio_device: Optional[str] = None  # None = system default
    sample_rate: int = 16000

    # Output settings
    device_name: str = "m4-mini"


def load_yaml_config(path: Path) -> dict:
    """Load configuration from YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    env_mapping = {
        "TRANSCRIBE_WHISPER_MODEL": "whisper_model",
        "TRANSCRIBE_DIARIZATION_MODEL": "diarization_model",
        "HF_TOKEN": "huggingface_token",
        "TRANSCRIBE_MIN_BATCH_DURATION": "min_batch_duration",
        "TRANSCRIBE_MAX_BATCH_DURATION": "max_batch_duration",
        "TRANSCRIBE_SILENCE_DURATION": "silence_duration",
        "TRANSCRIBE_SPEAKER_SIMILARITY_THRESHOLD": "speaker_similarity_threshold",
        "TRANSCRIBE_SPEAKER_INACTIVITY_TIMEOUT": "speaker_inactivity_timeout",
        "TRANSCRIBE_AUDIO_DEVICE": "audio_device",
        "TRANSCRIBE_SAMPLE_RATE": "sample_rate",
        "TRANSCRIBE_DEVICE_NAME": "device_name",
    }

    result = config.copy()
    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Type conversion
            if config_key == "sample_rate":
                value = int(value)
            elif config_key in (
                "min_batch_duration",
                "max_batch_duration",
                "silence_duration",
                "speaker_similarity_threshold",
                "speaker_inactivity_timeout",
            ):
                value = float(value)
            elif value.lower() in ("null", "none"):
                value = None
            result[config_key] = value

    return result


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration with override hierarchy:
    1. Defaults (from Config dataclass)
    2. YAML config file
    3. Environment variables
    """
    config = {}

    # Load YAML config
    if config_path is None:
        config_path = Path("config.m4.yaml")
    config.update(load_yaml_config(config_path))

    # Apply env overrides
    config = apply_env_overrides(config)

    # Remove keys not in Config dataclass (e.g., 'logs' section used by other tools)
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}
    config = {k: v for k, v in config.items() if k in valid_keys}

    return Config(**config)
