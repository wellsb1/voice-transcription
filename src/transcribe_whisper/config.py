"""Configuration loading for faster-whisper backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transcribe_shared.config import load_yaml, apply_env_overrides, filter_to_dataclass


@dataclass
class Config:
    """Faster-whisper backend configuration."""

    # Device identification
    device_name: str = "whisper-device"

    # Audio settings
    audio_device: Optional[str] = None
    sample_rate: int = 16000

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


# Environment variable mapping
_ENV_MAPPING = {
    "TRANSCRIBE_DEVICE_NAME": "device_name",
    "TRANSCRIBE_AUDIO_DEVICE": "audio_device",
    "TRANSCRIBE_SAMPLE_RATE": "sample_rate",
    "TRANSCRIBE_VAD_THRESHOLD": "vad_threshold",
    "TRANSCRIBE_SILENCE_DURATION": "silence_duration",
    "TRANSCRIBE_MIN_CHUNK_DURATION": "min_chunk_duration",
    "TRANSCRIBE_MAX_CHUNK_DURATION": "max_chunk_duration",
    "TRANSCRIBE_WHISPER_MODEL": "whisper_model",
    "TRANSCRIBE_WHISPER_COMPUTE_TYPE": "whisper_compute_type",
    "TRANSCRIBE_SPEAKER_SIMILARITY_THRESHOLD": "speaker_similarity_threshold",
    "TRANSCRIBE_SPEAKER_INACTIVITY_TIMEOUT": "speaker_inactivity_timeout",
    "TRANSCRIBE_QUEUE_DIR": "queue_dir",
}

_TYPE_CONVERSIONS = {
    "sample_rate": int,
    "vad_threshold": float,
    "silence_duration": float,
    "min_chunk_duration": float,
    "max_chunk_duration": float,
    "min_audio_energy": float,
    "min_speech_duration": float,
    "speaker_similarity_threshold": float,
    "speaker_inactivity_timeout": float,
}


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration with override hierarchy:
    1. Defaults (from Config dataclass)
    2. YAML config file (config-whisper.yaml)
    3. Environment variables
    """
    if config_path is None:
        config_path = Path("config-whisper.yaml")

    config = load_yaml(config_path)
    config = apply_env_overrides(config, _ENV_MAPPING, _TYPE_CONVERSIONS)
    config = filter_to_dataclass(config, Config)

    return Config(**config)
