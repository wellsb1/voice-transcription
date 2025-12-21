"""Configuration loading with YAML file, env var, and CLI overrides."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class Config:
    """Application configuration with defaults."""

    # Backend selection
    backend: str = "whisper"  # "whisper" (faster-whisper) or "sherpa" (sherpa-onnx)

    # Device identification
    device_name: str = "default"

    # Audio settings
    audio_device: Optional[str] = None
    sample_rate: int = 16000

    # VAD-aware chunking settings
    vad_threshold: float = 0.85  # Higher = less sensitive, filters more noise
    silence_duration: float = 0.5  # seconds of silence to trigger chunk boundary
    min_chunk_duration: float = 2.0  # minimum chunk size for reliable speaker embedding
    max_chunk_duration: float = 30.0  # force chunk even without silence
    min_audio_energy: float = 0.01  # minimum RMS energy to process (filters quiet noise)
    min_speech_duration: float = 0.4  # minimum seconds of speech required in chunk

    # Model settings
    whisper_model: str = "medium.en"  # medium.en is most accurate for Pi 5

    # Speaker identification
    speaker_similarity_threshold: float = 0.35  # Lower = more likely to match same speaker
    speaker_reset_timeout: float = 900.0  # Reset speakers after N seconds of silence (15 min)
    speaker_registry_path: str = "./data/speakers.json"

    # Queue settings
    queue_dir: str = "./data/queue"

    # Whisper-specific settings
    whisper_compute_type: str = "int8"  # float16, int8, int8_float16

    # Sherpa-specific settings
    sherpa_use_int8: bool = False  # Use INT8 quantized models
    sherpa_provider: str = "cpu"  # cpu, cuda, coreml
    sherpa_num_threads: int = 4


def load_yaml_config(path: Path) -> dict:
    """Load configuration from YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides (TRANSCRIBE_* prefix)."""
    env_mapping = {
        "TRANSCRIBE_DEVICE_NAME": "device_name",
        "TRANSCRIBE_AUDIO_DEVICE": "audio_device",
        "TRANSCRIBE_SAMPLE_RATE": "sample_rate",
        "TRANSCRIBE_VAD_THRESHOLD": "vad_threshold",
        "TRANSCRIBE_SILENCE_DURATION": "silence_duration",
        "TRANSCRIBE_MIN_CHUNK_DURATION": "min_chunk_duration",
        "TRANSCRIBE_MAX_CHUNK_DURATION": "max_chunk_duration",
        "TRANSCRIBE_WHISPER_MODEL": "whisper_model",
        "TRANSCRIBE_SPEAKER_SIMILARITY_THRESHOLD": "speaker_similarity_threshold",
        "TRANSCRIBE_SPEAKER_REGISTRY_PATH": "speaker_registry_path",
        "TRANSCRIBE_QUEUE_DIR": "queue_dir",
    }

    result = config.copy()
    for env_var, config_key in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert types as needed
            if config_key == "sample_rate":
                value = int(value)
            elif config_key in (
                "vad_threshold",
                "silence_duration",
                "min_chunk_duration",
                "max_chunk_duration",
                "min_audio_energy",
                "speaker_similarity_threshold",
            ):
                value = float(value)
            elif value.lower() == "null" or value.lower() == "none":
                value = None
            result[config_key] = value

    return result


def apply_cli_overrides(config: dict, cli_args: dict) -> dict:
    """Apply CLI argument overrides."""
    result = config.copy()
    for key, value in cli_args.items():
        if value is not None:
            result[key] = value
    return result


def load_config(
    config_path: Optional[Path] = None,
    cli_args: Optional[dict] = None
) -> Config:
    """
    Load configuration with override hierarchy:
    1. Defaults (from Config dataclass)
    2. YAML config file
    3. Environment variables (TRANSCRIBE_* prefix)
    4. CLI arguments
    """
    # Start with empty dict, will use dataclass defaults for missing keys
    config = {}

    # Load YAML config
    if config_path is None:
        config_path = Path("config.yaml")
    yaml_config = load_yaml_config(config_path)

    # Extract backend-specific sections before merging
    whisper_overrides = yaml_config.pop("whisper", {})
    sherpa_overrides = yaml_config.pop("sherpa", {})

    # Apply shared config first
    config.update(yaml_config)

    # Apply backend-specific overrides based on selected backend
    backend = config.get("backend", "whisper")
    if backend == "whisper":
        config.update(whisper_overrides)
    elif backend == "sherpa":
        config.update(sherpa_overrides)

    # Apply env overrides
    config = apply_env_overrides(config)

    # Apply CLI overrides
    if cli_args:
        config = apply_cli_overrides(config, cli_args)

    # Create Config instance with merged values
    return Config(**config)
