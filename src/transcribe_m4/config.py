"""Configuration loading for M4 transcription."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transcribe_shared.config import load_yaml, apply_env_overrides, filter_to_dataclass, resolve_config_path


@dataclass
class Config:
    """M4 transcription configuration."""

    # ASR settings
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Transcript filtering - words to always ignore (e.g., keyboard click sounds)
    ignore_words: list[str] = field(default_factory=list)

    # Diarization settings
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    huggingface_token: Optional[str] = None

    # Batch parameters
    min_batch_duration: float = 30.0  # Minimum batch for pyannote context
    max_batch_duration: float = 60.0  # Force batch during monologues
    silence_duration: float = 0.5  # Silence threshold for batch boundary
    vad_threshold: float = 0.5  # VAD sensitivity (0.0-1.0, higher = less sensitive)
    min_audio_energy: float = 0.01  # Minimum RMS energy to process (filters quiet noise)

    # Speaker tracking
    speaker_similarity_threshold: float = 0.75  # Cosine similarity for matching
    speaker_inactivity_timeout: float = 1800.0  # Reset after 30 min silence

    # Audio settings
    audio_device: Optional[str] = None  # None = system default (input)
    audio_output_device: Optional[str] = None  # None = system default (output, for debug playback)
    sample_rate: int = 16000

    # Output settings
    device_name: str = "m4-mini"


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration with override hierarchy:
    1. Defaults (from Config dataclass)
    2. YAML config file (with ${VAR:-default} interpolation)
    3. Environment variables (UPPERCASE_KEY format)

    Config path resolution: CLI arg > TRANSCRIBE_M4_CONFIG env var > ./config-backend-m4.yaml
    """
    config_path = resolve_config_path(
        cli_path=config_path,
        env_var="TRANSCRIBE_M4_CONFIG",
        default_path=Path("config-backend-m4.yaml"),
    )

    config = load_yaml(config_path)
    config = apply_env_overrides(config, Config)
    config = filter_to_dataclass(config, Config)

    return Config(**config)
