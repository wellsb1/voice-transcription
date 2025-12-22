"""Shared utilities for transcription backends."""

from .config import load_yaml, apply_env_overrides
from .speaker_registry import SpeakerRegistry
from .utterance import Utterance, aggregate_utterances, aggregate_consecutive

__all__ = [
    "load_yaml",
    "apply_env_overrides",
    "SpeakerRegistry",
    "Utterance",
    "aggregate_utterances",
    "aggregate_consecutive",
]
