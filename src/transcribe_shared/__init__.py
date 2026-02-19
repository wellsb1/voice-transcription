"""Shared utilities for transcription backends."""

from .config import load_yaml, apply_env_overrides
from .speaker_registry import SpeakerRegistry
from .utterance import Utterance, aggregate_utterances, aggregate_consecutive
from .capture import AudioCapture
from .tap_capture import AudioTapCapture
from .wakeword import WakeWordDetector
from .trigger import TriggerDef, play_sound, normalize_text

__all__ = [
    "load_yaml",
    "apply_env_overrides",
    "SpeakerRegistry",
    "Utterance",
    "aggregate_utterances",
    "aggregate_consecutive",
    "AudioCapture",
    "AudioTapCapture",
    "WakeWordDetector",
    "TriggerDef",
    "play_sound",
    "normalize_text",
]
