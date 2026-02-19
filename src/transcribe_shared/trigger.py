"""Shared trigger definitions and utilities."""

import re
import subprocess
from dataclasses import dataclass


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def play_sound(path: str) -> None:
    """Play a sound file non-blocking via afplay."""
    try:
        subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


@dataclass
class TriggerDef:
    """Definition of a voice trigger."""

    name: str
    wakeword_model: str  # openwakeword model name (e.g., "hey_jarvis_v0.1")
    stop_word: str
    command: str
    max_duration: float = 60.0
