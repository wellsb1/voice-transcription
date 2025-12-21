"""Backend abstraction for ASR and speaker identification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TranscriptSegment:
    """A transcribed segment of audio."""
    text: str
    start: float
    end: float


@dataclass
class SpeakerResult:
    """Speaker identification result."""
    speaker_id: int
    confidence: float


class Backend(ABC):
    """Abstract base class for transcription backends."""

    @abstractmethod
    def load(self) -> None:
        """Preload models (downloads if needed)."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[TranscriptSegment]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio

        Returns:
            List of transcript segments
        """
        pass

    @abstractmethod
    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> SpeakerResult:
        """
        Identify or assign speaker for audio.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio

        Returns:
            Speaker identification result
        """
        pass

    @abstractmethod
    def reset_speakers(self) -> None:
        """Reset speaker registry (e.g., after silence timeout)."""
        pass

    @abstractmethod
    def get_speaker_count(self) -> int:
        """Return number of known speakers."""
        pass
