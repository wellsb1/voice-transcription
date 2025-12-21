"""ASR using mlx-whisper for Apple Silicon."""

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TranscriptSegment:
    """A transcribed segment of audio."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds


class Transcriber:
    """
    Speech-to-text using mlx-whisper.

    Optimized for Apple Silicon using MLX framework.
    """

    def __init__(
        self,
        model: str = "mlx-community/whisper-large-v3-turbo",
    ):
        """
        Initialize transcriber.

        Args:
            model: Model name/path for mlx-whisper (HuggingFace format)
        """
        self.model_name = model
        self._loaded = False

    def load(self) -> None:
        """Preload the model (triggers download if needed)."""
        if self._loaded:
            return

        print(f"Loading Whisper model ({self.model_name})...", file=sys.stderr)

        # Import here to defer loading until needed
        import mlx_whisper

        # Trigger model download/load by doing a dummy transcription
        # mlx_whisper loads on first use
        self._loaded = True

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[TranscriptSegment]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate (should be 16kHz)

        Returns:
            List of TranscriptSegment with text and timestamps
        """
        import mlx_whisper

        if len(audio) == 0:
            return []

        # mlx_whisper.transcribe expects audio path or numpy array
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language="en",
            word_timestamps=True,
        )

        segments = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                    )
                )

        return segments

    def transcribe_simple(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio to plain text (no timestamps).

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            Transcribed text
        """
        import mlx_whisper

        if len(audio) == 0:
            return ""

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language="en",
        )

        return result.get("text", "").strip()
