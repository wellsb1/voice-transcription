"""Batch detection using Silero VAD for silence-based boundaries."""

import sys
from typing import Optional

import numpy as np
import torch


class BatchDetector:
    """
    Detects batch boundaries using Silero VAD.

    Triggers a batch when:
    1. Silence is detected (silence_duration seconds) AND min_batch_duration reached
    2. OR max_batch_duration is reached (regardless of silence)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        min_batch_duration: float = 30.0,
        max_batch_duration: float = 60.0,
        silence_duration: float = 0.5,
        vad_threshold: float = 0.5,
    ):
        """
        Initialize batch detector.

        Args:
            sample_rate: Audio sample rate
            min_batch_duration: Minimum batch size before allowing silence cut
            max_batch_duration: Force batch regardless of silence
            silence_duration: Seconds of silence to trigger batch
            vad_threshold: VAD probability threshold (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.min_batch_duration = min_batch_duration
        self.max_batch_duration = max_batch_duration
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold

        self._vad_model = None
        self._silence_samples = 0
        self._silence_samples_threshold = int(silence_duration * sample_rate)

    def _load_vad(self) -> None:
        """Lazy-load Silero VAD model."""
        if self._vad_model is not None:
            return

        print("Loading Silero VAD model...", file=sys.stderr)
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    def check_batch_ready(
        self,
        audio_chunk: np.ndarray,
        total_duration: float,
    ) -> bool:
        """
        Check if a batch boundary should be triggered.

        Args:
            audio_chunk: Latest audio chunk to analyze
            total_duration: Total buffered audio duration in seconds

        Returns:
            True if batch should be processed now
        """
        self._load_vad()

        # Force batch at max duration
        if total_duration >= self.max_batch_duration:
            self._silence_samples = 0
            return True

        # Not enough audio yet
        if total_duration < self.min_batch_duration:
            return False

        # Check for speech in this chunk using Silero VAD
        # Silero expects 512-sample chunks at 16kHz
        chunk_size = 512
        is_speech = False

        for i in range(0, len(audio_chunk) - chunk_size + 1, chunk_size):
            chunk = audio_chunk[i : i + chunk_size]
            audio_tensor = torch.tensor(chunk, dtype=torch.float32)
            prob = self._vad_model(audio_tensor, self.sample_rate).item()
            if prob > self.vad_threshold:
                is_speech = True
                break

        if is_speech:
            # Reset silence counter on speech
            self._silence_samples = 0
            return False
        else:
            # Accumulate silence
            self._silence_samples += len(audio_chunk)

            # Check if silence threshold reached
            if self._silence_samples >= self._silence_samples_threshold:
                self._silence_samples = 0
                return True

        return False

    def reset(self) -> None:
        """Reset silence counter."""
        self._silence_samples = 0
