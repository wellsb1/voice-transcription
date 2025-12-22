"""Voice Activity Detection using sherpa-onnx Silero VAD."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np


class VoiceActivityDetector:
    """
    Voice Activity Detection using sherpa-onnx Silero VAD model.

    Provides cross-platform VAD without requiring PyTorch.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        models_dir: str = "./models/sherpa",
    ):
        """
        Initialize VAD.

        Args:
            threshold: Detection threshold (0.0-1.0), higher = less sensitive
            sample_rate: Audio sample rate
            models_dir: Directory containing sherpa models
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.models_dir = Path(models_dir)
        self._vad = None
        self._available = None

    def _get_model_path(self) -> Path:
        """Get the VAD model path."""
        return self.models_dir / "silero_vad.onnx"

    def is_available(self) -> bool:
        """Check if the VAD model is available."""
        if self._available is None:
            self._available = self._get_model_path().exists()
        return self._available

    def _load_model(self) -> None:
        """Lazy-load VAD model."""
        if self._vad is not None:
            return

        if not self.is_available():
            print(
                f"VAD model not found at {self._get_model_path()}",
                file=sys.stderr,
            )
            print(
                "Download from: https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
                file=sys.stderr,
            )
            return

        try:
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(self._get_model_path())
            config.silero_vad.threshold = self.threshold
            config.silero_vad.min_silence_duration = 0.25
            config.silero_vad.min_speech_duration = 0.25
            config.sample_rate = self.sample_rate

            self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        except ImportError:
            self._available = False

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            True if speech is detected above threshold
        """
        self._load_model()

        if self._vad is None:
            return True  # Assume speech if VAD not available

        self._vad.reset()
        self._vad.accept_waveform(audio)
        self._vad.flush()

        return not self._vad.empty()

    def get_speech_segments(self, audio: np.ndarray) -> list[dict]:
        """
        Get timestamps of speech segments in audio.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            List of {"start": sample_idx, "end": sample_idx} dicts
        """
        self._load_model()

        if self._vad is None:
            return []

        self._vad.reset()
        self._vad.accept_waveform(audio)
        self._vad.flush()

        segments = []
        while not self._vad.empty():
            seg = self._vad.front()
            self._vad.pop()
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.start + len(seg.samples),
                }
            )

        return segments

    def get_speech_duration(self, audio: np.ndarray) -> float:
        """
        Get total seconds of speech in audio.

        Args:
            audio: Audio data as numpy array

        Returns:
            Total speech duration in seconds
        """
        segments = self.get_speech_segments(audio)
        if not segments:
            return 0.0

        total_samples = sum(seg["end"] - seg["start"] for seg in segments)
        return total_samples / self.sample_rate

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Get ratio of audio that contains speech.

        Args:
            audio: Audio data as numpy array

        Returns:
            Ratio from 0.0 (no speech) to 1.0 (all speech)
        """
        if len(audio) == 0:
            return 0.0

        speech_duration = self.get_speech_duration(audio)
        total_duration = len(audio) / self.sample_rate
        return speech_duration / total_duration
