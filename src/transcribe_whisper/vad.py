"""Voice Activity Detection using Silero VAD."""

import numpy as np
import torch


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero VAD model.

    Filters audio chunks to identify which contain speech.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize VAD.

        Args:
            threshold: Detection threshold (0.0-1.0), higher = less sensitive
            sample_rate: Audio sample rate (must match audio input)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = None
        self._utils = None

    def _load_model(self) -> None:
        """Lazy-load Silero VAD model."""
        if self._model is not None:
            return

        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._VADIterator,
            self._collect_chunks,
        ) = self._utils

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            True if speech is detected above threshold
        """
        self._load_model()

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
        )

        return len(speech_timestamps) > 0

    def get_speech_segments(self, audio: np.ndarray) -> list[dict]:
        """
        Get timestamps of speech segments in audio.

        Args:
            audio: Audio data as numpy array (float32, mono)

        Returns:
            List of {"start": sample_idx, "end": sample_idx} dicts
        """
        self._load_model()

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
        )

        return speech_timestamps

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Get ratio of audio that contains speech.

        Args:
            audio: Audio data as numpy array

        Returns:
            Ratio from 0.0 (no speech) to 1.0 (all speech)
        """
        segments = self.get_speech_segments(audio)
        if not segments:
            return 0.0

        total_samples = len(audio)
        speech_samples = sum(seg["end"] - seg["start"] for seg in segments)
        return speech_samples / total_samples
