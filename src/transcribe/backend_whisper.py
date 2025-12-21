"""Faster-Whisper backend for ASR and speaker identification."""

import sys
from typing import Optional
import numpy as np

from .backend import Backend, TranscriptSegment, SpeakerResult
from .transcribe import Transcriber
from .embeddings import SpeakerEmbedding
from .diarize import SpeakerRegistry


class FasterWhisperBackend(Backend):
    """
    Backend using faster-whisper for ASR and SpeechBrain ECAPA-TDNN for speaker ID.

    This is the original implementation, good for CPU-based inference.
    """

    def __init__(
        self,
        whisper_model: str = "medium.en",
        speaker_similarity_threshold: float = 0.35,
        whisper_compute_type: str = "int8",
    ):
        """
        Initialize the faster-whisper backend.

        Args:
            whisper_model: Whisper model name (tiny.en, base.en, small.en, medium.en)
            speaker_similarity_threshold: Threshold for speaker matching (0.0-1.0)
            whisper_compute_type: Compute type for faster-whisper (int8, float16, int8_float16)
        """
        self.whisper_model = whisper_model
        self.speaker_similarity_threshold = speaker_similarity_threshold
        self.whisper_compute_type = whisper_compute_type

        self._transcriber: Optional[Transcriber] = None
        self._embedder: Optional[SpeakerEmbedding] = None
        self._registry: Optional[SpeakerRegistry] = None

    def load(self) -> None:
        """Preload models."""
        print(f"Loading Whisper model ({self.whisper_model}, compute_type={self.whisper_compute_type})...", file=sys.stderr)
        self._transcriber = Transcriber(
            model_name=self.whisper_model,
            compute_type=self.whisper_compute_type,
        )
        self._transcriber.load()

        print("Loading speaker embedding model...", file=sys.stderr)
        self._embedder = SpeakerEmbedding()
        # Embedder loads lazily on first use

        self._registry = SpeakerRegistry(
            similarity_threshold=self.speaker_similarity_threshold,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[TranscriptSegment]:
        """Transcribe audio using faster-whisper."""
        if self._transcriber is None:
            self.load()

        segments = self._transcriber.transcribe(audio)
        return [
            TranscriptSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
            )
            for seg in segments
        ]

    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> SpeakerResult:
        """Identify speaker using ECAPA-TDNN embeddings."""
        if self._embedder is None or self._registry is None:
            self.load()

        embedding = self._embedder.extract(audio, sample_rate=sample_rate)
        speaker_id, confidence = self._registry.identify_speaker(embedding)

        return SpeakerResult(speaker_id=speaker_id, confidence=confidence)

    def reset_speakers(self) -> None:
        """Reset speaker registry."""
        if self._registry is not None:
            self._registry.clear()

    def get_speaker_count(self) -> int:
        """Return number of known speakers."""
        if self._registry is None:
            return 0
        return self._registry.get_speaker_count()
