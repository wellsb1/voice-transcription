"""Processing pipeline: diarization + transcription."""

import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from .asr import Transcriber
from .diarize import Diarizer, SpeakerSegment


@dataclass
class DiarizedTranscript:
    """A transcribed segment with speaker attribution."""

    speaker: str  # Speaker label
    text: str  # Transcribed text
    start: float  # Start time relative to batch
    end: float  # End time relative to batch
    batch_timestamp: datetime  # When the batch started (wall clock)


class SpeakerRegistry:
    """Tracks speakers across batches using embeddings with timeout expiration."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        timeout_seconds: float = 1800.0,
    ):
        self.threshold = similarity_threshold
        self.timeout = timeout_seconds
        self.speakers: dict[int, dict] = {}  # id → {embedding, last_seen}
        self.next_id = 0

    def identify(self, embedding: np.ndarray, timestamp: datetime) -> int:
        """Match embedding to existing speaker or create new one."""
        self._expire_inactive(timestamp)

        # Find best match using cosine similarity
        best_id, best_score = None, -1.0
        for sid, data in self.speakers.items():
            score = self._cosine_similarity(embedding, data["embedding"])
            if score > best_score:
                best_id, best_score = sid, score

        # Match existing or create new
        if best_score >= self.threshold and best_id is not None:
            self.speakers[best_id]["last_seen"] = timestamp
            return best_id

        new_id = self.next_id
        self.next_id += 1
        self.speakers[new_id] = {"embedding": embedding, "last_seen": timestamp}
        return new_id

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _expire_inactive(self, now: datetime) -> None:
        """Remove speakers inactive for longer than timeout."""
        expired = [
            sid
            for sid, data in self.speakers.items()
            if (now - data["last_seen"]).total_seconds() > self.timeout
        ]
        for sid in expired:
            del self.speakers[sid]


class Pipeline:
    """
    Orchestrates diarization and transcription.

    Flow:
    1. Diarize audio → get speaker segments
    2. For each segment, transcribe with mlx-whisper
    3. Return list of DiarizedTranscript
    """

    def __init__(
        self,
        whisper_model: str = "mlx-community/whisper-large-v3-turbo",
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        huggingface_token: Optional[str] = None,
        speaker_similarity_threshold: float = 0.75,
        speaker_inactivity_timeout: float = 1800.0,
    ):
        """
        Initialize pipeline.

        Args:
            whisper_model: Model for mlx-whisper
            diarization_model: Model for pyannote
            huggingface_token: HuggingFace token for pyannote
            speaker_similarity_threshold: Cosine similarity threshold for speaker matching
            speaker_inactivity_timeout: Seconds of inactivity before speaker ID expires
        """
        self.transcriber = Transcriber(model=whisper_model)
        self.diarizer = Diarizer(
            model=diarization_model,
            huggingface_token=huggingface_token,
        )
        self.speaker_registry = SpeakerRegistry(
            similarity_threshold=speaker_similarity_threshold,
            timeout_seconds=speaker_inactivity_timeout,
        )

    def load(self) -> None:
        """Preload all models."""
        self.diarizer.load()
        self.transcriber.load()

    def process(
        self,
        audio: np.ndarray,
        batch_timestamp: datetime,
        sample_rate: int = 16000,
    ) -> list[DiarizedTranscript]:
        """
        Process an audio batch: diarize then transcribe.

        Args:
            audio: Audio data as numpy array (float32, mono)
            batch_timestamp: When this batch started (wall clock time)
            sample_rate: Audio sample rate

        Returns:
            List of DiarizedTranscript with speaker, text, and timestamps
        """
        if len(audio) == 0:
            return []

        duration = len(audio) / sample_rate
        print(f"Processing {duration:.1f}s batch...", file=sys.stderr)

        # Step 1: Diarize
        print("  Diarizing...", file=sys.stderr)
        speaker_segments, embeddings = self.diarizer.diarize(audio, sample_rate)

        if not speaker_segments:
            print("  No speakers detected", file=sys.stderr)
            return []

        print(f"  Found {len(speaker_segments)} speaker segments", file=sys.stderr)

        # Step 2: Transcribe each segment
        results = []
        for seg in speaker_segments:
            # Skip very short segments
            if seg.end - seg.start < 0.3:
                continue

            # Extract audio for this segment
            segment_audio = self.diarizer.get_speaker_audio(audio, seg, sample_rate)

            if len(segment_audio) == 0:
                continue

            # Get persistent speaker ID from registry
            embedding = embeddings.get(seg.speaker)
            if embedding is not None:
                persistent_id = self.speaker_registry.identify(embedding, batch_timestamp)
                speaker_label = f"SPEAKER_{persistent_id:02d}"
            else:
                speaker_label = seg.speaker  # Fallback to batch-local label

            # Transcribe
            text = self.transcriber.transcribe_simple(segment_audio, sample_rate)

            if text:
                results.append(
                    DiarizedTranscript(
                        speaker=speaker_label,
                        text=text,
                        start=seg.start,
                        end=seg.end,
                        batch_timestamp=batch_timestamp,
                    )
                )

        print(f"  Transcribed {len(results)} segments", file=sys.stderr)
        return results
