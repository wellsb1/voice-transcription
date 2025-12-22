"""Speaker registry for consistent speaker identification across batches."""

from datetime import datetime
from typing import Optional
import numpy as np


class SpeakerRegistry:
    """
    Tracks speakers across audio batches using embedding similarity with timeout expiration.

    Uses cosine similarity to match new embeddings against known speakers.
    Speakers are expired after a configurable inactivity timeout.

    Note: This is a code-only utility - each backend creates its own instance.
    No state is shared between backends.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        timeout_seconds: float = 1800.0,
    ):
        """
        Initialize speaker registry.

        Args:
            similarity_threshold: Minimum cosine similarity to match existing speaker (0.0-1.0)
            timeout_seconds: Seconds of inactivity before speaker ID expires (default 30 min)
        """
        self.threshold = similarity_threshold
        self.timeout = timeout_seconds
        self.speakers: dict[int, dict] = {}  # id -> {embedding, last_seen}
        self.next_id = 0

    def identify(self, embedding: np.ndarray, timestamp: datetime) -> int:
        """
        Match embedding to existing speaker or create new one.

        Args:
            embedding: Speaker embedding vector (e.g., 192-dim or 512-dim)
            timestamp: Current timestamp for tracking speaker activity

        Returns:
            Persistent speaker ID (int)
        """
        self._expire_inactive(timestamp)

        # Find best match using cosine similarity
        best_id, best_score = None, -1.0
        for sid, data in self.speakers.items():
            score = self._cosine_similarity(embedding, data["embedding"])
            if score > best_score:
                best_id, best_score = sid, score

        # Match existing speaker if similarity exceeds threshold
        if best_score >= self.threshold and best_id is not None:
            self.speakers[best_id]["last_seen"] = timestamp
            return best_id

        # Create new speaker
        new_id = self.next_id
        self.next_id += 1
        self.speakers[new_id] = {"embedding": embedding, "last_seen": timestamp}
        return new_id

    def get_speaker_count(self) -> int:
        """Return number of tracked speakers."""
        return len(self.speakers)

    def clear(self) -> None:
        """Clear all tracked speakers."""
        self.speakers = {}
        self.next_id = 0

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
