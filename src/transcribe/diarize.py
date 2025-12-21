"""Speaker registry for consistent speaker identification across chunks."""

import json
from pathlib import Path
from typing import Optional
import numpy as np


class SpeakerRegistry:
    """
    Maintains consistent speaker IDs across audio chunks.

    Uses cosine similarity to match new embeddings against known speakers.
    Persists to JSON for crash recovery.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_embeddings_per_speaker: int = 20,
    ):
        """
        Initialize speaker registry.

        Args:
            similarity_threshold: Minimum similarity to match existing speaker (0.0-1.0)
            max_embeddings_per_speaker: Max embeddings to keep per speaker for averaging
        """
        self.similarity_threshold = similarity_threshold
        self.max_embeddings = max_embeddings_per_speaker
        self.speakers: dict[int, dict] = {}  # speaker_id -> {"embeddings": [...], "centroid": [...]}
        self.next_id = 0

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def identify_speaker(self, embedding: np.ndarray) -> tuple[int, float]:
        """
        Match embedding to existing speaker or create new one.

        Args:
            embedding: 192-dim speaker embedding

        Returns:
            Tuple of (speaker_id, confidence_score)
        """
        best_match = None
        best_score = -1.0

        # Compare to all known speakers
        for speaker_id, data in self.speakers.items():
            score = self.cosine_similarity(embedding, np.array(data["centroid"]))
            if score > best_score:
                best_score = score
                best_match = speaker_id

        # Decide: existing speaker or new one?
        if best_score >= self.similarity_threshold and best_match is not None:
            self._update_speaker(best_match, embedding)
            return best_match, best_score
        else:
            # Create new speaker
            new_id = self.next_id
            self.next_id += 1
            self.speakers[new_id] = {
                "embeddings": [embedding.tolist()],
                "centroid": embedding.tolist(),
            }
            return new_id, 1.0

    def _update_speaker(self, speaker_id: int, embedding: np.ndarray) -> None:
        """Update speaker's embedding history and centroid."""
        speaker = self.speakers[speaker_id]
        speaker["embeddings"].append(embedding.tolist())

        # Keep only last N embeddings (sliding window)
        if len(speaker["embeddings"]) > self.max_embeddings:
            speaker["embeddings"] = speaker["embeddings"][-self.max_embeddings:]

        # Recompute centroid
        embeddings_array = np.array(speaker["embeddings"])
        speaker["centroid"] = np.mean(embeddings_array, axis=0).tolist()

    def get_speaker_count(self) -> int:
        """Return number of known speakers."""
        return len(self.speakers)

    def save(self, filepath: str | Path) -> None:
        """
        Persist registry to JSON file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "speakers": self.speakers,
            "next_id": self.next_id,
            "similarity_threshold": self.similarity_threshold,
            "max_embeddings": self.max_embeddings,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str | Path) -> bool:
        """
        Load registry from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            True if file was loaded, False if file didn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return False

        with open(filepath) as f:
            data = json.load(f)

        self.speakers = {int(k): v for k, v in data.get("speakers", {}).items()}
        self.next_id = data.get("next_id", 0)
        # Don't override threshold/max_embeddings from file - use constructor values

        return True

    def clear(self) -> None:
        """Clear all known speakers."""
        self.speakers = {}
        self.next_id = 0
