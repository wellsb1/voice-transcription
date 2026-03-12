"""Audio capture with memory buffer."""

import threading
from datetime import datetime, timezone

import numpy as np

from transcribe_shared.capture import AudioCapture  # noqa: F401


class AudioBuffer:
    """
    Thread-safe memory buffer for audio.

    Accumulates audio chunks and provides batch extraction.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._buffer: list[np.ndarray] = []
        self._timestamps: list[datetime] = []
        self._lock = threading.Lock()

    def append(self, audio: np.ndarray, timestamp: datetime) -> None:
        """Add audio chunk to buffer."""
        with self._lock:
            self._buffer.append(audio)
            self._timestamps.append(timestamp)

    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            total_samples = sum(len(chunk) for chunk in self._buffer)
            return total_samples / self.sample_rate

    def get_audio(self) -> tuple[np.ndarray, datetime]:
        """
        Extract all buffered audio and clear buffer.

        Returns:
            Tuple of (concatenated audio array, timestamp of first chunk)
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32), datetime.now(tz=timezone.utc)

            audio = np.concatenate(self._buffer)
            first_timestamp = self._timestamps[0]

            self._buffer.clear()
            self._timestamps.clear()

            return audio, first_timestamp

    def clear(self) -> None:
        """Clear buffer without returning data."""
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()
