"""Disk-based audio queue for crash recovery."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np


class DiskQueue:
    """
    Simple disk-based queue for audio chunks.

    Audio is saved as .npy files with timestamp filenames.
    Provides crash recovery - audio isn't lost if process dies.
    """

    def __init__(self, queue_dir: str | Path):
        """
        Initialize disk queue.

        Args:
            queue_dir: Directory to store audio files
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def put(self, audio: np.ndarray, timestamp: Optional[datetime] = None) -> Path:
        """
        Save audio chunk to queue.

        Args:
            audio: Audio data as numpy array
            timestamp: Timestamp for filename (default: now)

        Returns:
            Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now()

        filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".npy"
        filepath = self.queue_dir / filename

        np.save(filepath, audio)
        return filepath

    def get(self) -> Optional[tuple[np.ndarray, Path]]:
        """
        Get oldest audio chunk from queue.

        Returns:
            Tuple of (audio, filepath) or None if queue is empty
        """
        files = sorted(self.queue_dir.glob("*.npy"))
        if not files:
            return None

        filepath = files[0]
        audio = np.load(filepath)
        return audio, filepath

    def delete(self, filepath: Path) -> None:
        """
        Delete a processed audio file.

        Args:
            filepath: Path to file to delete
        """
        if filepath.exists():
            filepath.unlink()

    def clear(self) -> int:
        """
        Clear all files from queue.

        Returns:
            Number of files deleted
        """
        files = list(self.queue_dir.glob("*.npy"))
        for f in files:
            f.unlink()
        return len(files)

    def size(self) -> int:
        """Return number of files in queue."""
        return len(list(self.queue_dir.glob("*.npy")))
