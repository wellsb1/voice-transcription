"""Disk-backed audio queue for decoupling capture from processing."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import os


class DiskQueue:
    """
    Disk-backed queue for audio chunks.

    Audio chunks are stored as .npy files with timestamp-based names.
    This allows unbounded queueing without memory exhaustion and
    survives process crashes.
    """

    def __init__(self, queue_dir: str | Path):
        """
        Initialize disk queue.

        Args:
            queue_dir: Directory to store queue files
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self) -> str:
        """Generate timestamp-based filename for queue ordering."""
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S_%f") + ".npy"

    def put(self, audio: np.ndarray, timestamp: Optional[datetime] = None) -> Path:
        """
        Write audio chunk to queue.

        Args:
            audio: Audio data as numpy array
            timestamp: Optional timestamp (uses current time if not provided)

        Returns:
            Path to the created file
        """
        if timestamp:
            filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".npy"
        else:
            filename = self._generate_filename()

        filepath = self.queue_dir / filename
        np.save(filepath, audio)
        return filepath

    def peek(self) -> Optional[Path]:
        """
        Get path to oldest file without removing it.

        Returns:
            Path to oldest file, or None if queue is empty
        """
        files = self._list_files()
        return files[0] if files else None

    def get(self) -> Optional[tuple[np.ndarray, Path]]:
        """
        Read and return oldest audio chunk.

        Does NOT delete the file - call delete() after processing.

        Returns:
            Tuple of (audio_array, file_path), or None if queue is empty
        """
        filepath = self.peek()
        if filepath is None:
            return None

        try:
            audio = np.load(filepath)
            return audio, filepath
        except ValueError:
            # File not fully written yet (race condition on shutdown)
            return None

    def delete(self, filepath: Path) -> None:
        """
        Delete a processed file from the queue.

        Args:
            filepath: Path to the file to delete
        """
        if filepath.exists():
            filepath.unlink()

    def count(self) -> int:
        """Return number of files in queue."""
        return len(self._list_files())

    def _list_files(self) -> list[Path]:
        """List queue files sorted by name (oldest first)."""
        files = list(self.queue_dir.glob("*.npy"))
        files.sort()  # Timestamp-based names sort chronologically
        return files

    def clear(self) -> int:
        """
        Remove all files from queue.

        Returns:
            Number of files deleted
        """
        files = self._list_files()
        for f in files:
            f.unlink()
        return len(files)
