"""Audio capture with memory buffer."""

import threading
import queue
from typing import Optional, Callable
from datetime import datetime

import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Continuous audio capture to memory buffer.

    Runs in a separate thread, feeds audio to a callback for batch detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        on_audio: Optional[Callable[[np.ndarray, datetime], None]] = None,
    ):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate (default 16kHz for Whisper)
            device: Audio device name/index, None for system default
            on_audio: Callback receiving (audio_chunk, timestamp) for each chunk
        """
        self.sample_rate = sample_rate
        self.device = device
        self.on_audio = on_audio

        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice callback - receives audio chunks."""
        if status:
            print(f"Audio capture status: {status}")

        if self.on_audio is not None:
            # Copy data and send to callback
            audio = indata[:, 0].copy().astype(np.float32)
            timestamp = datetime.now()
            self.on_audio(audio, timestamp)

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms chunks
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


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
                return np.array([], dtype=np.float32), datetime.now()

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
