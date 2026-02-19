"""Audio capture from microphone."""

from typing import Optional, Callable
from datetime import datetime

import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Continuous audio capture from microphone.

    Runs in a separate thread via sounddevice, feeds audio to a callback.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        on_audio: Optional[Callable[[np.ndarray, datetime], None]] = None,
    ):
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
