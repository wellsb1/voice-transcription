"""Audio capture with VAD-aware chunking, running in a dedicated thread."""

import threading
import sys
from datetime import datetime
from typing import Optional, Callable
from collections import deque
import numpy as np
import sounddevice as sd
import torch

from .queue import DiskQueue


class AudioCapture:
    """
    Continuous audio capture with VAD-aware chunking.

    Uses a callback-based stream so audio capture never stops while
    VAD processing runs. Chunks on silence boundaries to avoid
    cutting words in half.
    """

    def __init__(
        self,
        queue: DiskQueue,
        sample_rate: int = 16000,
        device: Optional[str | int] = None,
        vad_threshold: float = 0.5,
        silence_duration: float = 0.5,
        min_chunk_duration: float = 1.0,
        max_chunk_duration: float = 30.0,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize audio capture with VAD-aware chunking.

        Args:
            queue: Disk queue to write audio chunks to
            sample_rate: Sample rate in Hz (default 16000)
            device: Audio device name or index (None for default)
            vad_threshold: VAD sensitivity (0.0-1.0, higher = less sensitive)
            silence_duration: Seconds of silence to trigger chunk boundary
            min_chunk_duration: Minimum chunk size before checking for silence
            max_chunk_duration: Maximum chunk size (force chunk even without silence)
            on_error: Callback for error handling
        """
        self.queue = queue
        self.sample_rate = sample_rate
        self.device = device
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.on_error = on_error

        self._raw_buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()

        self._stream: Optional[sd.InputStream] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._vad_model = None

    def _load_vad(self) -> None:
        """Lazy-load Silero VAD model."""
        if self._vad_model is not None:
            return

        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        with self._buffer_lock:
            self._raw_buffer.append(indata[:, 0].copy())

    def _is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio window contains speech."""
        self._load_vad()

        vad_chunk_size = 512

        for i in range(0, len(audio) - vad_chunk_size + 1, vad_chunk_size):
            chunk = audio[i : i + vad_chunk_size]
            audio_tensor = torch.tensor(chunk, dtype=torch.float32)
            speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()
            if speech_prob > self.vad_threshold:
                return True

        return False

    def _process_loop(self) -> None:
        """Process audio from buffer, detect silence boundaries, emit chunks."""
        window_duration = 0.1
        window_samples = int(window_duration * self.sample_rate)

        silence_samples = int(self.silence_duration * self.sample_rate)
        min_samples = int(self.min_chunk_duration * self.sample_rate)
        max_samples = int(self.max_chunk_duration * self.sample_rate)

        chunk_buffer: list[np.ndarray] = []
        chunk_samples = 0
        chunk_start_time = datetime.now()

        silence_counter = 0

        window_buffer = np.array([], dtype=np.float32)

        while not self._stop_event.is_set():
            new_audio: list[np.ndarray] = []
            with self._buffer_lock:
                while self._raw_buffer:
                    new_audio.append(self._raw_buffer.popleft())

            if new_audio:
                window_buffer = np.concatenate([window_buffer] + new_audio)

            while len(window_buffer) >= window_samples:
                window = window_buffer[:window_samples]
                window_buffer = window_buffer[window_samples:]

                chunk_buffer.append(window)
                chunk_samples += len(window)

                try:
                    has_speech = self._is_speech(window)
                except Exception as e:
                    if self.on_error:
                        self.on_error(e)
                    else:
                        print(f"VAD error: {e}", file=sys.stderr)
                    has_speech = True

                if has_speech:
                    silence_counter = 0
                else:
                    silence_counter += len(window)

                should_emit = False

                if chunk_samples >= max_samples:
                    should_emit = True
                elif chunk_samples >= min_samples and silence_counter >= silence_samples:
                    should_emit = True

                if should_emit and chunk_buffer:
                    full_audio = np.concatenate(chunk_buffer)
                    self.queue.put(full_audio, timestamp=chunk_start_time)

                    chunk_buffer = []
                    chunk_samples = 0
                    silence_counter = 0
                    chunk_start_time = datetime.now()

            if not new_audio:
                self._stop_event.wait(0.01)

        if chunk_buffer:
            full_audio = np.concatenate(chunk_buffer)
            self.queue.put(full_audio, timestamp=chunk_start_time)

    def start(self) -> None:
        """Start audio capture stream and processing thread."""
        if self._stream is not None:
            return

        self._stop_event.clear()
        self._raw_buffer.clear()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._audio_callback,
            blocksize=int(0.05 * self.sample_rate),
        )
        self._stream.start()

        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop capture stream and processing thread."""
        self._stop_event.set()

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._process_thread is not None:
            self._process_thread.join(timeout=timeout)
            self._process_thread = None

    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._stream is not None and self._stream.active


def get_device_info(device: Optional[str | int] = None) -> dict:
    """Get info about specified or default audio device."""
    if device is None:
        return sd.query_devices(kind="input")

    if isinstance(device, int):
        return sd.query_devices(device)

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if device.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return sd.query_devices(i)

    raise ValueError(f"Audio device not found: {device}")
