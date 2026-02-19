"""Audio capture from system tap (mic + system audio via Core Audio process tap)."""

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Default binary location relative to project root
_DEFAULT_BINARY = Path(__file__).resolve().parents[2] / "bin" / "audio-tap"


class AudioTapCapture:
    """
    Captures system audio + microphone via the audio-tap Swift CLI.

    Same interface as AudioCapture: on_audio callback, start(), stop().
    The Swift tool outputs raw PCM float32 mono to stdout which we read in chunks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        exclude_apps: Optional[list[str]] = None,
        mic_device: Optional[str] = None,
        no_mic: bool = False,
        binary_path: Optional[str] = None,
        on_audio: Optional[Callable[[np.ndarray, datetime], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.exclude_apps = exclude_apps or []
        self.mic_device = mic_device
        self.no_mic = no_mic
        self.binary_path = Path(binary_path) if binary_path else _DEFAULT_BINARY
        self.on_audio = on_audio

        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the audio-tap subprocess and begin reading audio."""
        if self._running:
            return

        if not self.binary_path.exists():
            raise FileNotFoundError(
                f"audio-tap binary not found at {self.binary_path}. "
                f"Build it with: cd tools/audio-tap && make build"
            )

        cmd = [str(self.binary_path), "--sample-rate", str(self.sample_rate)]
        if self.exclude_apps:
            cmd.extend(["--exclude", ",".join(self.exclude_apps)])
        if self.no_mic:
            cmd.append("--no-mic")
        elif self.mic_device:
            cmd.extend(["--mic", self.mic_device])

        self._running = True
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        self._reader_thread = threading.Thread(target=self._read_audio, daemon=True)
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def stop(self) -> None:
        """Stop the audio-tap subprocess."""
        self._running = False
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def _read_audio(self) -> None:
        """Reader thread: read PCM float32 chunks from subprocess stdout."""
        chunk_samples = int(self.sample_rate * 0.1)  # 100ms chunks
        chunk_bytes = chunk_samples * 4  # float32 = 4 bytes

        proc = self._process
        if proc is None or proc.stdout is None:
            return

        while self._running:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break

            audio = np.frombuffer(data, dtype=np.float32)
            if len(audio) > 0 and self.on_audio is not None:
                self.on_audio(audio, datetime.now())

    def _read_stderr(self) -> None:
        """Forward subprocess stderr to our stderr."""
        proc = self._process
        if proc is None or proc.stderr is None:
            return

        for line in proc.stderr:
            if not self._running:
                break
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
