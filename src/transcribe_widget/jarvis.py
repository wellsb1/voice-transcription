"""Jarvis: always-on wake word listener and command dispatcher."""

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import numpy as np

from transcribe_shared.capture import AudioCapture
from transcribe_shared.trigger import TriggerDef, normalize_text, play_sound
from transcribe_shared.wakeword import WakeWordDetector


class JarvisMode(Enum):
    LISTENING = "listening"
    CAPTURE = "capture"
    DISABLED = "disabled"


@dataclass
class JarvisConfig:
    """Configuration for Jarvis listener."""

    audio_device: Optional[str] = None
    sample_rate: int = 16000

    # Wake word
    wakeword_models: list[str] = field(default_factory=list)
    wakeword_threshold: float = 0.5

    # Sounds
    sound_trigger: str = "/System/Library/Sounds/Tink.aiff"
    sound_done: str = "/System/Library/Sounds/Pop.aiff"
    sound_cancel: str = "/System/Library/Sounds/Funk.aiff"

    # Capture mode
    capture_max_duration: float = 30.0
    capture_silence_duration: float = 1.5
    capture_silence_threshold: float = 0.005  # RMS energy below this = silence

    # ASR for captured commands
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # Built-in commands
    builtin_commands: dict[str, list[str]] = field(default_factory=lambda: {
        "start_transcribing": ["start transcribing", "begin transcribing", "start recording"],
        "stop_transcribing": ["stop transcribing", "stop recording"],
    })

    # Default config for voice-triggered transcription start
    default_transcription_config: str = "m4"

    # Catch-all command for unmatched voice input
    catchall_command: Optional[str] = None

    # Named triggers (external programs launched by specific wake words)
    triggers: list[TriggerDef] = field(default_factory=list)


class CaptureASR:
    """Lightweight ASR for short captured voice commands. Lazy-loads mlx-whisper."""

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo"):
        self.model = model
        self._loaded = False

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe short audio to text."""
        import mlx_whisper

        if not self._loaded:
            print("Jarvis: Loading ASR model (first capture)...", file=sys.stderr)
            self._loaded = True

        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model, language="en")
        return result.get("text", "").strip()


class JarvisListener:
    """Always-on wake word listener and command dispatcher."""

    def __init__(
        self,
        config: JarvisConfig,
        on_builtin_command: Optional[Callable[[str], None]] = None,
        on_mode_change: Optional[Callable[[JarvisMode], None]] = None,
    ):
        self.config = config
        self.on_builtin_command = on_builtin_command
        self.on_mode_change = on_mode_change

        self.mode = JarvisMode.DISABLED
        self._capture: Optional[AudioCapture] = None
        self._wakeword: Optional[WakeWordDetector] = None
        self._asr: Optional[CaptureASR] = None

        # Capture state
        self._capture_buffer: list[np.ndarray] = []
        self._capture_lock = threading.Lock()
        self._silence_samples = 0
        self._capture_start: Optional[datetime] = None
        self._has_speech = False

        # Processing thread
        self._process_event = threading.Event()
        self._process_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        """Start audio capture and wake word detection."""
        if not self.config.wakeword_models:
            print("Jarvis: No wake word models configured", file=sys.stderr)
            return

        # Load wake word detector
        self._wakeword = WakeWordDetector(
            model_names=self.config.wakeword_models,
            threshold=self.config.wakeword_threshold,
        )
        self._wakeword.load()

        # Create ASR (lazy-loaded on first use)
        self._asr = CaptureASR(model=self.config.whisper_model)

        # Start processing thread
        self._shutdown.clear()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        # Start audio capture
        self._capture = AudioCapture(
            sample_rate=self.config.sample_rate,
            device=self.config.audio_device,
            on_audio=self._on_audio,
        )
        self._capture.start()

        self.mode = JarvisMode.LISTENING
        self._notify_mode_change()
        print("Jarvis: Listening...", file=sys.stderr)

    def stop(self) -> None:
        """Stop audio capture and wake word detection."""
        self._shutdown.set()
        self._process_event.set()  # Wake up processing thread

        if self._capture:
            self._capture.stop()
            self._capture = None

        if self._process_thread:
            self._process_thread.join(timeout=5)
            self._process_thread = None

        self.mode = JarvisMode.DISABLED
        self._notify_mode_change()
        print("Jarvis: Stopped", file=sys.stderr)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable Jarvis."""
        if enabled and self.mode == JarvisMode.DISABLED:
            self.start()
        elif not enabled and self.mode != JarvisMode.DISABLED:
            self.stop()

    def _on_audio(self, audio_chunk: np.ndarray, timestamp: datetime) -> None:
        """Audio callback — runs on PortAudio thread."""
        if self.mode == JarvisMode.LISTENING:
            # Run wake word detection
            if self._wakeword:
                detected = self._wakeword.process(audio_chunk)
                if detected:
                    self._enter_capture()

        elif self.mode == JarvisMode.CAPTURE:
            # Accumulate audio for capture
            with self._capture_lock:
                self._capture_buffer.append(audio_chunk.copy())

            # Check for silence to end capture
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < self.config.capture_silence_threshold:
                self._silence_samples += len(audio_chunk)
            else:
                self._silence_samples = 0
                self._has_speech = True

            silence_duration = self._silence_samples / self.config.sample_rate

            # Check timeout
            if self._capture_start:
                elapsed = (datetime.now() - self._capture_start).total_seconds()
                if elapsed > self.config.capture_max_duration:
                    print(f"Jarvis: Capture timed out ({elapsed:.0f}s)", file=sys.stderr)
                    play_sound(self.config.sound_cancel)
                    self._finish_capture_signal()
                    return

            # End capture after sufficient silence (only if we've heard speech)
            if self._has_speech and silence_duration >= self.config.capture_silence_duration:
                self._finish_capture_signal()

    def _enter_capture(self) -> None:
        """Switch to capture mode (called from audio thread)."""
        self.mode = JarvisMode.CAPTURE
        self._capture_buffer = []
        self._silence_samples = 0
        self._capture_start = datetime.now()
        self._has_speech = False

        if self._wakeword:
            self._wakeword.set_enabled(False)

        play_sound(self.config.sound_trigger)
        self._notify_mode_change()
        print("Jarvis: Capture mode", file=sys.stderr)

    def _finish_capture_signal(self) -> None:
        """Signal the processing thread to handle captured audio."""
        self._process_event.set()

    def _process_loop(self) -> None:
        """Background thread that processes captured audio."""
        while not self._shutdown.is_set():
            self._process_event.wait()
            self._process_event.clear()

            if self._shutdown.is_set():
                break

            if self.mode != JarvisMode.CAPTURE:
                continue

            # Get captured audio
            with self._capture_lock:
                if not self._capture_buffer:
                    self._resume_listening()
                    continue
                audio = np.concatenate(self._capture_buffer)
                self._capture_buffer = []

            # Need minimum audio to transcribe
            min_samples = int(self.config.sample_rate * 0.3)  # 300ms minimum
            if len(audio) < min_samples or not self._has_speech:
                print("Jarvis: No speech captured", file=sys.stderr)
                play_sound(self.config.sound_cancel)
                self._resume_listening()
                continue

            # Transcribe
            try:
                text = self._asr.transcribe(audio)
                print(f"Jarvis: Heard: '{text}'", file=sys.stderr)
            except Exception as e:
                print(f"Jarvis: ASR error: {e}", file=sys.stderr)
                play_sound(self.config.sound_cancel)
                self._resume_listening()
                continue

            if not text.strip():
                play_sound(self.config.sound_cancel)
                self._resume_listening()
                continue

            # Match and dispatch command
            matched = self._dispatch_command(text)
            self._resume_listening(icon_handled=matched)

    def _dispatch_command(self, text: str) -> bool:
        """Match captured text against commands and dispatch.

        Returns:
            True if a command was matched (caller should not restore icon).
        """
        normalized = normalize_text(text)

        # Check built-in commands
        for command_id, patterns in self.config.builtin_commands.items():
            for pattern in patterns:
                if normalize_text(pattern) in normalized:
                    print(f"Jarvis: Built-in command: {command_id}", file=sys.stderr)
                    play_sound(self.config.sound_done)
                    if self.on_builtin_command:
                        self.on_builtin_command(command_id)
                    return True

        # Check named triggers
        # (Named triggers use their own wake word models, handled separately)

        # Catch-all dispatcher
        if self.config.catchall_command:
            print(f"Jarvis: Dispatching to catchall: {self.config.catchall_command}", file=sys.stderr)
            self._dispatch_to_command(self.config.catchall_command, text)
            return True

        # No match
        print(f"Jarvis: No command matched for: '{text}'", file=sys.stderr)
        play_sound(self.config.sound_cancel)
        return False

    def _dispatch_to_command(self, command: str, text: str) -> None:
        """Spawn a subprocess and pipe text to its stdin."""
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(input=text + "\n", timeout=30)
            if stderr:
                print(f"Jarvis: Command stderr: {stderr.strip()}", file=sys.stderr)
            if stdout:
                print(f"Jarvis: Command output: {stdout.strip()}", file=sys.stderr)
            play_sound(self.config.sound_done)
        except subprocess.TimeoutExpired:
            print("Jarvis: Command timed out", file=sys.stderr)
            proc.kill()
            proc.wait()
            play_sound(self.config.sound_cancel)
        except Exception as e:
            print(f"Jarvis: Command error: {e}", file=sys.stderr)
            play_sound(self.config.sound_cancel)

    def _resume_listening(self, icon_handled: bool = False) -> None:
        """Return to listening mode.

        Args:
            icon_handled: If True, skip notifying mode change (caller handles icon).
        """
        self.mode = JarvisMode.LISTENING
        self._capture_buffer = []
        self._silence_samples = 0
        self._capture_start = None
        self._has_speech = False

        if self._wakeword:
            self._wakeword.set_enabled(True)

        if not icon_handled:
            self._notify_mode_change()

    def _notify_mode_change(self) -> None:
        """Notify the widget of a mode change."""
        if self.on_mode_change:
            self.on_mode_change(self.mode)
