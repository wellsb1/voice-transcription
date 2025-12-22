"""Main entry point for M4 transcription service."""

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .batch import BatchDetector
from .capture import AudioBuffer, AudioCapture
from .config import Config, load_config
from .output import JsonlOutput
from .pipeline import Pipeline


def parse_args() -> tuple[Config, bool, bool]:
    """Parse command-line arguments.

    Returns:
        Tuple of (config, should_exit, debug_mode)
    """
    parser = argparse.ArgumentParser(
        prog="transcribe_m4",
        description="M4-optimized transcription with speaker diarization. Outputs JSONL to stdout.",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.m4.yaml (default: ./config.m4.yaml)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        help="Device name for JSONL output",
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        help="Audio input device name or index",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: play back audio chunks before processing",
    )

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        devices = sd.query_devices()
        default_input = sd.query_devices(kind="input")
        default_output = sd.query_devices(kind="output")

        print("Available audio INPUT devices:")
        print("-" * 60)
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                default = " (default)" if device == default_input else ""
                print(f"  [{i}] {device['name']}{default}")

        print("\nAvailable audio OUTPUT devices:")
        print("-" * 60)
        for i, device in enumerate(devices):
            if device["max_output_channels"] > 0:
                default = " (default)" if device == default_output else ""
                print(f"  [{i}] {device['name']}{default}")

        return Config(), True, False

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.device_name:
        config.device_name = args.device_name
    if args.audio_device:
        config.audio_device = args.audio_device

    return config, False, args.debug


def main() -> int:
    """Main entry point."""
    config, should_exit, debug_mode = parse_args()
    if should_exit:
        return 0

    if debug_mode:
        print("DEBUG MODE: Will play back audio before processing", file=sys.stderr)

    # Initialize components
    print(f"Initializing M4 transcription...", file=sys.stderr)
    print(f"  Whisper model: {config.whisper_model}", file=sys.stderr)
    print(f"  Diarization model: {config.diarization_model}", file=sys.stderr)
    print(f"  Batch: {config.min_batch_duration}-{config.max_batch_duration}s", file=sys.stderr)
    print(f"  Speaker timeout: {config.speaker_inactivity_timeout}s", file=sys.stderr)

    # Create pipeline and preload models
    pipeline = Pipeline(
        whisper_model=config.whisper_model,
        diarization_model=config.diarization_model,
        huggingface_token=config.huggingface_token,
        speaker_similarity_threshold=config.speaker_similarity_threshold,
        speaker_inactivity_timeout=config.speaker_inactivity_timeout,
        ignore_words=config.ignore_words,
    )
    pipeline.load()

    # Create batch detector
    batch_detector = BatchDetector(
        sample_rate=config.sample_rate,
        min_batch_duration=config.min_batch_duration,
        max_batch_duration=config.max_batch_duration,
        silence_duration=config.silence_duration,
        vad_threshold=config.vad_threshold,
    )

    # Create output
    output = JsonlOutput(device_name=config.device_name)

    # Create audio buffer
    audio_buffer = AudioBuffer(sample_rate=config.sample_rate)

    # Batch ready event
    batch_ready = threading.Event()
    shutdown_requested = threading.Event()

    def on_audio(audio_chunk, timestamp):
        """Handle incoming audio from capture."""
        audio_buffer.append(audio_chunk, timestamp)

        # Check if batch is ready
        duration = audio_buffer.get_duration()
        if batch_detector.check_batch_ready(audio_chunk, duration):
            batch_ready.set()

    # Create and start audio capture
    capture = AudioCapture(
        sample_rate=config.sample_rate,
        device=config.audio_device,
        on_audio=on_audio,
    )

    # Signal handler
    def handle_signal(signum, frame):
        print("\nShutdown requested...", file=sys.stderr)
        shutdown_requested.set()
        batch_ready.set()  # Wake up main loop

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start capture
    capture.start()
    print(
        f"Started audio capture (device={config.audio_device or 'default'})",
        file=sys.stderr,
    )
    print("Listening...", file=sys.stderr)

    # Main processing loop
    try:
        while not shutdown_requested.is_set():
            # Wait for batch ready or shutdown
            batch_ready.wait(timeout=1.0)

            if shutdown_requested.is_set():
                break

            if not batch_ready.is_set():
                continue

            batch_ready.clear()

            # Get audio from buffer
            audio, timestamp = audio_buffer.get_audio()

            if len(audio) == 0:
                continue

            # Debug: play back audio before processing
            if debug_mode:
                import sounddevice as sd
                duration = len(audio) / config.sample_rate
                out_dev = config.audio_output_device or "default"
                print(f"\n[DEBUG] Playing {duration:.1f}s of audio on {out_dev}...", file=sys.stderr)
                sd.play(audio, config.sample_rate, device=config.audio_output_device)
                sd.wait()
                print("[DEBUG] Playback complete. Processing...", file=sys.stderr)

            # Process batch
            try:
                transcripts = pipeline.process(
                    audio,
                    batch_timestamp=timestamp,
                    sample_rate=config.sample_rate,
                )

                # Output results
                output.write_batch(transcripts)

            except Exception as e:
                print(f"Error processing batch: {e}", file=sys.stderr)

    finally:
        capture.stop()
        print("Stopped.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
