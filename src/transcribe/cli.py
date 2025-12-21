"""Command-line interface and argument parsing."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import sounddevice as sd

from .config import Config, load_config


def list_audio_devices() -> None:
    """Print available audio input devices."""
    print("Available audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            default = " (default)" if device == sd.query_devices(kind="input") else ""
            print(f"  [{i}] {device['name']}{default}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz")
    print()


def parse_args(args: Optional[list] = None) -> tuple[Config, bool]:
    """
    Parse command-line arguments and return config.

    Returns:
        Tuple of (Config, should_exit) - should_exit is True for --list-devices
    """
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Voice transcription with speaker diarization. Outputs JSONL to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m transcribe                     # Use default config
  python -m transcribe --list-devices      # Show audio devices
  python -m transcribe --device-name=kitchen --whisper-model=base.en

Environment variables (override config.yaml):
  TRANSCRIBE_DEVICE_NAME, TRANSCRIBE_AUDIO_DEVICE, TRANSCRIBE_WHISPER_MODEL, etc.
""",
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        help="Device name for JSONL output (e.g., 'office', 'kitchen')",
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        help="Audio input device name or index",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny.en", "base.en", "small.en", "medium.en", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--speaker-similarity-threshold",
        type=float,
        help="Speaker matching threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        help="Voice activity detection threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["whisper", "sherpa"],
        help="Transcription backend: 'whisper' (faster-whisper) or 'sherpa' (sherpa-onnx)",
    )

    parsed = parser.parse_args(args)

    # Handle --list-devices
    if parsed.list_devices:
        list_audio_devices()
        return Config(), True

    # Build CLI overrides dict (only non-None values)
    cli_overrides = {}
    if parsed.device_name is not None:
        cli_overrides["device_name"] = parsed.device_name
    if parsed.audio_device is not None:
        cli_overrides["audio_device"] = parsed.audio_device
    if parsed.whisper_model is not None:
        cli_overrides["whisper_model"] = parsed.whisper_model
    if parsed.speaker_similarity_threshold is not None:
        cli_overrides["speaker_similarity_threshold"] = parsed.speaker_similarity_threshold
    if parsed.vad_threshold is not None:
        cli_overrides["vad_threshold"] = parsed.vad_threshold
    if parsed.backend is not None:
        cli_overrides["backend"] = parsed.backend

    # Load config with all overrides applied
    config = load_config(
        config_path=parsed.config,
        cli_args=cli_overrides,
    )

    return config, False
