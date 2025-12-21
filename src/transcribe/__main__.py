"""Main entry point for the transcription service."""

import signal
import sys
import time
import warnings

# Suppress FutureWarning from speechbrain about torch.cuda.amp deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain")
from datetime import datetime

import math

import numpy as np
import torch

from .backend import Backend
from .cli import parse_args
from .capture import AudioCapture
from .output import JsonlOutput
from .queue import DiskQueue


def create_backend(config) -> Backend:
    """Create the appropriate backend based on config."""
    if config.backend == "sherpa":
        from .backend_sherpa import SherpaBackend
        return SherpaBackend(
            whisper_model=config.whisper_model,
            speaker_similarity_threshold=config.speaker_similarity_threshold,
            sherpa_use_int8=config.sherpa_use_int8,
            sherpa_provider=config.sherpa_provider,
            sherpa_num_threads=config.sherpa_num_threads,
        )
    else:
        from .backend_whisper import FasterWhisperBackend
        return FasterWhisperBackend(
            whisper_model=config.whisper_model,
            speaker_similarity_threshold=config.speaker_similarity_threshold,
            whisper_compute_type=config.whisper_compute_type,
        )


def get_speech_duration(
    audio: np.ndarray,
    vad_model,
    sample_rate: int,
    threshold: float,
) -> float:
    """Return seconds of speech in audio."""
    chunk_size = 512  # Silero VAD requirement at 16kHz
    speech_chunks = 0

    for i in range(0, len(audio) - chunk_size + 1, chunk_size):
        chunk = audio[i:i + chunk_size]
        audio_tensor = torch.tensor(chunk, dtype=torch.float32)
        prob = vad_model(audio_tensor, sample_rate).item()
        if prob > threshold:
            speech_chunks += 1

    # Each chunk is 512 samples at 16kHz = 0.032 seconds
    return speech_chunks * (chunk_size / sample_rate)


def main() -> int:
    """Main entry point."""
    # Parse config
    config, should_exit = parse_args()
    if should_exit:
        return 0

    # Initialize components
    queue = DiskQueue(config.queue_dir)

    # Clear stale audio from previous runs to avoid hallucinations
    stale_count = queue.clear()
    if stale_count > 0:
        print(f"Cleared {stale_count} stale audio files from queue", file=sys.stderr)

    # Create and load backend
    print(f"Using backend: {config.backend}", file=sys.stderr)
    backend = create_backend(config)
    backend.load()

    output = JsonlOutput(device_name=config.device_name)

    # Load VAD model for speech ratio check
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )

    # Set up signal handler for graceful shutdown
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\nShutdown requested, finishing current chunk...", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start audio capture thread with VAD-aware chunking
    capture = AudioCapture(
        queue=queue,
        sample_rate=config.sample_rate,
        device=config.audio_device,
        vad_threshold=config.vad_threshold,
        silence_duration=config.silence_duration,
        min_chunk_duration=config.min_chunk_duration,
        max_chunk_duration=config.max_chunk_duration,
    )
    capture.start()
    print(
        f"Started audio capture (device={config.audio_device or 'default'}, "
        f"VAD-aware chunking: {config.min_chunk_duration}-{config.max_chunk_duration}s, "
        f"silence threshold: {config.silence_duration}s)",
        file=sys.stderr,
    )

    # Process queue
    last_speech_time = time.time()
    try:
        while not shutdown_requested:
            # Get next audio chunk
            result = queue.get()
            if result is None:
                # Queue empty, wait a bit
                time.sleep(0.1)

                # Check for speaker reset timeout
                if config.speaker_reset_timeout > 0:
                    silence_duration = time.time() - last_speech_time
                    if silence_duration >= config.speaker_reset_timeout:
                        if backend.get_speaker_count() > 0:
                            print(
                                f"Resetting speakers after {silence_duration/60:.0f}min silence",
                                file=sys.stderr,
                            )
                            backend.reset_speakers()
                            last_speech_time = time.time()  # Reset timer

                continue

            audio, filepath = result
            chunk_duration = len(audio) / config.sample_rate

            # Check audio energy - skip quiet noise
            rms_energy = np.sqrt(np.mean(audio ** 2))
            db_level = 20 * math.log10(max(rms_energy, 1e-6))
            db_level = max(db_level, -60.0)

            if rms_energy < config.min_audio_energy:
                print(
                    f"[SKIP energy] {chunk_duration:.1f}s chunk, "
                    f"rms={rms_energy:.4f}, db={db_level:.1f}",
                    file=sys.stderr,
                )
                queue.delete(filepath)
                continue

            # Check speech duration - skip chunks with too little speech
            speech_seconds = get_speech_duration(
                audio, vad_model, config.sample_rate, config.vad_threshold
            )
            if speech_seconds < config.min_speech_duration:
                print(
                    f"[SKIP speech] {chunk_duration:.1f}s chunk, "
                    f"rms={rms_energy:.4f}, db={db_level:.1f}, speech={speech_seconds:.1f}s",
                    file=sys.stderr,
                )
                queue.delete(filepath)
                continue

            print(
                f"[PROCESS] {chunk_duration:.1f}s chunk, "
                f"rms={rms_energy:.4f}, db={db_level:.1f}, speech={speech_seconds:.1f}s",
                file=sys.stderr,
            )

            # Extract timestamp from filename (YYYYMMDD_HHMMSS_ffffff.npy)
            try:
                timestamp_str = filepath.stem
                chunk_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                chunk_timestamp = datetime.now()

            # Transcribe
            segments = backend.transcribe(audio, sample_rate=config.sample_rate)
            if not segments:
                queue.delete(filepath)
                continue

            # Identify speaker
            speaker_result = backend.identify_speaker(audio, sample_rate=config.sample_rate)

            # Output JSONL
            output.write_segments(
                segments=[{"text": seg.text, "start": seg.start, "end": seg.end} for seg in segments],
                speaker_id=speaker_result.speaker_id,
                confidence=speaker_result.confidence,
                rms=rms_energy,
                db=db_level,
                base_timestamp=chunk_timestamp,
            )

            # Remove processed file
            queue.delete(filepath)
            last_speech_time = time.time()

    finally:
        # Cleanup
        capture.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
