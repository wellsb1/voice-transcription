"""Main entry point for faster-whisper transcription backend."""

import signal
import sys
import time
import warnings
import math

# Suppress FutureWarning from speechbrain about torch.cuda.amp deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain")

from datetime import datetime

import numpy as np
import torch

from .config import load_config
from .capture import AudioCapture
from .asr import Transcriber
from .embeddings import SpeakerEmbedding
from .vad import VoiceActivityDetector
from .output import JsonlOutput
from .queue import DiskQueue

from transcribe_shared import SpeakerRegistry


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
        chunk = audio[i : i + chunk_size]
        audio_tensor = torch.tensor(chunk, dtype=torch.float32)
        prob = vad_model(audio_tensor, sample_rate).item()
        if prob > threshold:
            speech_chunks += 1

    return speech_chunks * (chunk_size / sample_rate)


def main() -> int:
    """Main entry point."""
    config = load_config()

    # Initialize queue
    queue = DiskQueue(config.queue_dir)

    # Clear stale audio from previous runs
    stale_count = queue.clear()
    if stale_count > 0:
        print(f"Cleared {stale_count} stale audio files from queue", file=sys.stderr)

    # Load models
    print(
        f"Loading Whisper model ({config.whisper_model}, compute_type={config.whisper_compute_type})...",
        file=sys.stderr,
    )
    transcriber = Transcriber(
        model_name=config.whisper_model,
        compute_type=config.whisper_compute_type,
    )
    transcriber.load()

    print("Loading speaker embedding model...", file=sys.stderr)
    embedder = SpeakerEmbedding()

    registry = SpeakerRegistry(
        similarity_threshold=config.speaker_similarity_threshold,
        timeout_seconds=config.speaker_inactivity_timeout,
    )

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

    # Start audio capture
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
    try:
        while not shutdown_requested:
            result = queue.get()
            if result is None:
                time.sleep(0.1)
                continue

            audio, filepath = result
            chunk_duration = len(audio) / config.sample_rate

            # Check audio energy
            rms_energy = np.sqrt(np.mean(audio**2))
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

            # Check speech duration
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

            # Extract timestamp from filename
            try:
                timestamp_str = filepath.stem
                chunk_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                chunk_timestamp = datetime.now()

            # Transcribe
            segments = transcriber.transcribe(audio)
            if not segments:
                queue.delete(filepath)
                continue

            # Identify speaker
            embedding = embedder.extract(audio, sample_rate=config.sample_rate)
            speaker_id = registry.identify(embedding, chunk_timestamp)

            # Compute confidence (not directly available with single embedding approach)
            confidence = 1.0

            # Output JSONL
            output.write_segments(
                segments=segments,
                speaker_id=speaker_id,
                confidence=confidence,
                rms=rms_energy,
                db=db_level,
                base_timestamp=chunk_timestamp,
            )

            queue.delete(filepath)

    finally:
        capture.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
