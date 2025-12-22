"""Main entry point for sherpa-onnx transcription backend."""

import signal
import sys
import time
import math

from datetime import datetime

import numpy as np

from .config import load_config
from .capture import AudioCapture
from .asr import Transcriber
from .embeddings import SpeakerEmbedding
from .vad import VoiceActivityDetector
from .output import JsonlOutput
from .queue import DiskQueue

from transcribe_shared import SpeakerRegistry


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
        f"Loading Sherpa Whisper model ({config.whisper_model}, "
        f"int8={config.sherpa_use_int8}, provider={config.sherpa_provider})...",
        file=sys.stderr,
    )
    transcriber = Transcriber(
        model_name=config.whisper_model,
        use_int8=config.sherpa_use_int8,
        provider=config.sherpa_provider,
        num_threads=config.sherpa_num_threads,
        models_dir=config.models_dir,
        ignore_words=config.ignore_words,
    )
    transcriber.load()

    embedder = SpeakerEmbedding(models_dir=config.models_dir)
    if embedder.is_available():
        print("Loading speaker embedding model...", file=sys.stderr)
    else:
        print("Speaker embedding model not found, speaker ID disabled", file=sys.stderr)

    registry = SpeakerRegistry(
        similarity_threshold=config.speaker_similarity_threshold,
        timeout_seconds=config.speaker_inactivity_timeout,
    )

    output = JsonlOutput(device_name=config.device_name)

    # Load VAD
    vad = VoiceActivityDetector(
        threshold=config.vad_threshold,
        sample_rate=config.sample_rate,
        models_dir=config.models_dir,
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
        vad=vad,
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
            speech_seconds = vad.get_speech_duration(audio)
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
            segments = transcriber.transcribe(audio, sample_rate=config.sample_rate)
            if not segments:
                queue.delete(filepath)
                continue

            # Identify speaker
            speaker_id = 0
            confidence = 1.0
            if embedder.is_available():
                embedding = embedder.extract(audio, sample_rate=config.sample_rate)
                if embedding is not None:
                    speaker_id = registry.identify(embedding, chunk_timestamp)

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
