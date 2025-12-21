"""Sherpa-ONNX backend for ASR and speaker identification."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

from .backend import Backend, TranscriptSegment, SpeakerResult


class SherpaBackend(Backend):
    """
    Backend using sherpa-onnx for ASR and speaker identification.

    Sherpa-ONNX provides a unified SDK that works across platforms
    (desktop, Android, iOS, Raspberry Pi) with ONNX Runtime.
    Good for mobile deployment with NPU acceleration.
    """

    # Model download paths (will be downloaded on first use)
    MODELS_DIR = Path("./models/sherpa")

    # Default model URLs from sherpa-onnx releases
    WHISPER_MODEL = "sherpa-onnx-whisper-small.en"
    SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0"
    EMBEDDING_MODEL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k"

    def __init__(
        self,
        whisper_model: str = "small.en",
        speaker_similarity_threshold: float = 0.35,
        sherpa_use_int8: bool = False,
        sherpa_provider: str = "cpu",
        sherpa_num_threads: int = 4,
    ):
        """
        Initialize the sherpa-onnx backend.

        Args:
            whisper_model: Whisper model size (tiny.en, small.en, etc.)
            speaker_similarity_threshold: Threshold for speaker matching
            sherpa_use_int8: Use INT8 quantized models for faster inference
            sherpa_provider: ONNX execution provider (cpu, cuda, coreml)
            sherpa_num_threads: Number of CPU threads for inference
        """
        self.whisper_model_name = whisper_model
        self.speaker_similarity_threshold = speaker_similarity_threshold
        self.sherpa_use_int8 = sherpa_use_int8
        self.sherpa_provider = sherpa_provider
        self.sherpa_num_threads = sherpa_num_threads

        self._recognizer = None
        self._speaker_embedding = None
        self._speaker_registry: dict[int, np.ndarray] = {}  # speaker_id -> centroid
        self._next_speaker_id = 0

    def _ensure_models(self) -> dict:
        """Ensure models are downloaded and return paths."""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx not installed. Install with: pip install sherpa-onnx"
            )

        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Map model name to sherpa model path
        model_map = {
            "tiny.en": "sherpa-onnx-whisper-tiny.en",
            "base.en": "sherpa-onnx-whisper-base.en",
            "small.en": "sherpa-onnx-whisper-small.en",
            "medium.en": "sherpa-onnx-whisper-medium.en",
        }

        whisper_name = model_map.get(self.whisper_model_name, "sherpa-onnx-whisper-small.en")
        whisper_dir = self.MODELS_DIR / whisper_name

        # Check if models exist, if not provide download instructions
        if not whisper_dir.exists():
            print(
                f"Whisper model not found at {whisper_dir}",
                file=sys.stderr,
            )
            print(
                f"Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models",
                file=sys.stderr,
            )
            print(
                f"Example: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{whisper_name}.tar.bz2",
                file=sys.stderr,
            )
            print(
                f"Then: tar xf {whisper_name}.tar.bz2 -C {self.MODELS_DIR}",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"Model not found: {whisper_dir}")

        return {
            "whisper_dir": whisper_dir,
        }

    def load(self) -> None:
        """Preload models."""
        import sherpa_onnx

        paths = self._ensure_models()
        whisper_dir = paths["whisper_dir"]

        int8_suffix = ".int8" if self.sherpa_use_int8 else ""
        print(
            f"Loading Sherpa Whisper model ({self.whisper_model_name}, "
            f"int8={self.sherpa_use_int8}, provider={self.sherpa_provider})...",
            file=sys.stderr,
        )

        # Determine model file prefix (e.g., "medium.en" from "sherpa-onnx-whisper-medium.en")
        model_prefix = whisper_dir.name.replace("sherpa-onnx-whisper-", "")

        # Create offline recognizer with Whisper
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(whisper_dir / f"{model_prefix}-encoder{int8_suffix}.onnx"),
            decoder=str(whisper_dir / f"{model_prefix}-decoder{int8_suffix}.onnx"),
            tokens=str(whisper_dir / f"{model_prefix}-tokens.txt"),
            language="en",
            num_threads=self.sherpa_num_threads,
            provider=self.sherpa_provider,
        )

        # Load speaker embedding model if available
        embedding_dir = self.MODELS_DIR / self.EMBEDDING_MODEL
        embedding_path = embedding_dir / "model.onnx"
        if embedding_path.exists():
            print("Loading speaker embedding model...", file=sys.stderr)
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding_path),
                num_threads=4,
            )
            self._speaker_embedding = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        else:
            print(
                f"Speaker embedding model not found at {embedding_path}, speaker ID disabled",
                file=sys.stderr,
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[TranscriptSegment]:
        """Transcribe audio using sherpa-onnx Whisper."""
        if self._recognizer is None:
            self.load()

        # Create stream and decode
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        if not text:
            return []

        # Sherpa doesn't provide word-level timestamps in offline mode,
        # so we return the full text as one segment
        duration = len(audio) / sample_rate
        return [
            TranscriptSegment(
                text=text,
                start=0.0,
                end=duration,
            )
        ]

    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> SpeakerResult:
        """Identify speaker using sherpa-onnx embeddings."""
        if self._speaker_embedding is None:
            # No speaker embedding model, assign all to speaker 0
            return SpeakerResult(speaker_id=0, confidence=1.0)

        # Extract embedding
        stream = self._speaker_embedding.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()

        if not self._speaker_embedding.is_ready(stream):
            return SpeakerResult(speaker_id=0, confidence=0.0)

        embedding = self._speaker_embedding.compute(stream)
        embedding = np.array(embedding)

        # Find best matching speaker
        best_speaker = None
        best_score = -1.0

        for speaker_id, centroid in self._speaker_registry.items():
            score = self._cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id

        # Check if match is good enough
        if best_score >= self.speaker_similarity_threshold and best_speaker is not None:
            # Update centroid with running average
            old_centroid = self._speaker_registry[best_speaker]
            self._speaker_registry[best_speaker] = 0.9 * old_centroid + 0.1 * embedding
            return SpeakerResult(speaker_id=best_speaker, confidence=best_score)
        else:
            # Create new speaker
            new_id = self._next_speaker_id
            self._next_speaker_id += 1
            self._speaker_registry[new_id] = embedding
            return SpeakerResult(speaker_id=new_id, confidence=1.0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def reset_speakers(self) -> None:
        """Reset speaker registry."""
        self._speaker_registry.clear()
        self._next_speaker_id = 0

    def get_speaker_count(self) -> int:
        """Return number of known speakers."""
        return len(self._speaker_registry)
