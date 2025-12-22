"""Speaker embedding extraction using sherpa-onnx 3D-Speaker."""

from pathlib import Path
from typing import Optional
import numpy as np


class SpeakerEmbedding:
    """
    Extract speaker embeddings using sherpa-onnx's speaker embedding model.

    Uses 3D-Speaker ERes2Net model for cross-platform inference.
    """

    EMBEDDING_MODEL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k"

    def __init__(self, models_dir: str = "./models/sherpa"):
        """
        Initialize speaker embedding extractor.

        Args:
            models_dir: Directory containing sherpa models
        """
        self.models_dir = Path(models_dir)
        self._extractor = None
        self._available = None

    def _get_model_path(self) -> Path:
        """Get the model file path."""
        return self.models_dir / self.EMBEDDING_MODEL / "model.onnx"

    def is_available(self) -> bool:
        """Check if the embedding model is available."""
        if self._available is None:
            self._available = self._get_model_path().exists()
        return self._available

    def _load_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._extractor is not None:
            return

        if not self.is_available():
            return

        try:
            import sherpa_onnx

            model_path = self._get_model_path()
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path),
                num_threads=4,
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except ImportError:
            self._available = False

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio

        Returns:
            Embedding numpy array or None if not available
        """
        if not self.is_available():
            return None

        self._load_model()

        if self._extractor is None:
            return None

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate, audio)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            return None

        embedding = self._extractor.compute(stream)
        return np.array(embedding)
