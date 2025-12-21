"""Speaker embedding extraction using SpeechBrain ECAPA-TDNN."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio

# Monkey-patch for torchaudio 2.9+ compatibility with speechbrain
# The list_audio_backends() function was removed in newer versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']


class SpeakerEmbedding:
    """
    Extract speaker embeddings using SpeechBrain's ECAPA-TDNN model.

    Produces 192-dimensional vectors representing speaker voice characteristics.
    """

    def __init__(self, model_dir: Optional[str | Path] = None):
        """
        Initialize speaker embedding extractor.

        Args:
            model_dir: Directory to save/load model (default: ./models/spkrec-ecapa-voxceleb)
        """
        if model_dir is None:
            model_dir = Path("./models/spkrec-ecapa-voxceleb")
        self.model_dir = Path(model_dir)
        self._classifier = None

    def _load_model(self) -> None:
        """Lazy-load SpeechBrain classifier."""
        if self._classifier is not None:
            return

        from speechbrain.inference.speaker import EncoderClassifier

        self._classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.model_dir),
        )

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio (default 16kHz)

        Returns:
            192-dimensional numpy array representing speaker
        """
        self._load_model()

        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        # Extract embedding
        embedding = self._classifier.encode_batch(audio_tensor)

        # Return as numpy array, removing batch dimension
        return embedding.squeeze().numpy()

    def extract_from_segments(
        self,
        audio: np.ndarray,
        segments: list[dict],
        sample_rate: int = 16000,
    ) -> list[np.ndarray]:
        """
        Extract embeddings for specific segments of audio.

        Args:
            audio: Full audio data as numpy array
            segments: List of {"start": seconds, "end": seconds} dicts
            sample_rate: Sample rate of audio

        Returns:
            List of embeddings, one per segment
        """
        embeddings = []
        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Skip very short segments
            if len(segment_audio) < sample_rate * 0.5:  # Less than 0.5 seconds
                continue

            embedding = self.extract(segment_audio, sample_rate)
            embeddings.append(embedding)

        return embeddings
