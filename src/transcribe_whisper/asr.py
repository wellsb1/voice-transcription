"""Speech transcription using faster-whisper."""

from typing import Optional
import numpy as np
from faster_whisper import WhisperModel

from transcribe_shared.transcript_filter import is_garbage_transcript


class Transcriber:
    """
    Speech-to-text transcription using faster-whisper.

    Uses CTranslate2 backend for efficient CPU inference.
    """

    def __init__(
        self,
        model_name: str = "medium.en",
        device: str = "cpu",
        compute_type: str = "int8",
        ignore_words: Optional[list[str]] = None,
    ):
        """
        Initialize transcriber.

        Args:
            model_name: Whisper model size (tiny.en, base.en, small.en, medium.en)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Compute type ("int8", "float16", "float32")
            ignore_words: Words to always filter from transcripts
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.ignore_words = ignore_words or []
        self._model: Optional[WhisperModel] = None

    def _load_model(self) -> None:
        """Lazy-load Whisper model."""
        if self._model is not None:
            return

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def load(self) -> None:
        """Preload the model (downloads if needed)."""
        self._load_model()

    def transcribe(self, audio: np.ndarray) -> list[dict]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)

        Returns:
            List of segments with "start", "end", "text" keys
        """
        self._load_model()

        segments, info = self._model.transcribe(
            audio,
            beam_size=1,  # Greedy decoding for speed
            language="en",
            vad_filter=False,  # We handle VAD separately
        )

        results = []
        for segment in segments:
            text = segment.text.strip()
            if text and not is_garbage_transcript(text, self.ignore_words):
                results.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": text,
                    }
                )

        return results

    def transcribe_with_words(self, audio: np.ndarray) -> list[dict]:
        """
        Transcribe audio with word-level timestamps.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz)

        Returns:
            List of segments with word-level timing
        """
        self._load_model()

        segments, info = self._model.transcribe(
            audio,
            beam_size=1,
            language="en",
            word_timestamps=True,
        )

        results = []
        for segment in segments:
            text = segment.text.strip()
            if text and not is_garbage_transcript(text, self.ignore_words):
                words = []
                if segment.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in segment.words
                    ]
                results.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": text,
                        "words": words,
                    }
                )

        return results
