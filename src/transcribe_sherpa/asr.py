"""Speech transcription using sherpa-onnx Whisper."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np


class Transcriber:
    """
    Speech-to-text transcription using sherpa-onnx.

    Uses ONNX Runtime for efficient cross-platform inference.
    """

    # Model name to sherpa directory mapping
    MODEL_MAP = {
        "tiny.en": "sherpa-onnx-whisper-tiny.en",
        "base.en": "sherpa-onnx-whisper-base.en",
        "small.en": "sherpa-onnx-whisper-small.en",
        "medium.en": "sherpa-onnx-whisper-medium.en",
    }

    def __init__(
        self,
        model_name: str = "small.en",
        use_int8: bool = False,
        provider: str = "cpu",
        num_threads: int = 4,
        models_dir: str = "./models/sherpa",
    ):
        """
        Initialize transcriber.

        Args:
            model_name: Whisper model size (tiny.en, base.en, small.en, medium.en)
            use_int8: Use INT8 quantized models for faster inference
            provider: ONNX execution provider (cpu, cuda, coreml)
            num_threads: Number of CPU threads for inference
            models_dir: Directory containing sherpa models
        """
        self.model_name = model_name
        self.use_int8 = use_int8
        self.provider = provider
        self.num_threads = num_threads
        self.models_dir = Path(models_dir)
        self._recognizer = None

    def _get_model_dir(self) -> Path:
        """Get the model directory path."""
        sherpa_name = self.MODEL_MAP.get(self.model_name, "sherpa-onnx-whisper-small.en")
        return self.models_dir / sherpa_name

    def load(self) -> None:
        """Load the model."""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError(
                "sherpa-onnx not installed. Install with: pip install sherpa-onnx"
            )

        model_dir = self._get_model_dir()

        if not model_dir.exists():
            print(
                f"Whisper model not found at {model_dir}",
                file=sys.stderr,
            )
            print(
                f"Download from: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models",
                file=sys.stderr,
            )
            sherpa_name = model_dir.name
            print(
                f"Example: wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{sherpa_name}.tar.bz2",
                file=sys.stderr,
            )
            print(
                f"Then: tar xf {sherpa_name}.tar.bz2 -C {self.models_dir}",
                file=sys.stderr,
            )
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Determine model file prefix
        model_prefix = model_dir.name.replace("sherpa-onnx-whisper-", "")
        int8_suffix = ".int8" if self.use_int8 else ""

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(model_dir / f"{model_prefix}-encoder{int8_suffix}.onnx"),
            decoder=str(model_dir / f"{model_prefix}-decoder{int8_suffix}.onnx"),
            tokens=str(model_dir / f"{model_prefix}-tokens.txt"),
            language="en",
            num_threads=self.num_threads,
            provider=self.provider,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of audio

        Returns:
            List of segments with "start", "end", "text" keys
        """
        if self._recognizer is None:
            self.load()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        if not text:
            return []

        # Sherpa doesn't provide word-level timestamps in offline mode
        duration = len(audio) / sample_rate
        return [
            {
                "text": text,
                "start": 0.0,
                "end": duration,
            }
        ]
