"""Wake word detection using OpenWakeWord."""

import sys
import threading
from typing import Callable, Optional

import numpy as np


class WakeWordDetector:
    """
    Runs OpenWakeWord on audio chunks to detect wake words.

    Thread-safe: can be called from the audio callback thread.
    Detection callback fires on the audio thread — keep it fast.
    """

    def __init__(
        self,
        model_names: list[str],
        threshold: float = 0.5,
        on_detect: Optional[Callable[[str, float], None]] = None,
    ):
        self.model_names = model_names
        self.threshold = threshold
        self.on_detect = on_detect
        self._model = None
        self._enabled = True
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load OpenWakeWord models."""
        from openwakeword.model import Model

        print(f"Loading wake word models: {', '.join(self.model_names)}", file=sys.stderr)
        self._model = Model(
            wakeword_models=self.model_names,
            inference_framework="onnx",
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable detection (e.g., during capture mode)."""
        with self._lock:
            self._enabled = enabled

    def process(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Feed an audio chunk and check for wake word detections.

        Args:
            audio_chunk: float32 audio from sounddevice

        Returns:
            Detected model name, or None
        """
        if self._model is None:
            return None

        with self._lock:
            if not self._enabled:
                return None

        # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        predictions = self._model.predict(audio_int16)

        for model_name, score in predictions.items():
            if score > self.threshold:
                print(f"  Wake word detected: {model_name} (score={score:.3f})", file=sys.stderr)
                # Reset model state to avoid repeated detections
                self._model.reset()
                return model_name

        return None
