"""Speaker diarization using pyannote."""

import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a speaker."""

    speaker: str  # Speaker label (e.g., "SPEAKER_00")
    start: float  # Start time in seconds
    end: float  # End time in seconds


class Diarizer:
    """
    Speaker diarization using pyannote.

    Identifies who spoke when in an audio segment.
    """

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-community-1",
        huggingface_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize diarizer.

        Args:
            model: Pyannote model name on HuggingFace
            huggingface_token: HuggingFace token for model access
            device: Device to run on (None for auto-detect, "mps" for Apple Silicon)
        """
        self.model_name = model
        self.huggingface_token = huggingface_token

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._pipeline = None
        self._embedding_inference = None

    def load(self) -> None:
        """Load the diarization pipeline."""
        if self._pipeline is not None:
            return

        print(f"Loading diarization model ({self.model_name})...", file=sys.stderr)
        print(f"Using device: {self.device}", file=sys.stderr)

        # Suppress torchcodec warning - we don't decode files, we capture audio directly
        # This must be set before any pyannote imports since it triggers on module load
        warnings.filterwarnings("ignore", message=".*torchcodec.*")
        warnings.filterwarnings("ignore", module="pyannote.audio.core.io")

        # PyTorch 2.6+ requires allowlisting pyannote classes for safe loading
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])

        from pyannote.audio import Pipeline

        self._pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=self.huggingface_token,
        )
        self._pipeline.to(self.device)

        # Load embedding model separately — pyannote's pipeline often returns
        # empty speaker_embeddings (shape (0, N)), so we compute them ourselves
        from pyannote.audio import Model, Inference

        embedding_model = Model.from_pretrained(
            self.model_name,
            subfolder="embedding",
            token=self.huggingface_token,
        )
        embedding_model.to(self.device)
        self._embedding_inference = Inference(embedding_model, window="whole")

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[list[SpeakerSegment], dict[str, np.ndarray]]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Audio sample rate

        Returns:
            Tuple of:
            - List of SpeakerSegment with speaker labels and timestamps
            - Dict mapping speaker labels to their embeddings
        """
        if self._pipeline is None:
            self.load()

        # Pyannote expects {"waveform": tensor, "sample_rate": int}
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        # Run diarization (suppress numpy warnings from empty slices in silent audio)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            diarization_output = self._pipeline(audio_input)

        # DiarizeOutput is a dataclass with speaker_diarization (Annotation) field
        diarization = diarization_output.speaker_diarization

        # Convert to segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        # Build speaker → embedding map
        embeddings_by_speaker = self._extract_embeddings(
            diarization_output, diarization, audio, segments, sample_rate,
        )

        return segments, embeddings_by_speaker

    def _extract_embeddings(
        self,
        diarization_output,
        diarization,
        audio: np.ndarray,
        segments: list[SpeakerSegment],
        sample_rate: int,
    ) -> dict[str, np.ndarray]:
        """Extract speaker embeddings, computing them ourselves if pyannote didn't."""
        embeddings_by_speaker: dict[str, np.ndarray] = {}

        # Check if pyannote returned usable embeddings
        emb_tensor = diarization_output.speaker_embeddings
        if emb_tensor is not None and emb_tensor.shape[0] > 0:
            labels = list(diarization.labels())
            print(f"  Embeddings from pipeline: {len(labels)} speakers", file=sys.stderr)
            for i, label in enumerate(labels):
                embeddings_by_speaker[label] = emb_tensor[i]
            return embeddings_by_speaker

        # Pipeline returned empty embeddings — compute them from speaker audio
        # Collect all audio for each speaker label
        speaker_audio: dict[str, list[np.ndarray]] = {}
        for seg in segments:
            chunk = audio[int(seg.start * sample_rate):int(seg.end * sample_rate)]
            if len(chunk) > 0:
                speaker_audio.setdefault(seg.speaker, []).append(chunk)

        for label, chunks in speaker_audio.items():
            combined = np.concatenate(chunks)
            # Need at least 0.5s of audio for a meaningful embedding
            if len(combined) < sample_rate // 2:
                continue
            waveform = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
            audio_in = {"waveform": waveform, "sample_rate": sample_rate}
            try:
                emb = self._embedding_inference(audio_in)
                embeddings_by_speaker[label] = emb
            except Exception as e:
                print(f"  WARNING: Failed to compute embedding for {label}: {e}", file=sys.stderr)

        print(f"  Embeddings computed manually: {len(embeddings_by_speaker)} speakers", file=sys.stderr)
        return embeddings_by_speaker

    def get_speaker_audio(
        self,
        audio: np.ndarray,
        segment: SpeakerSegment,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract audio for a specific speaker segment.

        Args:
            audio: Full audio array
            segment: Speaker segment to extract
            sample_rate: Audio sample rate

        Returns:
            Audio slice for the segment
        """
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        return audio[start_sample:end_sample]
