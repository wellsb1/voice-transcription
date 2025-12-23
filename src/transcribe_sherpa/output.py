"""JSON array output formatting and stdout streaming."""

import json
import sys
from datetime import datetime
from typing import TextIO


class JsonlOutput:
    """
    JSON array output formatter for transcription results.

    Writes one JSON array per chunk to stdout (or specified stream).
    """

    def __init__(
        self,
        device_name: str,
        stream: TextIO = None,
    ):
        """
        Initialize output.

        Args:
            device_name: Device identifier to include in output
            stream: Output stream (default: stdout)
        """
        self.device_name = device_name
        self.stream = stream or sys.stdout

    def write_segments(
        self,
        segments: list[dict],
        speaker_id: int,
        confidence: float,
        rms: float = 0.0,
        db: float = -60.0,
        base_timestamp: datetime = None,
    ) -> None:
        """
        Write transcript segments as a JSON array to stdout.

        Combines all segments into a single record within an array.

        Args:
            segments: List of {"text": str, "start": float, "end": float}
            speaker_id: Speaker identifier
            confidence: Speaker match confidence
            rms: RMS audio energy (0.0-1.0)
            db: Audio level in decibels (-60 to 0)
            base_timestamp: Base timestamp for the chunk
        """
        if base_timestamp is None:
            base_timestamp = datetime.now()

        # Aggregate all segment texts into one utterance
        texts = []
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                texts.append(text)

        if not texts:
            return

        combined_text = " ".join(texts)
        record = {
            "ts": base_timestamp.isoformat(timespec="milliseconds"),
            "device": self.device_name,
            "speaker": speaker_id,
            "confidence": round(confidence, 3),
            "rms": round(float(rms), 4),
            "db": round(float(db), 1),
            "text": combined_text,
        }

        # Output as single-element array for consistency with M4 backend
        line = json.dumps([record], ensure_ascii=False)
        self.stream.write(line + "\n")
        self.stream.flush()
