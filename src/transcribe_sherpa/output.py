"""JSONL output formatting and stdout streaming."""

import json
import sys
from datetime import datetime
from typing import TextIO


class JsonlOutput:
    """
    JSONL output formatter for transcription results.

    Writes one JSON object per line to stdout (or specified stream).
    """

    def __init__(
        self,
        device_name: str,
        stream: TextIO = None,
    ):
        """
        Initialize JSONL output.

        Args:
            device_name: Device identifier to include in output
            stream: Output stream (default: stdout)
        """
        self.device_name = device_name
        self.stream = stream or sys.stdout

    def write(
        self,
        text: str,
        speaker_id: int,
        confidence: float,
        rms: float = 0.0,
        db: float = -60.0,
        timestamp: datetime = None,
    ) -> None:
        """
        Write a transcript line to output.

        Args:
            text: Transcribed text
            speaker_id: Speaker identifier
            confidence: Speaker match confidence (0.0-1.0)
            rms: RMS audio energy (0.0-1.0)
            db: Audio level in decibels (-60 to 0)
            timestamp: Timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        record = {
            "ts": timestamp.isoformat(timespec="milliseconds"),
            "device": self.device_name,
            "speaker": speaker_id,
            "confidence": round(confidence, 3),
            "rms": round(float(rms), 4),
            "db": round(float(db), 1),
            "text": text,
        }

        line = json.dumps(record, ensure_ascii=False)
        self.stream.write(line + "\n")
        self.stream.flush()

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
        Write multiple transcript segments.

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

        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            self.write(
                text=text,
                speaker_id=speaker_id,
                confidence=confidence,
                rms=rms,
                db=db,
                timestamp=base_timestamp,
            )
