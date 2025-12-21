"""JSONL output formatting."""

import json
import sys
from datetime import datetime, timedelta
from typing import Optional

from .pipeline import DiarizedTranscript


class JsonlOutput:
    """
    Outputs diarized transcripts as JSONL to stdout.

    Format:
    {
        "timestamp": "2024-01-01T12:00:00.000000",
        "device": "m4-mini",
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 2.5,
        "text": "Hello world"
    }
    """

    def __init__(self, device_name: str = "m4-mini"):
        """
        Initialize output.

        Args:
            device_name: Device identifier for JSONL output
        """
        self.device_name = device_name

    def write(self, transcript: DiarizedTranscript) -> None:
        """
        Write a single transcript to stdout as JSONL.

        Args:
            transcript: DiarizedTranscript to output
        """
        # Calculate absolute timestamp for this segment
        segment_offset = timedelta(seconds=transcript.start)
        segment_timestamp = transcript.batch_timestamp + segment_offset

        record = {
            "timestamp": segment_timestamp.isoformat(),
            "device": self.device_name,
            "speaker": transcript.speaker,
            "start": round(transcript.start, 3),
            "end": round(transcript.end, 3),
            "text": transcript.text,
        }

        print(json.dumps(record), flush=True)

    def write_batch(self, transcripts: list[DiarizedTranscript]) -> None:
        """
        Write multiple transcripts to stdout.

        Args:
            transcripts: List of DiarizedTranscript to output
        """
        for transcript in transcripts:
            self.write(transcript)
