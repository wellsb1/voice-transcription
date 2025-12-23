"""JSON array output formatting."""

import json
from datetime import timedelta

from .pipeline import DiarizedTranscript


class JsonlOutput:
    """
    Outputs diarized transcripts as JSON arrays to stdout.

    Each batch is output as a single JSON array on one line:
    [{"timestamp": "...", "speaker": "SPEAKER_00", "text": "Hello"}, ...]
    """

    def __init__(self, device_name: str = "m4-mini"):
        """
        Initialize output.

        Args:
            device_name: Device identifier for output
        """
        self.device_name = device_name

    def _to_record(self, transcript: DiarizedTranscript) -> dict:
        """Convert a transcript to a record dict."""
        segment_offset = timedelta(seconds=transcript.start)
        segment_timestamp = transcript.batch_timestamp + segment_offset

        return {
            "timestamp": segment_timestamp.isoformat(),
            "device": self.device_name,
            "speaker": transcript.speaker,
            "confidence": round(transcript.confidence, 3),
            "start": round(transcript.start, 3),
            "end": round(transcript.end, 3),
            "text": transcript.text,
        }

    def write_batch(self, transcripts: list[DiarizedTranscript]) -> None:
        """
        Write batch of transcripts as a JSON array to stdout.

        Args:
            transcripts: List of DiarizedTranscript to output
        """
        if not transcripts:
            return

        records = [self._to_record(t) for t in transcripts]
        print(json.dumps(records), flush=True)
