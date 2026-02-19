"""JSONL output formatting with batch envelope."""

import json
import uuid
from datetime import timedelta

from .pipeline import DiarizedTranscript


class JsonlOutput:
    """
    Outputs diarized transcripts as JSONL batch envelopes to stdout.

    Each batch is a single JSON object per line:
    {"id": "...", "timestamp": "...", "device": "...", "utterances": [...]}
    """

    def __init__(self, device_name: str = "m4-mini"):
        self.device_name = device_name

    def write_batch(self, transcripts: list[DiarizedTranscript]) -> None:
        """Write batch of transcripts as a JSONL envelope to stdout."""
        if not transcripts:
            return

        batch_timestamp = transcripts[0].batch_timestamp

        utterances = []
        for t in transcripts:
            utterances.append({
                "speaker": t.speaker,
                "confidence": round(t.confidence, 3),
                "start": round(t.start, 3),
                "end": round(t.end, 3),
                "text": t.text,
            })

        envelope = {
            "id": str(uuid.uuid4()),
            "timestamp": batch_timestamp.isoformat(),
            "device": self.device_name,
            "utterances": utterances,
        }

        print(json.dumps(envelope), flush=True)
