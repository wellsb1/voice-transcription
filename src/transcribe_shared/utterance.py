"""Shared utterance dataclass and aggregation utilities."""

from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar, Callable


@dataclass
class Utterance:
    """A transcribed utterance with speaker attribution."""

    speaker: str  # Speaker label (e.g., "SPEAKER_00" or int)
    text: str  # Transcribed text
    start: float  # Start time (seconds)
    end: float  # End time (seconds)
    timestamp: datetime  # Wall clock time
    confidence: float = 1.0  # Speaker match confidence (0.0-1.0)


T = TypeVar("T")


def aggregate_consecutive(
    items: list[T],
    get_speaker: Callable[[T], str],
    get_text: Callable[[T], str],
    get_start: Callable[[T], float],
    get_end: Callable[[T], float],
    merge: Callable[[T, T], T],
) -> list[T]:
    """
    Aggregate consecutive items with the same speaker.

    Generic function that works with any data type. Provide accessor
    functions for speaker, text, start, end, and a merge function.

    Args:
        items: List of items to aggregate
        get_speaker: Function to get speaker from item
        get_text: Function to get text from item
        get_start: Function to get start time from item
        get_end: Function to get end time from item
        merge: Function to merge two consecutive same-speaker items

    Returns:
        Aggregated list with consecutive same-speaker items merged
    """
    if not items:
        return []

    aggregated = []
    current = items[0]

    for item in items[1:]:
        if get_speaker(item) == get_speaker(current):
            current = merge(current, item)
        else:
            aggregated.append(current)
            current = item

    aggregated.append(current)
    return aggregated


def aggregate_utterances(utterances: list[Utterance]) -> list[Utterance]:
    """
    Aggregate consecutive utterances from the same speaker.

    Example: 5 utterances from SPEAKER_00 followed by 3 from SPEAKER_01
    becomes 2 aggregated utterances.

    Args:
        utterances: List of Utterance to aggregate

    Returns:
        Aggregated list with consecutive same-speaker utterances merged
    """
    if not utterances:
        return []

    def merge(a: Utterance, b: Utterance) -> Utterance:
        return Utterance(
            speaker=a.speaker,
            text=a.text + " " + b.text,
            start=a.start,
            end=b.end,
            timestamp=a.timestamp,
            confidence=min(a.confidence, b.confidence),
        )

    return aggregate_consecutive(
        utterances,
        get_speaker=lambda u: u.speaker,
        get_text=lambda u: u.text,
        get_start=lambda u: u.start,
        get_end=lambda u: u.end,
        merge=merge,
    )
