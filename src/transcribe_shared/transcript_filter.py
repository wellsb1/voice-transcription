"""Shared transcript filtering for garbage/hallucination detection."""

import re
from typing import Optional


# Known Whisper hallucination patterns (regex)
HALLUCINATION_PATTERNS = [
    r"^\[.*\]$",  # [inaudible], [music], etc.
    r"^♪",  # Music notes
    r"^\.+$",  # Just periods
    r"^-+$",  # Just dashes
    r"^\*+$",  # Just asterisks
    r"^thank(s| you)( for watching| for listening)?\.?$",  # Common hallucination
    r"^please subscribe",  # Common hallucination
    r"^(uh|um|ah|oh|eh)[\s,\.]*$",  # Just filler words
]


def is_garbage_transcript(
    text: str,
    ignore_words: Optional[list[str]] = None,
) -> bool:
    """
    Detect garbage/hallucinated transcripts from non-speech audio.

    Args:
        text: Transcript text to check
        ignore_words: Optional list of words to always filter out.
            If the transcript consists entirely of these words, it's garbage.

    Returns:
        True if text appears to be noise, keyboard clicks, or hallucination.
    """
    if not text or len(text.strip()) < 2:
        return True

    text_lower = text.strip().lower()

    # Check hallucination patterns
    for pattern in HALLUCINATION_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return True

    # Parse words for further checks
    words = text_lower.split()

    # Check if all words are in the ignore list
    if ignore_words:
        ignore_set = {w.lower() for w in ignore_words}
        non_ignored = [w for w in words if w not in ignore_set]
        if not non_ignored:
            return True

    # Detect repetitive text (e.g., "ACL ACL ACL", "the the the")
    if len(words) >= 2:
        # Check if all words are the same
        if len(set(words)) == 1:
            return True
        # Check if it's highly repetitive (>70% same word)
        word_counts: dict[str, int] = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_count = max(word_counts.values())
        if max_count / len(words) > 0.7 and len(words) > 2:
            return True

    # Single nonsense word (less than 3 chars)
    if len(words) == 1 and len(words[0]) <= 2:
        return True

    return False


def filter_transcript(
    text: str,
    ignore_words: Optional[list[str]] = None,
) -> str:
    """
    Filter a transcript, returning empty string if it's garbage.

    Args:
        text: Transcript text to check
        ignore_words: Optional list of words to always filter out

    Returns:
        Original text if valid, empty string if garbage
    """
    if is_garbage_transcript(text, ignore_words):
        return ""
    return text.strip()
