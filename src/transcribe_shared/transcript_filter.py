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


def _is_nonsense_word(word: str) -> bool:
    """Check if a single token looks like hallucinated noise rather than a real word.

    Catches things like 'ucherucherucherucher', 'tiktiktik', 'erererererer' —
    long strings with no spaces that have very low character diversity relative
    to their length. Real words use a wider variety of characters.
    """
    if len(word) < 10:
        return False
    # Strip punctuation for analysis
    alpha = re.sub(r'[^a-z]', '', word.lower())
    if len(alpha) < 10:
        return False
    # Character diversity ratio: unique chars / length
    # Real words: "internationalization" = 12 unique / 20 len = 0.60
    # Noise: "ucherucherucherucher" = 5 unique / 20 len = 0.25
    # Noise: "tiktiktiktiktik" = 3 unique / 15 len = 0.20
    # Real: "Mississippi" = 4 unique / 11 len = 0.36 (borderline but short)
    ratio = len(set(alpha)) / len(alpha)
    # Longer words need more diversity to be real
    if len(alpha) >= 15 and ratio < 0.3:
        return True
    if len(alpha) >= 10 and ratio < 0.2:
        return True
    return False


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

    words = text_lower.split()

    # Check for nonsense words (low character diversity, long tokens)
    for w in words:
        if _is_nonsense_word(w):
            return True

    # Check if all words are in the ignore list
    if ignore_words:
        ignore_set = {w.lower() for w in ignore_words}
        non_ignored = [w for w in words if w not in ignore_set]
        if not non_ignored:
            return True

    # Detect repetitive text (e.g., "ACL ACL ACL", "the the the")
    if len(words) >= 2:
        if len(set(words)) == 1:
            return True
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


def dedup_repetition(text: str) -> str:
    """Remove looping phrase repetitions from Whisper hallucinations.

    Detects patterns like "you know how you know how you know how..." and
    collapses them to a single occurrence, preserving any non-repeating
    prefix/suffix.
    """
    words = text.split()
    if len(words) < 6:
        return text

    # Try phrase lengths from 2 to 8 words
    for phrase_len in range(2, 9):
        i = 0
        while i <= len(words) - phrase_len:
            phrase = words[i:i + phrase_len]
            # Count consecutive repeats of this phrase
            repeats = 1
            j = i + phrase_len
            while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                repeats += 1
                j += phrase_len
            if repeats >= 3:
                # Collapse: keep prefix + one phrase + suffix
                result = words[:i] + phrase + words[j:]
                return dedup_repetition(" ".join(result))
            i += 1

    return text


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
