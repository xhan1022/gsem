"""String normalization utilities for entity normalization."""
import re
import unicodedata


def normalize_string(text: str) -> str:
    """Normalize a string for entity matching and lexicon lookup.

    Performs minimal cleaning to create consistent entity representations:
    - Strips leading/trailing whitespace
    - Converts to lowercase (for English)
    - Normalizes full-width to half-width characters
    - Removes redundant spaces and punctuation
    - Preserves essential symbols (e.g., "CT", "HbA1c", "C-reactive protein")

    Args:
        text: Raw entity string

    Returns:
        Normalized string
    """
    if not text:
        return ""

    # 1. Strip leading/trailing whitespace
    text = text.strip()

    # 2. Normalize Unicode (NFKC form handles full-width to half-width)
    text = unicodedata.normalize('NFKC', text)

    # 3. Convert to lowercase (English terms)
    text = text.lower()

    # 4. Remove redundant spaces (multiple spaces → single space)
    text = re.sub(r'\s+', ' ', text)

    # 5. Remove leading/trailing punctuation (but preserve internal hyphens, slashes, etc.)
    # Only strip common punctuation from edges
    text = text.strip('.,;:!?()[]{}"\' ')

    # 6. Normalize common medical abbreviations spacing
    # e.g., "C - reactive protein" → "c-reactive protein"
    text = re.sub(r'\s*-\s*', '-', text)

    return text


def are_strings_equivalent(str1: str, str2: str) -> bool:
    """Check if two strings are equivalent after normalization.

    Args:
        str1: First string
        str2: Second string

    Returns:
        True if normalized forms are identical
    """
    return normalize_string(str1) == normalize_string(str2)
