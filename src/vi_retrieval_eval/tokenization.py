import re
from typing import List

HAVE_UTS = False
try:
    from underthesea import word_tokenize as uts_word_tokenize

    HAVE_UTS = True
except Exception:
    HAVE_UTS = False


def regex_tokens(s: str) -> List[str]:
    """Tokenize text using a Unicode-aware regex fallback.

    Args:
        s: Input text.

    Returns:
        List[str]: Extracted tokens.
    """
    # Find contiguous alphanumeric token candidates, including Vietnamese letters.
    return re.findall(r"[0-9A-Za-zÀ-ỹ_]+", s or "")


def vi_segment(s: str) -> List[str]:
    """Segment Vietnamese text.

    Uses `underthesea` when available, otherwise falls back to regex tokenization.

    Args:
        s: Input text.

    Returns:
        List[str]: Segmented tokens.
    """
    if HAVE_UTS:
        return uts_word_tokenize(s or "", format="text").split()
    return regex_tokens(s or "")
