# -*- coding: utf-8 -*-
import re
import unicodedata

# Zero-width / invisible chars
_ZW_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\ufeff",  # BOM
]

# Collapse all whitespace into a single space
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# A broad emoji / pictograph range (covers most common emoji)
_EMOJI_RE = re.compile(
    "["  # start char class
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)

# Control characters (ASCII 0..31, 127) — removed for cleanliness
_CTRL_RE = re.compile(r"[\x00-\x1F\x7F]")


def normalize_for_dedup(
    s: str, *, do_lower: bool = False, remove_emoji: bool = False
) -> str:
    """
    Normalize a string for deduplication / matching:
      - Unicode normalize NFKC
      - remove control chars (ASCII control)
      - remove zero-width/invisible characters
      - optional: remove emoji/icons if remove_emoji=True
      - strip leading/trailing whitespace
      - collapse whitespace to a single space
      - optional: lowercase if do_lower=True
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    # Remove control chars
    s = _CTRL_RE.sub("", s)
    # Remove zero-width/invisible chars
    for ch in _ZW_CHARS:
        s = s.replace(ch, "")
    # Optional: remove emoji/icon
    if remove_emoji:
        s = _EMOJI_RE.sub("", s)
    # Trim + collapse spaces
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    # Optional lower
    if do_lower:
        s = s.lower()
    return s
