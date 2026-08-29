"""Pre-tokenization: splitting raw text into chunks before BPE sees it.

Two jobs:

  1. Split on a regex so that merges can never cross a word boundary. Without
     this, BPE learns cross-word merges like "the cat" as a single symbol, which
     wastes vocabulary and generalises badly.

  2. Make whitespace visible. Space becomes "Ġ", newline "Ċ", tab "ĉ", so
     whitespace can be merged like any other symbol AND recovered exactly on
     decode -- a tokenizer that eats its own spaces cannot round-trip.

The regex alternation is ordered so the matches concatenate back to the input
exactly. tests/test_tokenizer.py asserts that property; if it ever fails,
characters are silently vanishing before the model ever sees them.
"""

from __future__ import annotations

import re

PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"      # common English contractions
    r"| ?[^\W\d_]+"              # optional leading space + letters
    r"| ?\d+"                    # optional leading space + digits
    r"| ?[^\s\w]+"               # optional leading space + punctuation
    r"|\s+(?!\S)"                # trailing whitespace run
    r"|\s+"                      # any other whitespace run
)

WHITESPACE_MARKERS: dict[str, str] = {" ": "Ġ", "\n": "Ċ", "\t": "ĉ", "\r": "Ř"}
MARKER_TO_WHITESPACE: dict[str, str] = {v: k for k, v in WHITESPACE_MARKERS.items()}
MARKER_CHARS: set[str] = set(MARKER_TO_WHITESPACE)


def mark_whitespace(text: str) -> str:
    for whitespace, marker in WHITESPACE_MARKERS.items():
        text = text.replace(whitespace, marker)
    return text


def unmark_whitespace(text: str) -> str:
    for marker, whitespace in MARKER_TO_WHITESPACE.items():
        text = text.replace(marker, whitespace)
    return text


def pretokenize(text: str) -> list[str]:
    """Split text into whitespace-marked chunks."""
    return [mark_whitespace(chunk) for chunk in PRETOKEN_PATTERN.findall(text)]
