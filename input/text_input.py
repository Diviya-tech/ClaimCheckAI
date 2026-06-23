"""Text input handler.

The simplest input path: pasted text. We only normalize it — strip surrounding
whitespace, normalize unicode, and collapse excessive blank lines — so every
downstream stage receives clean, consistent text regardless of input source.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(raw: str) -> str:
    """Return a normalized, trimmed version of `raw` text.

    Steps:
      * Unicode NFKC normalization (folds look-alike / full-width chars,
        normalizes smart quotes' composition, etc.).
      * Normalize line endings to ``\\n``.
      * Strip trailing spaces on each line.
      * Collapse 3+ consecutive blank lines down to a single blank line.
      * Strip leading/trailing whitespace overall.

    Raises:
        ValueError: if `raw` is empty or whitespace-only after cleaning.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected str, got {type(raw).__name__}.")

    # Normalize unicode composition (e.g. full-width and compatibility chars).
    text = unicodedata.normalize("NFKC", raw)

    # Unify line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing whitespace per line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse runs of blank lines (3+ newlines) into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")
    return cleaned
