"""Text input handler.

The simplest input path: pasted text. We only normalize it — strip surrounding
whitespace, normalize unicode, and collapse excessive blank lines — so every
downstream stage receives clean, consistent text regardless of input source.
"""

from __future__ import annotations

import re
import unicodedata

# Control characters and invisible formatting marks that survive NFKC but break
# things downstream: they corrupt terminal output, get counted as claim text by
# the extractor, and are a classic way to smuggle content past a text filter.
# Newline and tab are deliberately preserved — they carry real structure.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Zero-width and bidi-override marks. Scraped pages and social copy-paste are
# full of them, and they make otherwise-identical claims hash differently
# (which would silently defeat the dossier cache).
_INVISIBLES = re.compile("[​-‏  ‪-‮⁠﻿]")


def clean_text(raw: str) -> str:
    """Return a normalized, trimmed version of `raw` text.

    Steps:
      * Unicode NFKC normalization (folds look-alike / full-width chars,
        normalizes smart quotes' composition, etc.).
      * Normalize line endings to ``\\n``.
      * Drop control characters and zero-width / bidi marks (keeping ``\\n``
        and ``\\t``).
      * Strip trailing spaces on each line.
      * Collapse 3+ consecutive blank lines down to a single blank line.
      * Strip leading/trailing whitespace overall.

    Ordinary non-ASCII text — emoji, accents, Greek letters, non-Latin scripts —
    is preserved untouched; only characters with no visible glyph are removed.

    Raises:
        TypeError: if `raw` is not a string.
        ValueError: if `raw` is empty or whitespace-only after cleaning.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected str, got {type(raw).__name__}.")

    # Normalize unicode composition (e.g. full-width and compatibility chars).
    text = unicodedata.normalize("NFKC", raw)

    # Unify line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove invisible junk *after* line endings are unified, so \r has already
    # become a real newline rather than being deleted as a control character.
    text = _CONTROL_CHARS.sub("", text)
    text = _INVISIBLES.sub("", text)

    # Trim trailing whitespace per line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse runs of blank lines (3+ newlines) into a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")
    return cleaned
