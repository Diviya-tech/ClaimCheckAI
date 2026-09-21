"""In-memory dossier cache (weeks 11-12).

The cheapest pipeline run is the one that never happens. Checking the same
claim twice costs two premium LLM calls and a round of PubMed/Tavily traffic to
reproduce a dossier that already exists — so the exact same claim text returns
the cached dossier instead (same UUID, same verdicts, same evidence).

Deliberately simple: a bounded, thread-safe LRU keyed on the *normalized* claim
text. Normalization (`input.text_input.clean_text`) means trailing whitespace,
smart quotes, zero-width characters, and line-ending differences all hit the
same entry. It is per-process and not persistent; semantic (embedding-based)
caching of paraphrased claims is a later iteration.

Only text input is cached. A URL can change content between visits and an
image/video is a different upload every time, so those always run fresh.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from input.text_input import clean_text
from schemas.models import Dossier

logger = logging.getLogger("claimcheck.cache")

DEFAULT_MAX_ENTRIES = 256


class DossierCache:
    """Bounded LRU of `Dossier`s keyed by normalized claim text."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")
        self._max = max_entries
        self._entries: OrderedDict[str, Dossier] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(text: str) -> str | None:
        """Normalize `text` into a cache key; None if it normalizes to nothing."""
        try:
            return clean_text(text)
        except (TypeError, ValueError):
            return None

    def get(self, text: str) -> Dossier | None:
        key = self.key_for(text)
        if key is None:
            return None
        with self._lock:
            dossier = self._entries.get(key)
            if dossier is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)  # most recently used
            self.hits += 1
        logger.info("Cache hit for claim %r (dossier %s).", _preview(key), dossier.id)
        return dossier

    def put(self, text: str, dossier: Dossier) -> None:
        key = self.key_for(text)
        if key is None:
            return
        with self._lock:
            self._entries[key] = dossier
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                evicted, _ = self._entries.popitem(last=False)
                logger.debug("Evicted cached dossier for %r.", _preview(evicted))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, text: object) -> bool:
        if not isinstance(text, str):
            return False
        key = self.key_for(text)
        with self._lock:
            return key is not None and key in self._entries


def _preview(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# One shared cache per process — the CLI creates a fresh one per run (so it
# never hits), the API server keeps this one alive across requests.
dossier_cache = DossierCache()
