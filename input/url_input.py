"""URL input handler.

Fetches a web page and extracts just the article body using Trafilatura
(boilerplate, nav, ads, and comments are stripped). The extracted text is then
run through the same `clean_text` normalizer as pasted text so downstream stages
can't tell the difference.
"""

from __future__ import annotations

from urllib.parse import urlparse

from input.text_input import clean_text


class URLExtractionError(Exception):
    """Raised when a URL can't be fetched or yields no usable article text."""


def _looks_like_url(value: str) -> bool:
    """Cheap structural check that `value` is an http(s) URL."""
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def extract_from_url(url: str) -> str:
    """Download `url` and return the cleaned article body text.

    Args:
        url: An http(s) URL pointing at an article / blog post.

    Returns:
        Cleaned article body text.

    Raises:
        URLExtractionError: for invalid URLs, failed fetches, or empty
            extractions. The original cause is chained for debugging.
    """
    if not isinstance(url, str) or not _looks_like_url(url):
        raise URLExtractionError(f"Not a valid http(s) URL: {url!r}")

    # Imported lazily so the rest of the package imports without trafilatura
    # installed (e.g. when only running text-input tests).
    try:
        import trafilatura  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise URLExtractionError(
            "trafilatura is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:  # network/SSL/etc. — surface as a clean error
        raise URLExtractionError(f"Failed to fetch {url!r}: {exc}") from exc

    if not downloaded:
        raise URLExtractionError(f"Could not download content from {url!r}.")

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not extracted or not extracted.strip():
        raise URLExtractionError(f"No article text could be extracted from {url!r}.")

    try:
        return clean_text(extracted)
    except ValueError as exc:
        raise URLExtractionError(f"Extracted text from {url!r} was empty after cleaning.") from exc
