"""Screenshot input handler (Week 3).

Misinformation overwhelmingly lives in images — Instagram carousels, TikTok
frames, tweet screenshots, YouTube thumbnails. This handler reads such a
screenshot with Claude's vision capability (via the provider abstraction, premium
tier) and extracts just the central health claim as clean text, discarding the
surrounding social-media noise (usernames, hashtags, UI chrome, comments).

The output is the same clean text every other input handler produces, so it flows
straight into the claim extractor.
"""

from __future__ import annotations

import base64
from pathlib import Path

from config import providers
from input.text_input import clean_text


class ImageExtractionError(Exception):
    """Raised when an image can't be read or yields no usable claim text."""


# Supported formats mapped to their MIME type (what the vision API expects).
_SUPPORTED_FORMATS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# Sentinel the model returns when the image contains no health claim at all.
_NO_CLAIM_SENTINEL = "NO_CLAIM_FOUND"

_SYSTEM_PROMPT = f"""\
You are the screenshot-reading stage of a health-claim evaluation engine.

You are given a screenshot of social-media content (an Instagram post, a TikTok
frame, a tweet, a YouTube thumbnail, a Facebook post, etc.). Your ONLY job is to
read the image and extract the central HEALTH claim being made, as clean plain text.

Extract ONLY the substantive claim. Specifically:
- IGNORE usernames, handles, display names, and profile pictures.
- IGNORE hashtags, @-mentions, and emoji.
- IGNORE UI chrome: like / comment / share / view counts, buttons, timestamps,
  "Follow", navigation bars, "link in bio", and similar.
- IGNORE comments and replies from other users.
- IGNORE watermarks, app logos, and stickers.

Capture the main assertion exactly as stated, preserving any numbers, doses, or
time frames that are part of the claim (e.g. "12%", "5000 IU", "in two weeks").
Do not summarize away specifics, do not add commentary, and do not fact-check or
judge the claim — just transcribe the claim itself.

Return the claim text and nothing else. If the image contains NO health-related
claim at all, respond with exactly: {_NO_CLAIM_SENTINEL}
"""

_USER_PROMPT = "Extract the central health claim from this screenshot."


def extract_from_image(image_path: str, allow_over_budget: bool = False) -> str:
    """Read a screenshot and return the extracted health-claim text.

    Args:
        image_path: Path to a .png, .jpg, .jpeg, or .webp image file.
        allow_over_budget: Forwarded to the LLM layer's soft budget guardrail.

    Returns:
        Cleaned claim text, ready for the claim extractor.

    Raises:
        ImageExtractionError: if the path is missing/not a file, the format is
            unsupported, or no health claim could be read from the image.
        providers.BudgetWarning: if the daily token budget is exhausted and
            `allow_over_budget` is False.
    """
    path = Path(image_path)
    if not path.is_file():
        raise ImageExtractionError(f"Image file not found: {image_path!r}")

    media_type = _SUPPORTED_FORMATS.get(path.suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise ImageExtractionError(
            f"Unsupported image format {path.suffix!r}. Supported: {supported}."
        )

    try:
        encoded = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    except OSError as exc:
        raise ImageExtractionError(f"Could not read image {image_path!r}: {exc}") from exc

    # Vision needs reasoning over a noisy image, so this uses the premium tier.
    result = providers.llm_call(
        prompt=_USER_PROMPT,
        system_prompt=_SYSTEM_PROMPT,
        model_tier="premium",
        images=[{"media_type": media_type, "data": encoded}],
        allow_over_budget=allow_over_budget,
    )

    # A vision call without a response_schema always returns text.
    text = result.strip() if isinstance(result, str) else str(result).strip()

    if not text or text == _NO_CLAIM_SENTINEL:
        raise ImageExtractionError(
            f"No health claim could be read from the image {image_path!r}."
        )

    try:
        return clean_text(text)
    except ValueError as exc:
        raise ImageExtractionError(
            f"Extracted claim from {image_path!r} was empty after cleaning."
        ) from exc
