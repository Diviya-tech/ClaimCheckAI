"""ClaimCheck AI — pipeline orchestrator.

The full pipeline runs end to end: input normalization -> claim extraction ->
evidence retrieval -> verdict evaluation -> dossier assembly. This module owns
the *sequence*; the stages live in `core/`. Both the CLI (below) and the HTTP
API (`api/server.py`) call the same functions here, so there is exactly one
place that decides how the stages chain, how long each took, and what happens
when one of them degrades.

Usage:
    python main.py --text "Green tea melts belly fat in two weeks."
    python main.py --url   "https://example.com/some-health-article"
    python main.py --image shot.png
    python main.py --video "https://tiktok.com/@x/video/123"
    python main.py --text "..." --no-evidence   # extract + decompose only
    python main.py --text "..." --json          # also emit the dossier as JSON
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from pydantic import ValidationError

from config import providers
from core.cache import dossier_cache
from core.claim_extractor import extract_claims
from core.dossier_builder import build_dossier, format_dossier
from core.evidence_retriever import collect_limitations, retrieve_evidence
from core.verdict_engine import ClaimEvaluation, evaluate, fallback_verdicts
from input.image_input import ImageExtractionError, extract_from_image
from input.text_input import clean_text
from input.url_input import URLExtractionError, extract_from_url
from input.video_input import VideoExtractionError, extract_from_video
from schemas.models import ClaimExtractionResult, Dossier, SourceFormat

logger = logging.getLogger("claimcheck.pipeline")

# User-readable note added to the dossier when the token budget runs out AFTER
# extraction but BEFORE verdicts. Retrieval is token-free, so the evidence is
# still gathered and shown — it just hasn't been weighed.
NOTE_VERDICT_BUDGET = (
    "The daily token budget ran out before the evidence could be evaluated. Each "
    "fact is marked 'Insufficient Evidence' and its retrieved sources are shown "
    "unassessed (as context) rather than as support or refutation."
)


# --------------------------------------------------------------------------- #
# Stage timing
# --------------------------------------------------------------------------- #
@contextmanager
def stage_timer(stage: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    """Log how long a pipeline stage took (and record it in `timings` if given)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[stage] = elapsed
        logger.info("[timing] %-10s %6.2fs", stage, elapsed)


# --------------------------------------------------------------------------- #
# Stages 1-2: normalize + extract
# --------------------------------------------------------------------------- #
def check_claim(
    text: str | None = None,
    url: str | None = None,
    image: str | None = None,
    video: str | None = None,
) -> ClaimExtractionResult:
    """Run the pipeline's first half: normalize input -> extract claims.

    Exactly one of `text`, `url`, `image`, or `video` must be provided.

    Returns:
        The ClaimExtractionResult (the primary claim decomposed into facts).

    Raises:
        ValueError: if not exactly one input is given, or the text is empty.
        URLExtractionError: if a URL can't be fetched/extracted.
        ImageExtractionError: if an image can't be read/extracted.
        VideoExtractionError: if a video can't be downloaded/processed.
        providers.BudgetWarning: if the daily token budget is exhausted.
        providers.MalformedOutputError: if the extractor's structured output was
            unusable twice in a row (already retried once).
    """
    provided = [v for v in (text, url, image, video) if v is not None]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of `text`, `url`, `image`, or `video`.")

    with stage_timer("normalize"):
        if url is not None:
            clean = extract_from_url(url)
            source_format = SourceFormat.URL
        elif image is not None:
            clean = extract_from_image(image)
            source_format = SourceFormat.SCREENSHOT
        elif video is not None:
            clean = extract_from_video(video).combined_text
            source_format = SourceFormat.VIDEO
        else:
            clean = clean_text(text or "")
            source_format = SourceFormat.TEXT

    with stage_timer("extract"):
        return extract_claims(clean, source_format=source_format)


# --------------------------------------------------------------------------- #
# Stages 3-5: retrieve -> evaluate -> assemble
# --------------------------------------------------------------------------- #
def complete_dossier(
    extraction: ClaimExtractionResult,
    allow_over_budget: bool = False,
    timings: dict[str, float] | None = None,
) -> Dossier:
    """Finish the pipeline from an extraction result: evidence -> verdicts -> dossier.

    Degrades rather than fails. Retrieval is already resilient (a dead source is
    a note, not an error). If the token budget runs out at the verdict stage the
    facts are marked Insufficient Evidence with their evidence attached unassessed,
    the dossier records why in `limitations`, and assembly continues. The only
    exception that escapes is a genuinely unexpected error.

    A no-claim extraction skips retrieval/evaluation entirely but still yields a
    (trivial) dossier so the output shape is always consistent.
    """
    limitations: list[str] = []
    verdicts = []
    flags = []

    if extraction.claim_found:
        with stage_timer("retrieve", timings):
            fact_evidence = retrieve_evidence(
                extraction.atomic_facts, allow_over_budget=allow_over_budget
            )
        limitations.extend(collect_limitations(fact_evidence))

        with stage_timer("evaluate", timings):
            try:
                evaluation = evaluate(
                    extraction, fact_evidence, allow_over_budget=allow_over_budget
                )
            except providers.BudgetWarning as exc:
                logger.warning("Budget exhausted at the verdict stage: %s", exc)
                evaluation = ClaimEvaluation(
                    verdicts=fallback_verdicts(fact_evidence), rhetorical_flags=[]
                )
                limitations.append(NOTE_VERDICT_BUDGET)
        verdicts = evaluation.verdicts
        flags = evaluation.rhetorical_flags

    with stage_timer("assemble", timings):
        return build_dossier(
            original_input=extraction.original_text,
            claim_extraction=extraction,
            verdicts=verdicts,
            source_format=extraction.source_format,
            rhetorical_flags=flags,
            allow_over_budget=allow_over_budget,
            limitations=limitations,
        )


def run_pipeline(
    text: str | None = None,
    url: str | None = None,
    image: str | None = None,
    video: str | None = None,
    use_cache: bool = True,
) -> Dossier:
    """The whole pipeline, input to dossier, with the text cache in front.

    Exactly one input must be given (see `check_claim`). Text input is looked up
    in the in-memory cache first; a hit returns the previously built dossier
    without spending any tokens. Extraction-stage errors propagate (see
    `check_claim`); everything after extraction degrades into `limitations`.
    """
    cacheable = text is not None and use_cache
    if cacheable:
        cached = dossier_cache.get(text)
        if cached is not None:
            return cached

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    extraction = check_claim(text=text, url=url, image=image, video=video)
    dossier = complete_dossier(extraction, timings=timings)
    logger.info("[timing] %-10s %6.2fs", "total", time.perf_counter() - total_start)

    if cacheable and extraction.claim_found:
        dossier_cache.put(text, dossier)
    return dossier


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_result(result: ClaimExtractionResult) -> None:
    """Pretty-print a ClaimExtractionResult to stdout."""
    print("=" * 70)
    print(f"Source format: {result.source_format.value}")
    print(f"Claim found:   {result.claim_found}")
    print(f"Primary claim: {result.primary_claim or '(none)'}")
    print("-" * 70)
    if result.atomic_facts:
        print(f"Atomic facts ({len(result.atomic_facts)}):")
        for i, fact in enumerate(result.atomic_facts, start=1):
            print(f"  [{i}] ({fact.fact_type.value}) {fact.text}")
            print(f"       context: {fact.original_context!r}")
    else:
        print("Atomic facts: (none)")
    print("=" * 70)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ClaimCheck AI — evaluate health claims from text, a URL, a screenshot, or a video."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Raw text to analyze.")
    group.add_argument("--url", help="URL of an article to analyze.")
    group.add_argument("--image", help="Path to a screenshot (.png/.jpg/.jpeg/.webp).")
    group.add_argument("--video", help="Video URL (TikTok / Instagram / YouTube / X).")
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip evidence retrieval and evaluation; only extract and decompose the claim.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print the full dossier as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-stage timing and pipeline logs on stderr.",
    )
    return parser


def _force_utf8_stdout() -> None:
    """Make stdout/stderr emit UTF-8 so non-ASCII evidence text never crashes.

    PubMed abstracts and web snippets routinely contain Unicode (hyphens like
    \\u2010, en/em dashes, Greek letters). On Windows the console defaults to a
    legacy code page (cp1252) that can't encode those, so printing raises
    UnicodeEncodeError. Reconfiguring to UTF-8 with `errors="replace"` keeps the
    run going regardless of the active code page.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # TextIOWrapper on 3.7+; absent if redirected oddly
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = _build_parser().parse_args(argv)
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
        )

    # --- Stages 1-2: normalize input + extract/decompose the claim. ---
    try:
        if args.video:
            # Video has two channels (spoken + on-screen); surface both, then
            # run claim extraction on the merged text.
            video = extract_from_video(args.video)
            print("=" * 70)
            print("TRANSCRIPTION (spoken):")
            print(video.transcription or "(none)")
            print("-" * 70)
            print("ON-SCREEN TEXT (visual):")
            print(video.visual_text or "(none)")
            result = extract_claims(video.combined_text, source_format=SourceFormat.VIDEO)
        else:
            result = check_claim(text=args.text, url=args.url, image=args.image)
    except ValidationError as exc:
        print(f"Invalid extraction result (invariant violated): {exc}", file=sys.stderr)
        return 1
    except (ValueError, URLExtractionError, ImageExtractionError, VideoExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except providers.MalformedOutputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except providers.BudgetWarning as exc:
        print(f"Budget guardrail: {exc}", file=sys.stderr)
        return 2

    # `--no-evidence` is the debug path: show the raw extraction and stop before
    # spending any retrieval/evaluation effort.
    if args.no_evidence:
        _print_result(result)
        return 0

    # --- Stages 3-5: retrieve evidence, evaluate verdicts, build the dossier. ---
    # Post-extraction failures degrade into the dossier's `limitations` rather
    # than aborting, so a run that got this far always prints a dossier.
    dossier = complete_dossier(result)

    print("\n" + format_dossier(dossier))
    if args.json:
        print("\n" + dossier.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
