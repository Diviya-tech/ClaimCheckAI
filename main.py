"""ClaimCheck AI — pipeline orchestrator (Weeks 1-8 scope).

The full pipeline now runs end to end: input normalization -> claim extraction
-> evidence retrieval -> verdict evaluation -> dossier assembly, and prints the
completed evidence dossier. This script gets wrapped in FastAPI in weeks 9-10.

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
import sys

from pydantic import ValidationError

from config import providers
from core.claim_extractor import extract_claims
from core.dossier_builder import build_dossier, format_dossier
from core.evidence_retriever import retrieve_evidence
from core.verdict_engine import evaluate
from input.image_input import ImageExtractionError, extract_from_image
from input.text_input import clean_text
from input.url_input import URLExtractionError, extract_from_url
from input.video_input import VideoExtractionError, extract_from_video
from schemas.models import ClaimExtractionResult, SourceFormat


def check_claim(
    text: str | None = None,
    url: str | None = None,
    image: str | None = None,
    video: str | None = None,
) -> ClaimExtractionResult:
    """Run the pipeline: normalize input -> extract claims.

    Exactly one of `text`, `url`, `image`, or `video` must be provided.

    Returns:
        The ClaimExtractionResult. (Evidence + verdicts + dossier come later.)

    Raises:
        ValueError: if not exactly one input is given.
        URLExtractionError: if a URL can't be fetched/extracted.
        ImageExtractionError: if an image can't be read/extracted.
        VideoExtractionError: if a video can't be downloaded/processed.
    """
    provided = [v for v in (text, url, image, video) if v is not None]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of `text`, `url`, `image`, or `video`.")

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

    return extract_claims(clean, source_format=source_format)


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
        description="ClaimCheck AI — extract and decompose health claims from text, a URL, a screenshot, or a video."
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
        help="Also print the full dossier as JSON (handy for the future API).",
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
    except providers.BudgetWarning as exc:
        print(f"Budget guardrail: {exc}", file=sys.stderr)
        return 2

    # `--no-evidence` is the debug path: show the raw extraction and stop before
    # spending any retrieval/evaluation effort.
    if args.no_evidence:
        _print_result(result)
        return 0

    # --- Stages 3-5: retrieve evidence, evaluate verdicts, build the dossier. ---
    # A no-claim input skips retrieval/evaluation entirely but still yields a
    # (trivial) dossier so the output shape is always consistent.
    try:
        if result.claim_found:
            fact_evidence = retrieve_evidence(result.atomic_facts)
            evaluation = evaluate(result, fact_evidence)
            verdicts = evaluation.verdicts
            flags = evaluation.rhetorical_flags
        else:
            verdicts = []
            flags = []

        dossier = build_dossier(
            original_input=result.original_text,
            claim_extraction=result,
            verdicts=verdicts,
            source_format=result.source_format,
            rhetorical_flags=flags,
        )
    except providers.BudgetWarning as exc:
        print(f"Budget guardrail: {exc}", file=sys.stderr)
        return 2

    print("\n" + format_dossier(dossier))
    if args.json:
        print("\n" + dossier.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
