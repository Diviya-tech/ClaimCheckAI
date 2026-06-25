"""ClaimCheck AI — pipeline orchestrator (Weeks 1-6 scope).

The pipeline now does input normalization -> claim extraction -> evidence
retrieval. Verdicts and dossier assembly arrive in later weeks; this script will
be wrapped in FastAPI in weeks 9-10.

Usage:
    python main.py --text "Green tea melts belly fat in two weeks."
    python main.py --url  "https://example.com/some-health-article"
    python main.py --text "..." --no-evidence    # skip evidence retrieval
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from config import providers
from config.settings import SOURCE_TIER_LABELS, SourceTier
from core.claim_extractor import extract_claims
from core.evidence_retriever import retrieve_evidence
from input.image_input import ImageExtractionError, extract_from_image
from input.text_input import clean_text
from input.url_input import URLExtractionError, extract_from_url
from input.video_input import VideoExtractionError, extract_from_video
from schemas.models import ClaimExtractionResult, FactEvidence, SourceFormat


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


def _tier_badge(tier: int) -> str:
    """Render a source tier as a compact badge, e.g. ``[T1]``."""
    return f"[T{tier}]"


def _print_evidence(fact_evidence: list[FactEvidence]) -> None:
    """Print retrieved evidence per atomic fact, grouped by source tier."""
    print("\n" + "#" * 70)
    print("EVIDENCE DOSSIER (retrieved — not yet evaluated)")
    print("#" * 70)
    for i, fe in enumerate(fact_evidence, start=1):
        fact = fe.atomic_fact
        print(f"\n[{i}] ({fact.fact_type.value}) {fact.text}")
        if fe.retrieval_note:
            print(f"    ! {fe.retrieval_note}")
        if not fe.evidence:
            print("    (no relevant evidence found — try a different phrasing or check API keys)")
            continue
        # Group by tier (1 -> 4) so the most authoritative sources come first.
        by_tier: dict[int, list] = {}
        for ev in fe.evidence:
            by_tier.setdefault(ev.source_tier, []).append(ev)
        for tier in sorted(by_tier):
            label = SOURCE_TIER_LABELS.get(SourceTier(tier), "")
            print(f"    {_tier_badge(tier)} {label}")
            for ev in by_tier[tier]:
                pub = ev.publication_date.date().isoformat() if ev.publication_date else "n.d."
                print(f"        • {ev.source_name} ({pub})  rel={ev.relevance_score:.2f}")
                print(f"          {ev.summary or ev.content[:200]}")
                print(f"          {ev.source_url}")
    print("#" * 70)


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
        help="Skip evidence retrieval; only extract and decompose the claim.",
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
    try:
        if args.video:
            # Video has two channels (spoken + on-screen); surface both, then
            # run claim extraction on the merged text.
            extraction = extract_from_video(args.video)
            print("=" * 70)
            print("TRANSCRIPTION (spoken):")
            print(extraction.transcription or "(none)")
            print("-" * 70)
            print("ON-SCREEN TEXT (visual):")
            print(extraction.visual_text or "(none)")
            result = extract_claims(
                extraction.combined_text, source_format=SourceFormat.VIDEO
            )
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

    _print_result(result)

    # Stage 3: evidence retrieval. Only runs when there's a claim to investigate
    # and the user didn't opt out. Retrieval is resilient — individual source
    # failures degrade to empty evidence rather than aborting the run.
    if not args.no_evidence and result.claim_found:
        try:
            fact_evidence = retrieve_evidence(result.atomic_facts)
        except providers.BudgetWarning as exc:
            print(f"Budget guardrail (evidence): {exc}", file=sys.stderr)
            return 2
        _print_evidence(fact_evidence)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
