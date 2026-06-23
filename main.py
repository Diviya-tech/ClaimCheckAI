"""ClaimCheck AI — pipeline orchestrator (Week 1 scope).

Right now the pipeline does input normalization + claim extraction only.
Evidence retrieval, verdicts, and dossier assembly arrive in later weeks; this
script will be wrapped in FastAPI in weeks 9-10.

Usage:
    python main.py --text "Green tea melts belly fat in two weeks."
    python main.py --url  "https://example.com/some-health-article"
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from config import providers
from core.claim_extractor import extract_claims
from input.text_input import clean_text
from input.url_input import URLExtractionError, extract_from_url
from schemas.models import ClaimExtractionResult, SourceFormat


def check_claim(
    text: str | None = None,
    url: str | None = None,
) -> ClaimExtractionResult:
    """Run the (Week 1) pipeline: normalize input -> extract claims.

    Exactly one of `text` or `url` must be provided.

    Returns:
        The ClaimExtractionResult. (Evidence + verdicts + dossier come later.)

    Raises:
        ValueError: if neither or both inputs are given.
        URLExtractionError: if a URL can't be fetched/extracted.
    """
    if (text is None) == (url is None):
        raise ValueError("Provide exactly one of `text` or `url`.")

    if url is not None:
        clean = extract_from_url(url)
        source_format = SourceFormat.URL
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
        description="ClaimCheck AI — extract and decompose health claims (Week 1)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Raw text to analyze.")
    group.add_argument("--url", help="URL of an article to analyze.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = check_claim(text=args.text, url=args.url)
    except ValidationError as exc:
        print(f"Invalid extraction result (invariant violated): {exc}", file=sys.stderr)
        return 1
    except (ValueError, URLExtractionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except providers.BudgetWarning as exc:
        print(f"Budget guardrail: {exc}", file=sys.stderr)
        return 2

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
