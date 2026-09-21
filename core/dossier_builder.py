"""Dossier builder — stage 5 of the pipeline (weeks 7-8).

Assembles the final evidence dossier: the extracted claim, the per-atom verdict
table, the evidence cited for each verdict (with source-tier badges and stance
markers), the rhetorical flags detected in the claim, and a plain-language
narrative summary a non-expert can follow. Every dossier is UUID-stamped and
timestamped (the model layer does that automatically).

The only reasoning step here is the narrative summary, which is written by the
PREMIUM model from the already-decided verdicts. Everything else is deterministic
assembly and formatting. If the narrative call is unavailable (budget exhausted or
API error) the builder degrades to a deterministic summary so a dossier is always
produced — the verdicts, which cost the most to compute, are never thrown away.
"""

from __future__ import annotations

import logging
import textwrap
from collections import Counter

from config import providers
from config.settings import SOURCE_TIER_LABELS, SourceTier
from schemas.models import (
    AtomicVerdict,
    ClaimExtractionResult,
    Dossier,
    Evidence,
    EvidenceStance,
    RhetoricalFlag,
    SourceFormat,
    Verdict,
)

logger = logging.getLogger("claimcheck.dossier_builder")

_WRAP_WIDTH = 78
_INPUT_PREVIEW_CHARS = 600


# --------------------------------------------------------------------------- #
# Narrative summary (the only LLM use in this stage)
# --------------------------------------------------------------------------- #
_NARRATIVE_SYSTEM_PROMPT = """\
You write the closing summary of a health-claim evidence dossier for a general
reader with no medical background. You are given the original claim, the verdict
reached for each part of it, and the reasoning behind each verdict.

Write 3-6 sentences in plain, calm language that:
  - state what the claim asserts and what the evidence overall shows,
  - are faithful to the individual verdicts — do not upgrade "Insufficient
    Evidence" into a clean yes or no, and do not overstate certainty,
  - explain briefly WHY (e.g. weak evidence, higher-quality studies disagree),
  - if rhetorical red flags were found, note that the claim is framed in a
    manipulative or exaggerated way, separately from what the evidence says.

Do not invent facts, studies, or numbers beyond what you are given. Do not give
medical advice. Return only the summary text matching the schema.
"""

_NARRATIVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "narrative_summary": {
            "type": "string",
            "description": "3-6 plain-language sentences summarizing the dossier.",
        }
    },
    "required": ["narrative_summary"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
# User-readable limitation notes the builder itself can add.
NOTE_NARRATIVE_BUDGET = (
    "The daily token budget ran out before the plain-language summary could be "
    "written; the summary below is an automatic tally of the verdicts."
)
NOTE_NARRATIVE_FAILED = (
    "The plain-language summary could not be generated (the reasoning model "
    "returned an unusable response); the summary below is an automatic tally of "
    "the verdicts."
)


def build_dossier(
    original_input: str,
    claim_extraction: ClaimExtractionResult,
    verdicts: list[AtomicVerdict],
    source_format: SourceFormat = SourceFormat.TEXT,
    rhetorical_flags: list[RhetoricalFlag] | None = None,
    allow_over_budget: bool = False,
    limitations: list[str] | None = None,
) -> Dossier:
    """Assemble the complete, UUID-stamped evidence dossier.

    Args:
        original_input: The normalized text the analysis was run on.
        claim_extraction: The extractor result (primary claim + atomic facts).
        verdicts: Per-atom verdicts from the verdict engine.
        source_format: Where the input came from (kept for provenance; the same
            value already lives on `claim_extraction`).
        rhetorical_flags: Flags from the verdict engine, if any.
        allow_over_budget: Forwarded to the narrative LLM call.
        limitations: User-readable notes about what degraded THIS run upstream
            (a source that was down, a budget cut at the verdict stage). The
            builder appends its own note if the narrative call itself degrades.

    Returns:
        A fully populated `Dossier`. Always — a failed narrative call never
        discards the verdicts; it degrades to a deterministic tally and says so.
    """
    flags = rhetorical_flags or []
    notes = list(limitations or [])
    narrative, note = _generate_narrative(
        claim_extraction, verdicts, flags, allow_over_budget=allow_over_budget
    )
    if note and note not in notes:
        notes.append(note)
    return Dossier(
        original_input=original_input,
        claim_extraction=claim_extraction,
        verdicts=verdicts,
        rhetorical_flags=flags,
        narrative_summary=narrative,
        limitations=notes,
    )


def _generate_narrative(
    claim_extraction: ClaimExtractionResult,
    verdicts: list[AtomicVerdict],
    flags: list[RhetoricalFlag],
    allow_over_budget: bool = False,
) -> tuple[str, str]:
    """Write the plain-language summary (premium LLM), with a deterministic fallback.

    Returns (summary, limitation_note); the note is empty when the LLM summary
    was produced normally.
    """
    if not claim_extraction.claim_found or not verdicts:
        return (
            "No evaluable health claim was found in the provided input, so no "
            "evidence assessment was performed.",
            "",
        )

    prompt = _build_narrative_prompt(claim_extraction, verdicts, flags)
    note = NOTE_NARRATIVE_FAILED
    try:
        result = providers.llm_call(
            prompt=prompt,
            system_prompt=_NARRATIVE_SYSTEM_PROMPT,
            model_tier="premium",
            response_schema=_NARRATIVE_SCHEMA,
            allow_over_budget=allow_over_budget,
        )
        assert isinstance(result, dict)
        summary = (result.get("narrative_summary") or "").strip()
        if summary:
            return summary, ""
        logger.warning("Narrative call returned empty text; using deterministic summary.")
    except providers.BudgetWarning:
        logger.warning("Budget exhausted before narrative; using deterministic summary.")
        note = NOTE_NARRATIVE_BUDGET
    except Exception as exc:  # noqa: BLE001 — the dossier must still assemble
        logger.warning("Narrative generation failed (%s); using deterministic summary.", exc)

    return _fallback_narrative(claim_extraction, verdicts), note


def _build_narrative_prompt(
    claim_extraction: ClaimExtractionResult,
    verdicts: list[AtomicVerdict],
    flags: list[RhetoricalFlag],
) -> str:
    lines = [f"CLAIM: {claim_extraction.primary_claim}", "", "VERDICTS:"]
    for i, v in enumerate(verdicts, start=1):
        lines.append(f"  {i}. [{v.verdict.value}] {v.atomic_fact.text}")
        if v.reasoning:
            lines.append(f"     reasoning: {v.reasoning}")
    if flags:
        lines.append("")
        lines.append("RHETORICAL RED FLAGS DETECTED:")
        for f in flags:
            lines.append(f"  - {f.pattern}: {f.explanation}")
    return "\n".join(lines)


def _fallback_narrative(
    claim_extraction: ClaimExtractionResult, verdicts: list[AtomicVerdict]
) -> str:
    """Deterministic summary assembled from verdict counts — no LLM."""
    counts = Counter(v.verdict for v in verdicts)
    tally = "; ".join(f"{n} {verdict.value}" for verdict, n in counts.items())
    claim = claim_extraction.primary_claim or "the claim"
    return (
        f"Across {len(verdicts)} atomic fact(s) drawn from \"{claim}\", the "
        f"evidence assessment found: {tally}. See the per-claim verdicts above "
        "for the specific reasoning and cited sources."
    )


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #
_STANCE_MARKER = {
    EvidenceStance.SUPPORTING: "(+)",
    EvidenceStance.OPPOSING: "(-)",
    EvidenceStance.NEUTRAL: "(.)",
}

# A one-glyph cue per verdict so the table scans quickly.
_VERDICT_MARKER = {
    Verdict.STRONGLY_SUPPORTED: "+++",
    Verdict.PARTIALLY_SUPPORTED: "+",
    Verdict.INSUFFICIENT_EVIDENCE: "?",
    Verdict.CONFLICTING_EVIDENCE: "><",
    Verdict.PARTIALLY_REFUTED: "-",
    Verdict.STRONGLY_REFUTED: "---",
    Verdict.TOO_VAGUE: "~",
}


def format_dossier(dossier: Dossier) -> str:
    """Render a complete dossier as clean, readable terminal text."""
    out: list[str] = []
    bar = "=" * _WRAP_WIDTH
    thin = "-" * _WRAP_WIDTH

    out.append(bar)
    out.append("CLAIMCHECK AI  —  EVIDENCE DOSSIER")
    out.append(bar)
    out.append(f"Dossier ID : {dossier.id}")
    out.append(f"Generated  : {dossier.timestamp.isoformat()}")
    out.append(f"Source     : {dossier.claim_extraction.source_format.value}")
    out.append(thin)

    out.append("INPUT")
    out.append(_indent_wrap(_preview(dossier.original_input), 2))
    out.append("")

    extraction = dossier.claim_extraction
    if not extraction.claim_found:
        out.append("PRIMARY CLAIM")
        out.append("  (no evaluable health claim found)")
        out.append("")
        out.append(thin)
        out.append("SUMMARY")
        out.append(_indent_wrap(dossier.narrative_summary, 2))
        out.append(bar)
        return "\n".join(out)

    out.append("PRIMARY CLAIM")
    out.append(_indent_wrap(extraction.primary_claim, 2))
    out.append("")

    out.extend(_format_flags(dossier.rhetorical_flags))
    out.append(thin)
    out.append(f"PER-CLAIM VERDICTS ({len(dossier.verdicts)})")
    out.append(thin)
    for i, verdict in enumerate(dossier.verdicts, start=1):
        out.extend(_format_verdict(i, verdict))
        out.append("")

    out.append(thin)
    out.append("NARRATIVE SUMMARY")
    out.append(_indent_wrap(dossier.narrative_summary, 2))
    out.extend(_format_limitations(dossier.limitations))
    out.append(bar)
    return "\n".join(out)


def _format_limitations(limitations: list[str]) -> list[str]:
    if not limitations:
        return []
    lines = ["", "LIMITATIONS OF THIS RUN"]
    for note in limitations:
        lines.append(_indent_wrap(f"- {note}", 2))
    return lines


def _format_flags(flags: list[RhetoricalFlag]) -> list[str]:
    lines = [f"RHETORICAL RED FLAGS ({len(flags)})"]
    if not flags:
        lines.append("  none detected")
        lines.append("")
        return lines
    for f in flags:
        lines.append(f"  [!] {f.pattern}")
        if f.excerpt:
            lines.append(_indent_wrap(f'"{f.excerpt}"', 6))
        if f.explanation:
            lines.append(_indent_wrap(f.explanation, 6))
    lines.append("")
    return lines


def _format_verdict(index: int, verdict: AtomicVerdict) -> list[str]:
    fact = verdict.atomic_fact
    marker = _VERDICT_MARKER.get(verdict.verdict, "")
    lines = [
        f"[{index}] ({fact.fact_type.value}) {fact.text}",
        f"    VERDICT: {verdict.verdict.value}  {marker}",
    ]
    if verdict.reasoning:
        lines.append(_indent_wrap(verdict.reasoning, 4))

    # Cite evidence in weight order: supporting, opposing, then context.
    cited = (
        verdict.supporting_evidence
        + verdict.opposing_evidence
        + verdict.neutral_evidence
    )
    if cited:
        lines.append("    Evidence:")
        for ev in sorted(cited, key=lambda e: e.source_tier):
            lines.extend(_format_evidence(ev))
    else:
        lines.append("    Evidence: none retrieved")
    return lines


def _format_evidence(ev: Evidence) -> list[str]:
    marker = _STANCE_MARKER.get(ev.evidence_stance, "(.)")
    pub = ev.publication_date.date().isoformat() if ev.publication_date else "n.d."
    header = f"      {marker} [T{ev.source_tier}] {ev.source_name} ({pub})  rel={ev.relevance_score:.2f}"
    snippet = (ev.summary or ev.content or "").strip()
    lines = [header]
    if snippet:
        lines.append(_indent_wrap(snippet, 10))
    if ev.source_url:
        lines.append(f"          {ev.source_url}")
    return lines


def tier_label(tier: int) -> str:
    """Human-readable label for a source tier (safe for out-of-range ints)."""
    try:
        return SOURCE_TIER_LABELS.get(SourceTier(tier), "")
    except ValueError:
        return ""


def _preview(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _INPUT_PREVIEW_CHARS:
        return text[:_INPUT_PREVIEW_CHARS].rstrip() + " […]"
    return text or "(empty)"


def _indent_wrap(text: str, indent: int) -> str:
    """Wrap `text` to the dossier width with a hanging indent of `indent` spaces."""
    pad = " " * indent
    wrapped = textwrap.fill(
        (text or "").strip(),
        width=_WRAP_WIDTH,
        initial_indent=pad,
        subsequent_indent=pad,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped or pad
