"""Pydantic data models for the ClaimCheck AI pipeline.

These are the typed objects that flow between pipeline stages:

    raw input
        -> ClaimExtractionResult (primary claim + atomic facts)
        -> Evidence (retrieved per atomic fact)
        -> AtomicVerdict (one categorical verdict per atomic fact)
        -> Dossier (assembled output, UUID-stamped)

Verdicts are categorical, never numeric trust scores (see CLAUDE.md).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class FactType(str, Enum):
    """The kind of testable claim an atomic fact makes."""

    CAUSAL = "causal"              # "X causes / prevents Y"
    QUANTITATIVE = "quantitative"  # "X reduces risk by 40%"
    PRESCRIPTIVE = "prescriptive"  # "you should take X daily"
    TEMPORAL = "temporal"          # "results appear within 2 weeks"
    EXISTENTIAL = "existential"    # "compound X exists / is present in Y"


class SourceFormat(str, Enum):
    """Where the original input came from (set by the input normalizer)."""

    TEXT = "text"
    SCREENSHOT = "screenshot"
    URL = "url"
    VIDEO = "video"


class EvidenceStance(str, Enum):
    """How a single piece of evidence relates to the atomic fact it was retrieved for.

    Retrieval itself leaves every piece NEUTRAL — determining whether evidence
    actually supports or refutes a fact requires reasoning against the fact, which
    is the verdict engine's job (weeks 7-8). The field exists here so retrieval can
    carry the placeholder and the verdict engine can fill it in.
    """

    SUPPORTING = "supporting"
    OPPOSING = "opposing"
    NEUTRAL = "neutral"


class Verdict(str, Enum):
    """The seven categorical verdicts. No numeric scores."""

    STRONGLY_SUPPORTED = "Strongly Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"
    CONFLICTING_EVIDENCE = "Conflicting Evidence"
    PARTIALLY_REFUTED = "Partially Refuted"
    STRONGLY_REFUTED = "Strongly Refuted"
    TOO_VAGUE = "Too Vague to Evaluate"


# --------------------------------------------------------------------------- #
# Core models
# --------------------------------------------------------------------------- #
class AtomicFact(BaseModel):
    """A single, independently testable factual assertion drawn from a claim."""

    text: str = Field(..., description="The atomic, testable fact in plain language.")
    fact_type: FactType = Field(..., description="Classification of what the fact asserts.")
    original_context: str = Field(
        ...,
        description="The surrounding phrasing from the source that this fact was distilled from.",
    )


class ClaimExtractionResult(BaseModel):
    """Output of the claim extractor stage."""

    original_text: str = Field(..., description="The cleaned input text the claim was extracted from.")
    primary_claim: str = Field(
        default="",
        description="The single main health claim, or empty if none was found.",
    )
    atomic_facts: list[AtomicFact] = Field(
        default_factory=list,
        description="The primary claim decomposed into atomic testable facts.",
    )
    claim_found: bool = Field(
        ...,
        description="False when the input contains no evaluable health claim.",
    )
    source_format: SourceFormat = Field(
        default=SourceFormat.TEXT,
        description="Which input handler produced the text.",
    )

    @model_validator(mode="after")
    def _check_claim_atom_invariant(self) -> "ClaimExtractionResult":
        """Enforce consistency between `claim_found` and `atomic_facts`.

        A found claim must decompose into at least one atomic fact, and a
        not-found claim must carry none. This catches extractor regressions
        (e.g. a prompt change that summarizes instead of decomposing) at the
        point the result is constructed, rather than letting an inconsistent
        object flow downstream.
        """
        if self.claim_found and not self.atomic_facts:
            raise ValueError(
                "claim_found is True but atomic_facts is empty: a found claim "
                "must decompose into at least one atomic fact."
            )
        if not self.claim_found and self.atomic_facts:
            raise ValueError(
                "claim_found is False but atomic_facts is non-empty: no atomic "
                "facts may be present when no claim was found."
            )
        return self


class Evidence(BaseModel):
    """A single piece of retrieved evidence relevant to an atomic fact."""

    content: str = Field(..., description="The full evidence text (e.g. a PubMed abstract).")
    summary: str = Field(
        default="",
        description="Compressed 2-3 sentence summary of `content`, for token efficiency.",
    )
    source_url: str = Field(..., description="Canonical URL of the source.")
    source_name: str = Field(..., description="Human-readable source name (e.g. 'Cochrane').")
    source_tier: int = Field(..., ge=1, le=4, description="Source quality tier, 1 (best) to 4.")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance (vector/lexical similarity or search score), 0-1.",
    )
    evidence_stance: EvidenceStance = Field(
        default=EvidenceStance.NEUTRAL,
        description="Supporting / opposing / neutral relative to the fact (set by the verdict engine).",
    )
    publication_date: datetime | None = Field(
        default=None,
        description="When the source was published, if known.",
    )


class FactEvidence(BaseModel):
    """An atomic fact paired with the evidence retrieved for it.

    The evidence retriever returns one of these per atomic fact (rather than a
    flat list of Evidence) so the verdict engine can evaluate each fact against
    exactly the evidence gathered for it. `evidence` is already ranked and capped
    to the top few results.
    """

    atomic_fact: AtomicFact = Field(..., description="The fact this evidence was gathered for.")
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Top-ranked evidence for the fact (best first).",
    )
    retrieval_note: str = Field(
        default="",
        description=(
            "Set to 'insufficient evidence found' when fewer than 2 sufficiently "
            "relevant items survived the relevance floor — so the verdict engine "
            "doesn't mistake a thin, noisy result set for real evidence."
        ),
    )


class AtomicVerdict(BaseModel):
    """A categorical verdict for one atomic fact, with the evidence behind it.

    Every piece of evidence gathered for the fact lands in exactly one of the
    three buckets below, its `evidence_stance` set by the verdict engine, so the
    dossier can cite each source alongside how it bore on the verdict.
    """

    atomic_fact: AtomicFact = Field(..., description="The fact this verdict evaluates.")
    verdict: Verdict = Field(..., description="One of the seven categorical verdicts.")
    supporting_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence that supports the fact.",
    )
    opposing_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence that refutes or contradicts the fact.",
    )
    neutral_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence retrieved for the fact that neither supports nor "
        "refutes it (background / context only).",
    )
    reasoning: str = Field(
        ...,
        description="Transparent explanation of how the verdict follows from the evidence.",
    )


class RhetoricalFlag(BaseModel):
    """A rhetorical / manipulation pattern detected in the original claim text.

    Flags describe HOW a claim is argued, not whether it is true — a claim can be
    both accurate and rhetorically manipulative, or false and soberly worded. They
    are surfaced in the dossier as context, never folded into the evidence verdict.
    """

    pattern: str = Field(
        ...,
        description="Name of the pattern, e.g. 'Conspiracy framing' or 'Appeal to nature'.",
    )
    explanation: str = Field(
        ...,
        description="Why this pattern is a red flag in a health claim.",
    )
    excerpt: str = Field(
        default="",
        description="The phrase from the original claim that triggered the flag.",
    )


class Dossier(BaseModel):
    """The final evidence dossier — the pipeline's user-facing output."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Stable ID for sharing.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the dossier was generated (UTC).",
    )
    original_input: str = Field(..., description="The raw input as received.")
    claim_extraction: ClaimExtractionResult = Field(..., description="Extracted claim + atomic facts.")
    verdicts: list[AtomicVerdict] = Field(
        default_factory=list,
        description="Per-atom categorical verdicts.",
    )
    rhetorical_flags: list[RhetoricalFlag] = Field(
        default_factory=list,
        description="Rhetorical / manipulation patterns detected in the original claim text.",
    )
    narrative_summary: str = Field(
        default="",
        description="Plain-language summary tying the per-atom verdicts together.",
    )
