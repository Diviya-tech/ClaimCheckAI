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

    content: str = Field(..., description="The evidence text / summary.")
    source_url: str = Field(..., description="Canonical URL of the source.")
    source_name: str = Field(..., description="Human-readable source name (e.g. 'Cochrane').")
    source_tier: int = Field(..., ge=1, le=4, description="Source quality tier, 1 (best) to 4.")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance (e.g. vector similarity), 0-1.",
    )
    publication_date: datetime | None = Field(
        default=None,
        description="When the source was published, if known.",
    )


class AtomicVerdict(BaseModel):
    """A categorical verdict for one atomic fact, with the evidence behind it."""

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
    reasoning: str = Field(
        ...,
        description="Transparent explanation of how the verdict follows from the evidence.",
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
    rhetorical_flags: list[str] = Field(
        default_factory=list,
        description="Detected rhetorical / manipulation patterns (additive module, later).",
    )
    narrative_summary: str = Field(
        default="",
        description="Plain-language summary tying the per-atom verdicts together.",
    )
