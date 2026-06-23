"""Central configuration for ClaimCheck AI.

Loads secrets from `.env`, defines the source-quality tiers, and declares the
two-tier model routing config. Everything that needs an API key, a model name,
or a tier definition should import it from here rather than hardcoding it.
"""

from __future__ import annotations

import os
from enum import IntEnum

from dotenv import load_dotenv

# Load variables from a local .env file into the environment (no-op if absent).
load_dotenv()


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")


# --------------------------------------------------------------------------- #
# Source quality tiers
# --------------------------------------------------------------------------- #
class SourceTier(IntEnum):
    """Evidence source reliability, lower is more authoritative.

    Mirrors the tiers in CLAUDE.md. Stored as small ints so they can be
    compared (`tier <= SourceTier.TIER_2`) and serialized cleanly.
    """

    TIER_1 = 1  # Cochrane reviews, PubMed meta-analyses, WHO, CDC
    TIER_2 = 2  # PubMed peer-reviewed studies, NIH MedlinePlus, Mayo, Cleveland Clinic
    TIER_3 = 3  # Credentialed health journalism, university health centers
    TIER_4 = 4  # General web, blogs, social media (context only, not evidence)


# Human-readable descriptions, handy for dossier rendering and debugging.
SOURCE_TIER_LABELS: dict[SourceTier, str] = {
    SourceTier.TIER_1: "Systematic reviews & official guidelines (Cochrane, WHO, CDC, PubMed meta-analyses)",
    SourceTier.TIER_2: "Peer-reviewed studies & major medical institutions (PubMed, NIH, Mayo, Cleveland Clinic)",
    SourceTier.TIER_3: "Credentialed health journalism & university health centers",
    SourceTier.TIER_4: "General web, blogs & social media (context only, not evidence)",
}


# --------------------------------------------------------------------------- #
# Two-tier model routing
# --------------------------------------------------------------------------- #
# "premium"     -> reasoning-heavy work (evidence evaluation, verdict generation)
# "lightweight" -> cheap, high-volume work (claim extraction, classification)
#
# All model identifiers are kept here so pipeline code never names a model
# directly. To swap providers/models, edit this dict only.
MODEL_CONFIG: dict[str, dict[str, object]] = {
    "premium": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
    },
    "lightweight": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
    },
}

# Default provider used when constructing clients. Add "openai" support later by
# extending config/providers.py — pipeline code does not need to change.
DEFAULT_PROVIDER: str = "anthropic"


# --------------------------------------------------------------------------- #
# Cost guardrails
# --------------------------------------------------------------------------- #
# Soft daily token ceiling (input + output, summed across tiers). When exceeded,
# providers.llm_call logs a warning and raises BudgetWarning. It is intentionally
# a *soft* limit — callers may catch the warning and continue.
DAILY_TOKEN_BUDGET: int = int(os.getenv("DAILY_TOKEN_BUDGET", "500000"))

# File where rolling daily token usage is persisted (gitignored).
DAILY_USAGE_FILE: str = os.getenv("DAILY_USAGE_FILE", "daily_usage.json")


def require_anthropic_key() -> str:
    """Return the Anthropic key or raise a clear error if it is missing.

    Call this at the point of use so importing the module never fails just
    because `.env` hasn't been filled in yet.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ANTHROPIC_API_KEY
