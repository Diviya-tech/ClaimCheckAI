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

# NCBI / PubMed (Bio.Entrez). NCBI *requires* an email on every request so they
# can contact you about misuse before blocking. An API key is optional but lifts
# the rate limit from 3 to 10 requests/second.
ENTREZ_EMAIL: str = os.getenv("ENTREZ_EMAIL", "claimcheck-ai@example.com")
NCBI_API_KEY: str | None = os.getenv("NCBI_API_KEY")


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

# Short names for the same tiers, sized to fit on a badge. "T1".."T4" means
# nothing to a reader who hasn't memorized the scale, so this is what the UI
# shows; SOURCE_TIER_LABELS above stays the longer explanatory form. Serialized
# on every Evidence as `human_readable_tier`, so clients never re-map numbers.
SOURCE_TIER_SHORT_LABELS: dict[SourceTier, str] = {
    SourceTier.TIER_1: "Systematic Review / Meta-analysis",
    SourceTier.TIER_2: "Peer-reviewed Study",
    SourceTier.TIER_3: "Medical Journalism",
    SourceTier.TIER_4: "General Web Source",
}


# --------------------------------------------------------------------------- #
# Source domain -> tier mapping (used by core/source_classifier.py)
# --------------------------------------------------------------------------- #
# Registered domains map deterministically to a tier. Matching is done on the
# *registrable* host and its subdomains (so "my.clevelandclinic.org" matches
# "clevelandclinic.org"). Anything unrecognized falls through to Tier 4.
#
# NOTE on PubMed: a PubMed article is Tier 1 *only* when it is a systematic
# review or meta-analysis; otherwise it is a Tier 2 peer-reviewed study. The
# classifier promotes pubmed.ncbi.nlm.nih.gov to Tier 1 based on publication
# type, so the static map lists it at its Tier 2 baseline.
TIER_1_DOMAINS: frozenset[str] = frozenset(
    {
        "cochranelibrary.com",
        "cochrane.org",
        "who.int",
        "cdc.gov",
    }
)

# Major medical institutions, national health bodies, and the NCBI repositories
# (PubMed / PMC). Peer-reviewed primary research / authoritative health info.
MEDICAL_INSTITUTION_DOMAINS: frozenset[str] = frozenset(
    {
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "nih.gov",
        "medlineplus.gov",
        "mayoclinic.org",
        "clevelandclinic.org",
        "nhs.uk",
        "hopkinsmedicine.org",
    }
)

# Academic publishers & journal platforms. A *regular* article here is Tier 2; a
# systematic review or meta-analysis is promoted to Tier 1 by the classifier
# (via publication-type / title detection) regardless of which publisher hosts it.
# Listed by registrable domain so subdomains match (e.g. link.springer.com,
# onlinelibrary.wiley.com, journals.plos.org, academic.oup.com).
ACADEMIC_PUBLISHER_DOMAINS: frozenset[str] = frozenset(
    {
        "frontiersin.org",     # Frontiers
        "sciencedirect.com",   # Elsevier / ScienceDirect
        "springer.com",        # Springer (link.springer.com)
        "wiley.com",           # Wiley (onlinelibrary.wiley.com)
        "bmj.com",             # BMJ
        "jamanetwork.com",     # JAMA Network
        "thelancet.com",       # The Lancet
        "nature.com",          # Nature
        "plos.org",            # PLOS (journals.plos.org)
        "oup.com",             # Oxford Academic (academic.oup.com)
        "tandfonline.com",     # Taylor & Francis
        "sagepub.com",         # SAGE (journals.sagepub.com)
        "peerj.com",           # PeerJ
        "mdpi.com",            # MDPI
    }
)

TIER_2_DOMAINS: frozenset[str] = MEDICAL_INSTITUTION_DOMAINS | ACADEMIC_PUBLISHER_DOMAINS

TIER_3_DOMAINS: frozenset[str] = frozenset(
    {
        "statnews.com",
        "health.harvard.edu",
        "medpagetoday.com",
        "kff.org",
        "healthcare.utah.edu",
    }
)

# Health/medical domains the web search should *prioritize* (re-rank to the top).
# Union of every tiered domain (institutions + academic publishers) plus a few
# more authoritative health/government sources.
MEDICAL_DOMAINS: frozenset[str] = (
    TIER_1_DOMAINS
    | TIER_2_DOMAINS
    | TIER_3_DOMAINS
    | frozenset({"medlineplus.gov", "cancer.gov", "fda.gov", "nibib.nih.gov"})
)


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
