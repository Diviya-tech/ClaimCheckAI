"""Source quality classifier.

Assigns every retrieved source a quality tier (1 = most authoritative, 4 =
general web) using two signals:

  1. **Domain matching** — the source's host is matched against the tier registries
     in ``config.settings`` (subdomains included).
  2. **Publication-type detection** — a PubMed article is only Tier 1 when it is a
     systematic review or meta-analysis; an ordinary peer-reviewed study is Tier 2.
     Callers pass the article's PubMed publication types (or we fall back to
     sniffing the title) so we can promote the strongest evidence.

Spends no LLM tokens — pure string/domain logic. Returns the tier plus a short
human-readable justification for the dossier's transparency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlparse

from config.settings import (
    TIER_1_DOMAINS,
    TIER_2_DOMAINS,
    TIER_3_DOMAINS,
    SourceTier,
)

# PubMed publication types (and title keywords) that elevate a study to Tier 1.
_TIER_1_PUB_TYPES = {"meta-analysis", "systematic review"}

# NCBI literature repositories. A regular article here is Tier 2; promotion to
# Tier 1 happens via the global systematic-review/meta-analysis rule below.
_PUBMED_HOSTS = {"pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}

# .edu spam heuristics. University domains are routinely hijacked (abandoned
# student pages, compromised CMS uploads, open redirects) to host SEO spam, and
# that content must not inherit the university's Tier-3 credibility. A .edu URL
# whose path/query shows any of these signals is demoted to Tier 4.
#
# Another URL injected into the path or query string (raw, percent-encoded, or a
# bare "www."/TLD-like token). Only the path + query are checked, never the host.
_INJECTED_URL_RE = re.compile(
    r"(https?:|https?%3a|(?:^|[/=&?])www\.|[a-z0-9-]+\.(?:com|net|org|info|biz|xyz|top|"
    r"club|shop|online|site|store|ru|cn)(?:[/?&=%]|$))",
    re.IGNORECASE,
)
# Marketing / affiliate tracking parameters — legitimate university health pages
# don't carry campaign or affiliate tags.
_MARKETING_PARAM_RE = re.compile(
    r"^(utm_\w+|gclid|fbclid|msclkid|dclid|ref|referrer|referral|affiliate|aff|"
    r"aff_id|affid|promo|promocode|coupon|discount|campaign|cid|partner)$",
    re.IGNORECASE,
)
# Redirect parameters and path segments (open redirects are the classic vector).
_REDIRECT_PARAM_RE = re.compile(
    r"^(\w+_)?(url|uri|redirect|redir|redirect_to|goto|go|dest|destination|target|"
    r"next|return|returnto|link|out|forward|jump)$",
    re.IGNORECASE,
)
_REDIRECT_PATH_SEGMENTS = {"redirect", "redir", "go", "goto", "out", "link", "click", "jump"}


@dataclass
class SourceClassification:
    """A tier assignment plus the reasoning behind it."""

    tier: int
    justification: str


def classify_source(
    url: str,
    source_name: str = "",
    publication_types: list[str] | None = None,
    title: str = "",
) -> SourceClassification:
    """Classify a source into a quality tier (1-4) with a justification.

    Args:
        url: The source URL (its host drives domain matching).
        source_name: Optional human-readable name (journal/site); also sniffed for
            systematic-review/meta-analysis keywords as a fallback.
        publication_types: Optional PubMed publication types (e.g.
            ``["Meta-Analysis", "Journal Article"]``) — the strongest Tier-1 signal.
        title: Optional article title; sniffed for "systematic review" /
            "meta-analysis" so a synthesis hosted anywhere is promoted to Tier 1.

    Returns:
        A `SourceClassification` with `tier` (int 1-4) and a `justification` string.
    """
    domain = _domain_of(url)
    pub_types_lower = {pt.lower() for pt in (publication_types or [])}

    # --- 1. Systematic reviews & meta-analyses are top-tier *regardless* of host. ---
    # Publication type is the reliable signal (PubMed-supplied); a title/name
    # keyword is the fallback for web results that carry no publication metadata.
    if _is_high_evidence(pub_types_lower, f"{title} {source_name}"):
        kind = next(
            (pt for pt in _TIER_1_PUB_TYPES if pt in pub_types_lower),
            "systematic review / meta-analysis",
        )
        return SourceClassification(
            SourceTier.TIER_1,
            f"{kind.title()} — top-tier synthesized evidence.",
        )

    # --- 2. Official guideline / systematic-review bodies (Cochrane, WHO, CDC). ---
    if _domain_in(domain, TIER_1_DOMAINS):
        return SourceClassification(
            SourceTier.TIER_1,
            f"{domain} is an official guideline / systematic-review body.",
        )

    # --- 3. PubMed / PMC repositories: peer-reviewed primary research baseline. ---
    if _host_in(domain, _PUBMED_HOSTS):
        return SourceClassification(
            SourceTier.TIER_2,
            "PubMed/PMC peer-reviewed study (primary research, not a synthesis).",
        )

    # --- 4. Major medical institutions & academic publishers. ---
    if _domain_in(domain, TIER_2_DOMAINS):
        return SourceClassification(
            SourceTier.TIER_2,
            f"{domain} is a major medical institution / peer-reviewed publisher.",
        )

    # --- 5. Credentialed journalism & university health centers. ---
    # A .edu host only earns Tier 3 when the URL itself looks like university
    # content; hijacked/spam pages on university domains fall through to Tier 4.
    if domain.endswith(".edu"):
        spam_signal = _edu_spam_signal(url)
        if spam_signal:
            return SourceClassification(
                SourceTier.TIER_4,
                f"{domain} is a university domain but the URL shows spam indicators "
                f"({spam_signal}) — likely hijacked content; context only, not evidence.",
            )
    if _domain_in(domain, TIER_3_DOMAINS) or domain.endswith(".edu"):
        why = (
            "university health center"
            if domain.endswith(".edu")
            else "credentialed health journalism"
        )
        return SourceClassification(
            SourceTier.TIER_3, f"{domain or source_name} is {why}."
        )

    # --- 6. Everything else is general web / context only. ---
    label = domain or source_name or "unknown source"
    return SourceClassification(
        SourceTier.TIER_4,
        f"{label} is general web / blog / social media — context only, not evidence.",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_high_evidence(pub_types_lower: set[str], source_name: str) -> bool:
    """True if publication types (or, as a fallback, the name) indicate Tier 1."""
    if pub_types_lower & _TIER_1_PUB_TYPES:
        return True
    name = source_name.lower()
    return any(kw in name for kw in _TIER_1_PUB_TYPES)


def _edu_spam_signal(url: str) -> str | None:
    """Return a short description of the spam indicator found in `url`, or None.

    Checks only the path and query string (never the host) for injected external
    URLs, marketing/affiliate parameters, and redirect parameters or segments.
    """
    parsed = urlparse(url if "//" in url else f"//{url}")
    path, query = parsed.path or "", parsed.query or ""

    if _INJECTED_URL_RE.search(f"{path}?{query}"):
        return "injected external URL"

    segments = {seg.lower() for seg in path.split("/") if seg}
    if segments & _REDIRECT_PATH_SEGMENTS:
        return "redirect path"

    for key, _ in parse_qsl(query, keep_blank_values=True):
        if _MARKETING_PARAM_RE.match(key):
            return f"marketing parameter '{key}'"
        if _REDIRECT_PARAM_RE.match(key):
            return f"redirect parameter '{key}'"
    return None


def _domain_of(url: str) -> str:
    """Return the lowercased host of `url`, sans any leading 'www.'.

    Falls back to treating a bare host (no scheme) as the domain.
    """
    parsed = urlparse(url if "//" in url else f"//{url}")
    host = parsed.netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _host_in(domain: str, hosts: set[str]) -> bool:
    return any(domain == h or domain.endswith("." + h) for h in hosts)


def _domain_in(domain: str, registry) -> bool:
    """True if `domain` equals or is a subdomain of any host in `registry`."""
    return any(domain == d or domain.endswith("." + d) for d in registry)
