"""Supplementary web search via Tavily.

Part of the evidence retrieval layer (weeks 5-6); spends ZERO LLM tokens. PubMed
is the primary source; this fills gaps with authoritative health sites (WHO, CDC,
Mayo, Cleveland Clinic, NIH, Cochrane, NHS, ...) and general web context.

We don't *restrict* Tavily to medical domains — general results still provide
useful context (and Tier-4 framing for the dossier) — but we re-rank so results
from recognized medical/health domains float to the top.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from config import settings

logger = logging.getLogger("claimcheck.sources.web_search")


class WebSearchError(Exception):
    """Raised when the web search can't run (missing key, client error, network)."""


@dataclass
class WebResult:
    """One web search hit, normalized for the evidence retriever."""

    title: str
    content: str
    url: str
    domain: str
    score: float = 0.0  # Tavily relevance score, 0-1 (0 if absent).

    @property
    def is_medical_domain(self) -> bool:
        return _domain_in(self.domain, settings.MEDICAL_DOMAINS)


def search_web(query: str, max_results: int = 5) -> list[WebResult]:
    """Search the web for `query`, prioritizing medical/health domains.

    Args:
        query: Free-text search string (typically the atomic fact, lightly rephrased).
        max_results: Maximum number of results to return.

    Returns:
        A list of `WebResult` ordered with medical-domain hits first, then by
        Tavily's own relevance score. Possibly empty.

    Raises:
        WebSearchError: if TAVILY_API_KEY is unset, the client is missing, or the
            request fails.
        ValueError: if `query` is empty or `max_results` is not positive.
    """
    if not query or not query.strip():
        raise ValueError("Web search query must be a non-empty string.")
    if max_results <= 0:
        raise ValueError("max_results must be positive.")
    if not settings.TAVILY_API_KEY:
        raise WebSearchError(
            "TAVILY_API_KEY is not set. Add it to .env to enable supplementary web search."
        )

    client = _client()

    try:
        # Ask for extra results so re-ranking by domain has something to work with,
        # then trim to max_results after sorting.
        response = client.search(
            query=query.strip(),
            max_results=max(max_results * 2, max_results),
            search_depth="basic",
        )
    except Exception as exc:
        raise WebSearchError(f"Tavily search failed for {query!r}: {exc}") from exc

    results = [_to_result(item) for item in response.get("results", [])]
    results = _rank(results)
    return results[:max_results]


def _client():
    """Lazily import and construct the Tavily client."""
    try:
        from tavily import TavilyClient  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise WebSearchError(
            "tavily-python is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def _to_result(item: dict) -> WebResult:
    url = str(item.get("url", "")).strip()
    return WebResult(
        title=str(item.get("title", "")).strip(),
        content=str(item.get("content", "")).strip(),
        url=url,
        domain=_domain_of(url),
        score=float(item.get("score", 0.0) or 0.0),
    )


def _rank(results: list[WebResult]) -> list[WebResult]:
    """Medical-domain hits first; within each group, higher Tavily score first."""
    return sorted(results, key=lambda r: (not r.is_medical_domain, -r.score))


def _domain_of(url: str) -> str:
    """Return the lowercased host of `url`, sans any leading 'www.'."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _domain_in(domain: str, registry) -> bool:
    """True if `domain` equals or is a subdomain of any host in `registry`."""
    domain = domain.lower()
    return any(domain == d or domain.endswith("." + d) for d in registry)
