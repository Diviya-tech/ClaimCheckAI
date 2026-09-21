"""PubMed source — searches the biomedical literature via NCBI E-utilities.

This is part of the *evidence retrieval* layer (weeks 5-6) and spends ZERO LLM
tokens: it is a pure API client. Given a search query (already phrased for
PubMed by the retriever), it runs an ``esearch`` to find PMIDs and an ``efetch``
to pull each article's metadata + abstract, then returns structured
``PubMedArticle`` records.

Each record carries both the full abstract and a compressed extractive summary
(the first 2-3 sentences) so downstream stages can send the cheap summary to the
LLM and keep the full text on hand for citation.

Rate limits: NCBI permits 3 requests/second without an API key and 10/second
with one. We pace requests accordingly (see ``config.settings.NCBI_API_KEY``).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from config import settings

logger = logging.getLogger("claimcheck.sources.pubmed")

PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# Seconds to wait between successive NCBI calls to stay under the rate limit.
# 3 req/s without a key, 10 req/s with one -> use the inverse plus a margin.
_DELAY_WITHOUT_KEY = 0.34
_DELAY_WITH_KEY = 0.11


class PubMedError(Exception):
    """Raised when a PubMed request fails (network, timeout, rate-limit, parse).

    "No results" is *not* an error — ``search_pubmed`` returns an empty list for
    that. This is reserved for genuine failures so the retriever can log them and
    fall back to other sources rather than crashing the whole pipeline.
    """


@dataclass
class PubMedArticle:
    """One PubMed record, normalized for the evidence retriever."""

    pmid: str
    title: str
    abstract: str
    summary: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    publication_date: datetime | None = None
    publication_types: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    is_animal_study: bool = False

    @property
    def url(self) -> str:
        return PUBMED_URL_TEMPLATE.format(pmid=self.pmid)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
# URL patterns for recognizing PubMed-family links coming back from web search.
_PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.IGNORECASE)
_PMCID_URL_RE = re.compile(
    r"(?:pmc\.ncbi\.nlm\.nih\.gov/articles/|ncbi\.nlm\.nih\.gov/pmc/articles/)(PMC\d+)",
    re.IGNORECASE,
)


def pmid_from_url(url: str) -> str | None:
    """Return the PMID embedded in a pubmed.ncbi.nlm.nih.gov URL, else None."""
    match = _PMID_URL_RE.search(url or "")
    return match.group(1) if match else None


def pmcid_from_url(url: str) -> str | None:
    """Return the PMCID (e.g. 'PMC5065707') embedded in a PMC URL, else None."""
    match = _PMCID_URL_RE.search(url or "")
    return match.group(1).upper() if match else None


def search_pubmed(query: str, max_results: int = 10) -> list[PubMedArticle]:
    """Search PubMed for `query` and return up to `max_results` structured articles.

    Args:
        query: A search string already phrased for PubMed (the retriever builds
            these from atomic facts via the lightweight LLM).
        max_results: Maximum number of articles to fetch.

    Returns:
        A list of `PubMedArticle` (possibly empty if nothing matched). Articles
        with no abstract are skipped — they offer nothing to evaluate.

    Raises:
        PubMedError: on network/timeout/rate-limit/parse failures.
        ValueError: if `query` is empty or `max_results` is not positive.
    """
    if not query or not query.strip():
        raise ValueError("PubMed query must be a non-empty string.")
    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    Entrez = _entrez()  # noqa: N806 - matches Bio.Entrez's own casing

    pmids = _esearch(Entrez, query.strip(), max_results)
    if not pmids:
        logger.debug("PubMed returned no results for %r.", query)
        return []

    records = _efetch(Entrez, pmids)
    return _parse_records(records)


def fetch_article_by_pmid(pmid: str) -> PubMedArticle | None:
    """Fetch a single article's clean metadata + abstract by PMID.

    Used by the retriever to replace a web-scraped PubMed page (which often
    captures nav chrome instead of the abstract) with the canonical API record.

    Returns:
        The `PubMedArticle`, or None if it has no abstract / can't be parsed.

    Raises:
        PubMedError: on network/timeout/rate-limit/parse failure.
        ValueError: if `pmid` is empty.
    """
    if not pmid or not str(pmid).strip():
        raise ValueError("pmid must be a non-empty string.")
    Entrez = _entrez()  # noqa: N806
    records = _efetch(Entrez, [str(pmid).strip()])
    articles = _parse_records(records)
    return articles[0] if articles else None


def convert_pmcid_to_pmid(pmcid: str) -> str | None:
    """Resolve a PMCID (e.g. 'PMC5065707') to its PMID via Entrez elink.

    PMC URLs carry a PMCID, not a PMID, so we ask NCBI for the linked PubMed
    record before fetching the clean abstract. Returns None when no link exists
    or the lookup fails (the caller then keeps the original web result).
    """
    if not pmcid:
        return None
    numeric = pmcid.upper().removeprefix("PMC").strip()
    if not numeric:
        return None

    Entrez = _entrez()  # noqa: N806
    try:
        handle = Entrez.elink(dbfrom="pmc", db="pubmed", id=numeric)
        result = Entrez.read(handle)
        handle.close()
    except Exception as exc:  # network / parse — non-fatal, just give up the link
        logger.warning("PMCID->PMID lookup failed for %s: %s", pmcid, exc)
        return None
    time.sleep(_rate_limit_delay())

    try:
        for linkset in result:
            for db in linkset.get("LinkSetDb", []):
                links = db.get("Link", [])
                if links:
                    return str(links[0]["Id"])
    except (IndexError, KeyError, TypeError):
        pass
    return None


# --------------------------------------------------------------------------- #
# NCBI plumbing
# --------------------------------------------------------------------------- #
def _entrez():
    """Import Bio.Entrez lazily and apply the required/optional global config.

    Imported here (not at module top) so the package imports fine without
    Biopython installed — only callers that actually search PubMed need it.
    """
    try:
        from Bio import Entrez  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PubMedError(
            "biopython is not installed. Run: pip install -r requirements.txt"
        ) from exc

    # NCBI requires an email on every request; an API key is optional but raises
    # the rate limit. Both are global module state on Bio.Entrez.
    Entrez.email = settings.ENTREZ_EMAIL
    if settings.NCBI_API_KEY:
        Entrez.api_key = settings.NCBI_API_KEY
    return Entrez


def _rate_limit_delay() -> float:
    return _DELAY_WITH_KEY if settings.NCBI_API_KEY else _DELAY_WITHOUT_KEY


def _esearch(Entrez, query: str, max_results: int) -> list[str]:  # noqa: N803
    """Run esearch and return the matching PMIDs (most relevant first)."""
    try:
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results, sort="relevance"
        )
        result = Entrez.read(handle)
        handle.close()
    except Exception as exc:  # network, HTTP 429, XML parse, etc.
        raise PubMedError(f"PubMed esearch failed for {query!r}: {exc}") from exc

    time.sleep(_rate_limit_delay())
    return list(result.get("IdList", []))


def _efetch(Entrez, pmids: list[str]):  # noqa: N803
    """Fetch full records for `pmids` as parsed Entrez structures."""
    try:
        handle = Entrez.efetch(
            db="pubmed", id=",".join(pmids), rettype="abstract", retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()
    except Exception as exc:
        raise PubMedError(f"PubMed efetch failed for {pmids}: {exc}") from exc

    time.sleep(_rate_limit_delay())
    return records


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_records(records) -> list[PubMedArticle]:
    """Turn raw Entrez efetch output into `PubMedArticle`s, skipping abstract-less ones."""
    articles: list[PubMedArticle] = []
    for entry in records.get("PubmedArticle", []):
        try:
            article = _parse_one(entry)
        except Exception as exc:  # one malformed record shouldn't drop the batch
            logger.warning("Skipping unparseable PubMed record: %s", exc)
            continue
        if article is not None:
            articles.append(article)
    return articles


def _parse_one(entry) -> PubMedArticle | None:
    citation = entry["MedlineCitation"]
    article = citation["Article"]

    pmid = str(citation["PMID"])
    title = _clean(str(article.get("ArticleTitle", "")).strip())

    abstract = _join_abstract(article.get("Abstract", {}).get("AbstractText", []))
    if not abstract:
        # No abstract -> nothing to evaluate; drop it.
        return None

    mesh_terms = _parse_mesh_terms(citation.get("MeshHeadingList", []))

    return PubMedArticle(
        pmid=pmid,
        title=title,
        abstract=abstract,
        summary=summarize_abstract(abstract),
        authors=_parse_authors(article.get("AuthorList", [])),
        journal=str(article.get("Journal", {}).get("Title", "")).strip(),
        publication_date=_parse_pub_date(article),
        publication_types=_parse_pub_types(article.get("PublicationTypeList", [])),
        mesh_terms=mesh_terms,
        is_animal_study=is_animal_study(mesh_terms, f"{title} {abstract}"),
    )


def _join_abstract(abstract_text) -> str:
    """Abstract may be a list of (possibly labeled) sections; join into one string."""
    if not abstract_text:
        return ""
    parts: list[str] = []
    for chunk in abstract_text:
        # Structured abstracts come as StringElement with a 'Label' attribute
        # (e.g. "BACKGROUND", "RESULTS"); preserve the label for readability.
        label = getattr(chunk, "attributes", {}).get("Label") if hasattr(chunk, "attributes") else None
        text = str(chunk).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return _clean(" ".join(parts))


def _parse_authors(author_list) -> list[str]:
    names: list[str] = []
    for author in author_list:
        last = str(author.get("LastName", "")).strip()
        initials = str(author.get("Initials", "")).strip()
        if last:
            names.append(f"{last} {initials}".strip())
        elif author.get("CollectiveName"):
            names.append(str(author["CollectiveName"]).strip())
    return names


def _parse_pub_types(pub_type_list) -> list[str]:
    return [str(pt).strip() for pt in pub_type_list if str(pt).strip()]


def _parse_mesh_terms(mesh_heading_list) -> list[str]:
    """Flatten a MeshHeadingList into descriptor names (e.g. ["Humans", "Rats"])."""
    terms: list[str] = []
    for heading in mesh_heading_list or []:
        descriptor = heading.get("DescriptorName") if hasattr(heading, "get") else None
        name = str(descriptor).strip() if descriptor is not None else ""
        if name:
            terms.append(name)
    return terms


# --------------------------------------------------------------------------- #
# Animal / in-vitro study detection (token-free)
# --------------------------------------------------------------------------- #
# MeSH indexes every PubMed article with "Humans" and/or "Animals" (plus species
# such as "Rats", "Mice"). That is the reliable signal: an article tagged
# Animals but NOT Humans studied animals only. Cell-culture work has no
# organism tag, so a keyword fallback on the title/abstract catches it, along
# with recent articles MeSH hasn't indexed yet.
_ANIMAL_MESH = {"animals", "rats", "mice", "rabbits", "dogs", "swine", "zebrafish",
                "drosophila", "caenorhabditis elegans", "rodentia", "mice, inbred c57bl"}
_NON_HUMAN_TEXT_RE = re.compile(
    r"\b(in (?:male|female|adult|young|aged|obese|diabetic|healthy)? ?(?:mice|rats|rodents|"
    r"rabbits|zebrafish|dogs|pigs|piglets|hamsters|guinea pigs|sheep|c57bl/6|wistar|"
    r"sprague[- ]dawley)|murine|rodent model|animal model|in vitro|cell culture|"
    r"cell line|cultured cells|hepg2|caco-2|raw ?264\.7|3t3-l1)\b",
    re.IGNORECASE,
)
_HUMAN_TEXT_RE = re.compile(
    r"\b(participants|patients|volunteers|subjects|men|women|adults|children|"
    r"randomi[sz]ed|placebo|double-blind|cohort|clinical trial)\b",
    re.IGNORECASE,
)


def is_animal_study(mesh_terms: list[str], text: str = "") -> bool:
    """True if the article studied animals / cells and not humans.

    MeSH wins when present: "Humans" anywhere -> human study; "Animals" (or a
    species) without "Humans" -> animal study. Without MeSH, fall back to the
    text: non-human keywords with no human-study vocabulary -> animal/in-vitro.
    """
    lowered = {t.lower() for t in mesh_terms}
    if "humans" in lowered:
        return False
    if lowered & _ANIMAL_MESH:
        return True
    if not text:
        return False
    return bool(_NON_HUMAN_TEXT_RE.search(text)) and not _HUMAN_TEXT_RE.search(text)


def _parse_pub_date(article) -> datetime | None:
    """Best-effort parse of the article's publication date (year is enough)."""
    issue = article.get("Journal", {}).get("JournalIssue", {})
    pub_date = issue.get("PubDate", {})
    year = pub_date.get("Year")
    if not year:
        return None
    month = _MONTHS.get(str(pub_date.get("Month", "1"))[:3].title(), 1)
    if isinstance(pub_date.get("Month"), str) and pub_date["Month"].isdigit():
        month = int(pub_date["Month"])
    day = int(pub_date["Day"]) if str(pub_date.get("Day", "")).isdigit() else 1
    try:
        return datetime(int(year), month, day)
    except (ValueError, TypeError):
        return None


_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Sentence splitter: end punctuation followed by whitespace + a capital/number.
# Deliberately simple — extractive summaries don't need linguistic perfection.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def summarize_abstract(abstract: str, max_sentences: int = 3, max_chars: int = 600) -> str:
    """Compress an abstract to its first few sentences (no LLM tokens spent).

    Evidence retrieval must stay token-free, so this is extractive, not abstractive:
    we take the leading 2-3 sentences (capped by length), which for a structured
    abstract is usually the background + objective — enough for relevance ranking
    and a citation snippet. The full abstract is kept separately for the verdict
    stage.
    """
    text = _clean(abstract)
    if not text:
        return ""
    sentences = _SENTENCE_BOUNDARY.split(text)
    summary = " ".join(sentences[:max_sentences]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip() + "…"
    return summary


def _clean(text: str) -> str:
    """Collapse internal whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip()
