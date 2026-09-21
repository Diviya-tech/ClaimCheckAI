"""Evidence retriever — stage 3 of the pipeline (weeks 5-6).

Takes the atomic facts from the claim extractor and gathers evidence for each one
from trusted medical sources. The retrieval itself spends ZERO LLM tokens — it is
PubMed (Bio.Entrez) + Tavily web search + deterministic ranking. The *only* LLM
use here is the lightweight model turning each atomic fact into effective search
queries (e.g. "cumin water causes weight loss" -> "cumin supplementation weight
loss clinical trial"), which is cheap and high-leverage.

Per atomic fact the retriever:
  1. generates 1-3 search queries (lightweight LLM, with a token-free fallback),
  2. searches PubMed (primary) across those queries, de-duplicating by PMID,
  3. searches Tavily (supplementary), prioritizing medical domains,
  4. classifies every source into a quality tier,
  5. scores each by lexical relevance and tier, and
  6. keeps the top few (3-5) — small on purpose, for token efficiency downstream.

Output is a list of `FactEvidence` (one per fact) so the verdict engine can
evaluate each fact against exactly the evidence gathered for it.
"""

from __future__ import annotations

import logging
import re

from config import providers
from core.source_classifier import classify_source
from schemas.models import AtomicFact, Evidence, EvidenceApplicability, FactEvidence
from sources import pubmed, web_search

logger = logging.getLogger("claimcheck.evidence_retriever")

# How many results to keep per atomic fact after ranking (top 3-5, per CLAUDE.md).
DEFAULT_PER_FACT_LIMIT = 5

# How many raw results to pull from each source before ranking/trimming.
_PUBMED_FETCH = 8
_WEB_FETCH = 5

# Minimum relevance an item must clear to count as evidence at all. Below this it
# is topical noise — a Tier-1 systematic review about an unrelated condition is
# useless for evaluating this claim — so we drop it *before* the tier bonus is
# applied. Tier should boost relevant results, never rescue irrelevant ones.
RELEVANCE_FLOOR = 0.15

# When fewer than this many items survive the floor, we flag the fact rather than
# present a thin/noisy result set as if it were solid evidence.
_MIN_SUFFICIENT_EVIDENCE = 2
_INSUFFICIENT_EVIDENCE_NOTE = "insufficient evidence found"

# Tier bonus blended with relevance during ranking: authoritative sources get a
# head start, but a highly-relevant lower-tier result can still outrank a barely
# relevant top-tier one. Keeps "rank by relevance AND tier" honest.
_TIER_BONUS = {1: 0.30, 2: 0.20, 3: 0.10, 4: 0.0}

# Animal / in-vitro studies are kept (they are real context and the dossier
# should show them) but never allowed to crowd out human evidence: they take a
# ranking penalty, at most this many survive per fact, and they don't count
# toward the "enough evidence to evaluate" threshold. A claim aimed at people
# cannot be established by rats.
_NON_HUMAN_PENALTY = 0.25
_MAX_NON_HUMAN_PER_FACT = 1

# User-readable notes for degraded retrieval. Attached to every affected fact and
# bubbled up to `Dossier.limitations` so the reader knows what the verdict lacks.
NOTE_PUBMED_DOWN = (
    "PubMed was unavailable during this check; evidence came from web sources only."
)
NOTE_WEB_SKIPPED = (
    "Supplementary web search was skipped (no TAVILY_API_KEY configured); "
    "evidence came from PubMed only."
)
NOTE_WEB_DOWN = (
    "Supplementary web search was unavailable during this check; evidence came "
    "from PubMed only."
)
NOTE_ALL_SOURCES_DOWN = (
    "No evidence source was reachable during this check; verdicts are based on "
    "no retrieved evidence."
)


# --------------------------------------------------------------------------- #
# Search-query generation (the ONLY LLM use in this stage)
# --------------------------------------------------------------------------- #
_QUERY_SYSTEM_PROMPT = """\
You convert a single atomic health claim into effective literature-search queries.

Produce 1-3 short search queries that would surface clinical evidence about the
claim in PubMed and on authoritative health sites. Use the terminology a
researcher would: prefer scientific names, intervention + outcome + study-type
keywords (e.g. "randomized controlled trial", "meta-analysis", "clinical trial"),
and drop marketing language and hedging.

Example: "cumin water melts belly fat in two weeks" ->
  ["cumin supplementation weight loss clinical trial",
   "cuminum cyminum body weight randomized controlled trial"]

Do NOT answer or judge the claim. Only return search queries.
"""

_QUERY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "description": "1-3 search queries optimized for medical literature retrieval.",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}


def _generate_search_queries(fact: AtomicFact, allow_over_budget: bool = False) -> list[str]:
    """Rephrase an atomic fact into effective search queries (lightweight LLM).

    Falls back to the raw fact text — keeping retrieval fully token-free — if the
    LLM is unavailable or the daily budget is exhausted. Search must never be
    blocked just because query optimization couldn't run.
    """
    try:
        result = providers.llm_call(
            prompt=f"Atomic claim: {fact.text}",
            system_prompt=_QUERY_SYSTEM_PROMPT,
            model_tier="lightweight",
            response_schema=_QUERY_SCHEMA,
            allow_over_budget=allow_over_budget,
        )
        assert isinstance(result, dict)
        queries = [q.strip() for q in result.get("queries", []) if q and q.strip()]
        if queries:
            return queries
        logger.warning("Query generation returned no queries; using raw fact text.")
    except providers.BudgetWarning:
        logger.warning(
            "Budget exhausted before query generation; falling back to raw fact text "
            "(retrieval continues token-free)."
        )
    except Exception as exc:  # never let query-gen failure kill retrieval
        logger.warning("Query generation failed (%s); using raw fact text.", exc)

    return [fact.text]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def retrieve_evidence(
    atomic_facts: list[AtomicFact],
    per_fact_limit: int = DEFAULT_PER_FACT_LIMIT,
    allow_over_budget: bool = False,
) -> list[FactEvidence]:
    """Gather and rank evidence for each atomic fact.

    Args:
        atomic_facts: The decomposed facts from the claim extractor.
        per_fact_limit: Max evidence items to keep per fact after ranking (3-5).
        allow_over_budget: Forwarded to the (single, lightweight) query-gen LLM call.

    Returns:
        One `FactEvidence` per input fact, each holding up to `per_fact_limit`
        ranked `Evidence` items (best first). A fact for which every source
        failed or returned nothing yields an empty evidence list — never an error.
    """
    results: list[FactEvidence] = []
    for fact in atomic_facts:
        queries = _generate_search_queries(fact, allow_over_budget=allow_over_budget)
        logger.debug("Fact %r -> queries %s", fact.text, queries)

        # PubMed first; the PMIDs it returns let us de-dup (and clean up) any
        # PubMed/PMC links the web search later turns up.
        pubmed_evidence, known_pmids, pubmed_note = _search_pubmed(queries, fact.text)
        web_evidence, web_note = _search_web(queries, known_pmids, fact.text)

        evidence = _dedupe_by_url(pubmed_evidence + web_evidence)

        # Drop topical noise BEFORE ranking so the tier bonus can't float an
        # irrelevant high-tier source to the top.
        relevant = [e for e in evidence if e.relevance_score >= RELEVANCE_FLOOR]
        relevant.sort(key=_rank_score, reverse=True)
        top = _cap_non_human(relevant)[:per_fact_limit]

        human_count = sum(1 for e in top if not _is_non_human(e))
        note = "" if human_count >= _MIN_SUFFICIENT_EVIDENCE else _INSUFFICIENT_EVIDENCE_NOTE
        if note:
            logger.info(
                "Fact %r: only %d relevant human-study item(s) cleared the floor (%s).",
                fact.text, human_count, note,
            )

        results.append(
            FactEvidence(
                atomic_fact=fact,
                evidence=top,
                retrieval_note=note,
                source_notes=_merge_source_notes(pubmed_note, web_note),
            )
        )

    return results


def collect_limitations(fact_evidence: list[FactEvidence]) -> list[str]:
    """Unique, order-preserving source notes across all facts (for the dossier)."""
    seen: set[str] = set()
    notes: list[str] = []
    for fe in fact_evidence:
        for note in fe.source_notes:
            if note and note not in seen:
                seen.add(note)
                notes.append(note)
    return notes


def _merge_source_notes(pubmed_note: str, web_note: str) -> list[str]:
    """Collapse per-source failure notes into what the reader needs to know."""
    if pubmed_note and web_note:
        return [NOTE_ALL_SOURCES_DOWN]
    return [n for n in (pubmed_note, web_note) if n]


# --------------------------------------------------------------------------- #
# Per-source search (each resilient — a failing source is logged, not fatal)
# --------------------------------------------------------------------------- #
def _search_pubmed(
    queries: list[str], fact_text: str
) -> tuple[list[Evidence], set[str], str]:
    """Search PubMed across all queries, de-duping by PMID.

    Returns the Evidence list, the set of PMIDs seen (so the web search can
    recognize and drop/clean PubMed links pointing at the same articles), and a
    user-readable note if PubMed was down for EVERY query (a partial failure
    with some results still counts as PubMed having answered).
    """
    seen_pmids: set[str] = set()
    evidence: list[Evidence] = []
    failures = 0
    for query in queries:
        try:
            articles = pubmed.search_pubmed(query, max_results=_PUBMED_FETCH)
        except (pubmed.PubMedError, ValueError) as exc:
            logger.warning("PubMed search failed for %r: %s", query, exc)
            failures += 1
            continue
        for article in articles:
            if article.pmid in seen_pmids:
                continue
            seen_pmids.add(article.pmid)
            evidence.append(_pubmed_to_evidence(article, fact_text))
    note = NOTE_PUBMED_DOWN if queries and failures == len(queries) else ""
    return evidence, seen_pmids, note


def _search_web(
    queries: list[str], known_pmids: set[str], fact_text: str
) -> tuple[list[Evidence], str]:
    """Search the web with the best (first) query and convert to Evidence.

    Returns the Evidence list and a user-readable note when the search could not
    run: distinguishes "not configured" (no key — a deliberate PubMed-only setup)
    from "down" (configured but failing).

    Web results that point at a PubMed/PMC article are reconciled with the PubMed
    API results instead of being trusted as-is (Tavily often scrapes the page's
    nav chrome rather than the abstract):
      * already have that PMID  -> drop the duplicate;
      * new PMID                -> re-fetch the clean abstract via the PubMed API;
      * can't resolve a PMID    -> keep the web result as a fallback.
    `known_pmids` is updated in place as new PubMed articles are pulled in.
    """
    if not queries:
        return [], ""
    try:
        hits = web_search.search_web(queries[0], max_results=_WEB_FETCH)
    except (web_search.WebSearchError, ValueError) as exc:
        logger.warning("Web search failed for %r: %s", queries[0], exc)
        note = NOTE_WEB_SKIPPED if "TAVILY_API_KEY" in str(exc) else NOTE_WEB_DOWN
        return [], note

    evidence: list[Evidence] = []
    for hit in hits:
        pmid = _resolve_pubmed_pmid(hit.url)
        if pmid is not None:
            if pmid in known_pmids:
                logger.debug("Dropping web duplicate of PMID %s (%s).", pmid, hit.url)
                continue
            article = _fetch_clean_article(pmid)
            if article is not None:
                known_pmids.add(pmid)
                evidence.append(_pubmed_to_evidence(article, fact_text))
                continue
            # Couldn't fetch the clean record — fall back to the web result.
        evidence.append(_web_to_evidence(hit))
    return evidence, ""


def _resolve_pubmed_pmid(url: str) -> str | None:
    """Return the PMID a web result points at (directly, or via PMCID lookup)."""
    pmid = pubmed.pmid_from_url(url)
    if pmid is not None:
        return pmid
    pmcid = pubmed.pmcid_from_url(url)
    if pmcid is None:
        return None
    try:
        return pubmed.convert_pmcid_to_pmid(pmcid)
    except pubmed.PubMedError as exc:
        logger.warning("PMCID->PMID conversion failed for %s: %s", pmcid, exc)
        return None


def _fetch_clean_article(pmid: str) -> pubmed.PubMedArticle | None:
    """Fetch a clean PubMed article by PMID, or None if it fails."""
    try:
        return pubmed.fetch_article_by_pmid(pmid)
    except (pubmed.PubMedError, ValueError) as exc:
        logger.warning("Clean re-fetch failed for PMID %s: %s", pmid, exc)
        return None


# --------------------------------------------------------------------------- #
# Source record -> Evidence
# --------------------------------------------------------------------------- #
def _pubmed_to_evidence(article: pubmed.PubMedArticle, fact_text: str) -> Evidence:
    classification = classify_source(
        article.url,
        source_name=article.journal,
        publication_types=article.publication_types,
        title=article.title,
    )
    return Evidence(
        content=article.abstract,
        summary=article.summary,
        source_url=article.url,
        source_name=article.journal or "PubMed",
        source_tier=classification.tier,
        relevance_score=_lexical_relevance(fact_text, f"{article.title} {article.summary}"),
        # Stamp animal / in-vitro work now (token-free); the verdict engine
        # decides direct vs indirect for the human studies.
        applicability=(
            EvidenceApplicability.NON_HUMAN
            if article.is_animal_study
            else EvidenceApplicability.UNKNOWN
        ),
        publication_date=article.publication_date,
    )


def _web_to_evidence(result: web_search.WebResult) -> Evidence:
    # Pass the article title so a synthesis (e.g. a Frontiers "... systematic
    # review and meta-analysis") is promoted to Tier 1 even without PubMed metadata.
    classification = classify_source(
        result.url, source_name=result.domain, title=result.title
    )
    # Tavily already returns a 0-1 relevance score; trust it for web results.
    relevance = min(max(result.score, 0.0), 1.0)
    return Evidence(
        content=result.content,
        summary=pubmed.summarize_abstract(result.content),
        source_url=result.url,
        source_name=result.domain or result.title or "web",
        source_tier=classification.tier,
        relevance_score=relevance,
    )


# --------------------------------------------------------------------------- #
# Ranking + de-duplication
# --------------------------------------------------------------------------- #
def _rank_score(evidence: Evidence) -> float:
    """Blend relevance with a tier head-start so authority and fit both count.

    Only applied to items that already cleared `RELEVANCE_FLOOR`, so the tier
    bonus orders *relevant* results — it never rescues irrelevant ones. Animal /
    in-vitro studies take a flat penalty so human evidence always sorts first.
    """
    score = evidence.relevance_score + _TIER_BONUS.get(evidence.source_tier, 0.0)
    if _is_non_human(evidence):
        score -= _NON_HUMAN_PENALTY
    return score


def _is_non_human(evidence: Evidence) -> bool:
    return evidence.applicability is EvidenceApplicability.NON_HUMAN


def _cap_non_human(ranked: list[Evidence]) -> list[Evidence]:
    """Keep at most `_MAX_NON_HUMAN_PER_FACT` animal/in-vitro items (best first)."""
    kept: list[Evidence] = []
    non_human_seen = 0
    for ev in ranked:
        if _is_non_human(ev):
            if non_human_seen >= _MAX_NON_HUMAN_PER_FACT:
                continue
            non_human_seen += 1
        kept.append(ev)
    return kept


def _dedupe_by_url(evidence: list[Evidence]) -> list[Evidence]:
    """Drop duplicate sources (same URL), keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[Evidence] = []
    for ev in evidence:
        key = ev.source_url.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        unique.append(ev)
    return unique


# Tiny stopword set — enough to stop "the/and/of" dominating short overlaps.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
        "with", "is", "are", "be", "that", "this", "it", "as", "at", "from",
        "your", "you", "can", "may", "will", "does", "do",
    }
)
_WORD = re.compile(r"[a-z0-9]+")


def _lexical_relevance(query: str, text: str) -> float:
    """Fraction of the query's content words that appear in `text` (0-1).

    A deliberately cheap, token-free relevance proxy for PubMed results (which,
    unlike Tavily, don't come with a score). Vector similarity via Qdrant replaces
    this in a later iteration; the interface — a 0-1 score — stays the same.
    """
    q_tokens = {w for w in _WORD.findall(query.lower()) if w not in _STOPWORDS}
    if not q_tokens:
        return 0.0
    t_tokens = {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}
    overlap = len(q_tokens & t_tokens)
    return round(overlap / len(q_tokens), 4)
