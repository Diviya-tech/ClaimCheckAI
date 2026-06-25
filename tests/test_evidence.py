"""Evidence-retrieval test suite (weeks 5-6).

All tests run offline. PubMed (Bio.Entrez) and Tavily are mocked — no network,
no API keys, no LLM tokens. They exercise:

  * source classification (domain + publication-type tiering),
  * PubMed parsing + extractive summarization,
  * web-search medical-domain ranking,
  * the evidence-retriever orchestration: query-gen fallback, per-source
    resilience, source -> Evidence mapping, ranking, de-dup, and top-N capping.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from schemas.models import AtomicFact, Evidence, EvidenceStance, FactEvidence, FactType


# --------------------------------------------------------------------------- #
# Source classifier
# --------------------------------------------------------------------------- #
class TestSourceClassifier:
    def test_who_and_cdc_are_tier_1(self):
        from core.source_classifier import classify_source

        assert classify_source("https://www.who.int/news/x").tier == 1
        assert classify_source("https://www.cdc.gov/flu/index.html").tier == 1

    def test_cochrane_is_tier_1(self):
        from core.source_classifier import classify_source

        assert classify_source("https://www.cochranelibrary.com/cdsr/123").tier == 1

    def test_pubmed_meta_analysis_promoted_to_tier_1(self):
        from core.source_classifier import classify_source

        result = classify_source(
            "https://pubmed.ncbi.nlm.nih.gov/123/",
            source_name="J Nutr",
            publication_types=["Meta-Analysis", "Journal Article"],
        )
        assert result.tier == 1
        assert "meta-analysis" in result.justification.lower()

    def test_pubmed_systematic_review_via_title_fallback(self):
        from core.source_classifier import classify_source

        # No publication_types given, but the name reveals it's a systematic review.
        result = classify_source(
            "https://pubmed.ncbi.nlm.nih.gov/999/",
            source_name="A systematic review of cumin and weight",
        )
        assert result.tier == 1

    def test_pubmed_plain_study_is_tier_2(self):
        from core.source_classifier import classify_source

        result = classify_source(
            "https://pubmed.ncbi.nlm.nih.gov/456/",
            publication_types=["Journal Article", "Randomized Controlled Trial"],
        )
        assert result.tier == 2

    def test_clinic_subdomain_matches_tier_2(self):
        from core.source_classifier import classify_source

        assert classify_source("https://my.clevelandclinic.org/health/x").tier == 2
        assert classify_source("https://www.mayoclinic.org/diseases").tier == 2

    def test_journalism_and_edu_are_tier_3(self):
        from core.source_classifier import classify_source

        assert classify_source("https://www.statnews.com/2024/x").tier == 3
        assert classify_source("https://med.stanford.edu/health").tier == 3

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.frontiersin.org/journals/nutrition/articles/10.3389/x",
            "https://www.sciencedirect.com/science/article/pii/x",
            "https://link.springer.com/article/10.1007/x",          # springer subdomain
            "https://onlinelibrary.wiley.com/doi/10.1002/x",        # wiley subdomain
            "https://www.bmj.com/content/x",
            "https://jamanetwork.com/journals/jama/x",
            "https://www.thelancet.com/journals/lancet/x",
            "https://www.nature.com/articles/x",
            "https://journals.plos.org/plosone/article?id=x",       # plos subdomain
            "https://academic.oup.com/ajcn/article/x",              # oxford subdomain
            "https://www.tandfonline.com/doi/full/x",
            "https://journals.sagepub.com/doi/x",                   # sage subdomain
            "https://peerj.com/articles/x",
            "https://www.mdpi.com/2072-6643/x",
        ],
    )
    def test_academic_publishers_are_tier_2(self, url):
        from core.source_classifier import classify_source

        # A regular (non-synthesis) article on a recognized publisher -> Tier 2.
        result = classify_source(url, title="A clinical study of cumin and weight")
        assert result.tier == 2

    def test_meta_analysis_title_promoted_regardless_of_domain(self):
        from core.source_classifier import classify_source

        # A Frontiers meta-analysis (the exact case that wrongly landed in T4 live).
        result = classify_source(
            "https://www.frontiersin.org/journals/nutrition/articles/10.3389/x",
            title="Effects of cumin: a GRADE-assessed systematic review and meta-analysis",
        )
        assert result.tier == 1

    def test_meta_analysis_pubtype_promoted_on_publisher(self):
        from core.source_classifier import classify_source

        result = classify_source(
            "https://www.mdpi.com/2072-6643/x", publication_types=["Meta-Analysis"]
        )
        assert result.tier == 1

    def test_pmc_is_tier_2(self):
        from core.source_classifier import classify_source

        assert classify_source("https://pmc.ncbi.nlm.nih.gov/articles/PMC5065707/").tier == 2

    def test_pmc_meta_analysis_promoted(self):
        from core.source_classifier import classify_source

        result = classify_source(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
            title="Cumin and metabolic syndrome: a meta-analysis",
        )
        assert result.tier == 1

    def test_unknown_domain_is_tier_4(self):
        from core.source_classifier import classify_source

        result = classify_source("https://wellnessblog.example.com/post")
        assert result.tier == 4
        assert result.justification  # always explains itself

    def test_bare_host_without_scheme_classifies(self):
        from core.source_classifier import classify_source

        assert classify_source("who.int").tier == 1


# --------------------------------------------------------------------------- #
# PubMed: summarization + parsing (Entrez mocked)
# --------------------------------------------------------------------------- #
class _FakeHandle:
    def __init__(self, data):
        self.data = data

    def close(self):
        pass


class _FakeEntrez:
    """Stand-in for Bio.Entrez: read() just returns the handle's payload."""

    def __init__(self, search_ids, fetch_records):
        self._search = {"IdList": list(search_ids)}
        self._fetch = fetch_records
        self.email = None
        self.api_key = None

    def esearch(self, **_kwargs):
        return _FakeHandle(self._search)

    def efetch(self, **_kwargs):
        return _FakeHandle(self._fetch)

    def read(self, handle):
        return handle.data


def _make_entry(pmid, title, abstract, pub_types=None, with_date=True):
    journal = {"Title": "Journal of Nutrition"}
    if with_date:
        journal["JournalIssue"] = {"PubDate": {"Year": "2021", "Month": "Jun", "Day": "15"}}
    return {
        "MedlineCitation": {
            "PMID": pmid,
            "Article": {
                "ArticleTitle": title,
                "Abstract": {"AbstractText": [abstract] if abstract else []},
                "AuthorList": [{"LastName": "Smith", "Initials": "J"}],
                "Journal": journal,
                "PublicationTypeList": pub_types or ["Journal Article"],
            },
        }
    }


class TestPubMedSummary:
    def test_summary_takes_first_sentences(self):
        from sources.pubmed import summarize_abstract

        abstract = "First sentence. Second sentence. Third sentence. Fourth sentence."
        summary = summarize_abstract(abstract, max_sentences=2)
        assert summary == "First sentence. Second sentence."

    def test_summary_truncates_long_text(self):
        from sources.pubmed import summarize_abstract

        long_abstract = "word " * 400  # one giant "sentence"
        summary = summarize_abstract(long_abstract, max_chars=100)
        assert len(summary) <= 101  # 100 chars + ellipsis
        assert summary.endswith("…")

    def test_empty_abstract_yields_empty_summary(self):
        from sources.pubmed import summarize_abstract

        assert summarize_abstract("") == ""


class TestPubMedUrlHelpers:
    def test_pmid_from_url(self):
        from sources.pubmed import pmid_from_url

        assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/25456022") == "25456022"
        assert pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/25456022/") == "25456022"
        assert pmid_from_url("https://www.cdc.gov/flu") is None
        assert pmid_from_url("") is None

    def test_pmcid_from_url(self):
        from sources.pubmed import pmcid_from_url

        assert (
            pmcid_from_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC5065707/")
            == "PMC5065707"
        )
        assert (
            pmcid_from_url("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5065707/")
            == "PMC5065707"
        )
        # A plain PubMed (non-PMC) link is not a PMCID.
        assert pmcid_from_url("https://pubmed.ncbi.nlm.nih.gov/123") is None


class TestPubMedSearch:
    def test_empty_query_raises(self):
        from sources.pubmed import search_pubmed

        with pytest.raises(ValueError):
            search_pubmed("   ")

    def test_no_results_returns_empty(self, monkeypatch):
        from sources import pubmed

        monkeypatch.setattr(pubmed.time, "sleep", lambda *_: None)
        monkeypatch.setattr(pubmed, "_entrez", lambda: _FakeEntrez([], {}))
        assert pubmed.search_pubmed("nothing matches") == []

    def test_parses_articles_and_builds_summary_and_url(self, monkeypatch):
        from sources import pubmed

        records = {
            "PubmedArticle": [
                _make_entry(
                    "12345",
                    "Cumin supplementation and weight loss",
                    "Background sentence one. Result sentence two. Detail sentence three. Extra four.",
                    pub_types=["Meta-Analysis"],
                )
            ]
        }
        monkeypatch.setattr(pubmed.time, "sleep", lambda *_: None)
        monkeypatch.setattr(pubmed, "_entrez", lambda: _FakeEntrez(["12345"], records))

        articles = pubmed.search_pubmed("cumin weight loss", max_results=5)
        assert len(articles) == 1
        art = articles[0]
        assert art.pmid == "12345"
        assert art.url == "https://pubmed.ncbi.nlm.nih.gov/12345/"
        assert art.journal == "Journal of Nutrition"
        assert art.publication_types == ["Meta-Analysis"]
        assert art.publication_date == datetime(2021, 6, 15)
        # Extractive summary = first 3 sentences, full abstract preserved.
        assert art.summary.startswith("Background sentence one.")
        assert "Extra four" in art.abstract
        assert "Extra four" not in art.summary

    def test_abstractless_articles_are_skipped(self, monkeypatch):
        from sources import pubmed

        records = {
            "PubmedArticle": [
                _make_entry("1", "Has abstract", "Some finding here."),
                _make_entry("2", "No abstract", ""),  # dropped
            ]
        }
        monkeypatch.setattr(pubmed.time, "sleep", lambda *_: None)
        monkeypatch.setattr(pubmed, "_entrez", lambda: _FakeEntrez(["1", "2"], records))

        articles = pubmed.search_pubmed("x")
        assert [a.pmid for a in articles] == ["1"]

    def test_esearch_failure_raises_pubmed_error(self, monkeypatch):
        from sources import pubmed

        class _Boom(_FakeEntrez):
            def esearch(self, **_kwargs):
                raise RuntimeError("HTTP 429 rate limited")

        monkeypatch.setattr(pubmed.time, "sleep", lambda *_: None)
        monkeypatch.setattr(pubmed, "_entrez", lambda: _Boom([], {}))
        with pytest.raises(pubmed.PubMedError):
            pubmed.search_pubmed("x")


# --------------------------------------------------------------------------- #
# Web search (Tavily mocked)
# --------------------------------------------------------------------------- #
class _FakeTavily:
    def __init__(self, results):
        self._results = results

    def search(self, **_kwargs):
        return {"results": self._results}


class TestWebSearch:
    def test_empty_query_raises(self):
        from sources.web_search import search_web

        with pytest.raises(ValueError):
            search_web("")

    def test_missing_key_raises(self, monkeypatch):
        from sources import web_search

        monkeypatch.setattr(web_search.settings, "TAVILY_API_KEY", None)
        with pytest.raises(web_search.WebSearchError):
            web_search.search_web("vitamin c colds")

    def test_medical_domains_ranked_first(self, monkeypatch):
        from sources import web_search

        results = [
            {"title": "Blog", "url": "https://randomblog.example.com/x",
             "content": "c", "score": 0.95},
            {"title": "WHO", "url": "https://www.who.int/vitamin-c",
             "content": "c", "score": 0.40},
            {"title": "Mayo", "url": "https://www.mayoclinic.org/vit-c",
             "content": "c", "score": 0.30},
        ]
        monkeypatch.setattr(web_search.settings, "TAVILY_API_KEY", "tvly-test")
        monkeypatch.setattr(web_search, "_client", lambda: _FakeTavily(results))

        hits = web_search.search_web("vitamin c", max_results=3)
        # Medical domains float to the top despite the blog's higher score.
        assert hits[0].domain == "who.int"
        assert hits[1].domain == "mayoclinic.org"
        assert hits[2].domain == "randomblog.example.com"
        assert hits[0].is_medical_domain is True

    def test_respects_max_results(self, monkeypatch):
        from sources import web_search

        results = [
            {"title": str(i), "url": f"https://who.int/{i}", "content": "c", "score": 0.5}
            for i in range(10)
        ]
        monkeypatch.setattr(web_search.settings, "TAVILY_API_KEY", "tvly-test")
        monkeypatch.setattr(web_search, "_client", lambda: _FakeTavily(results))
        assert len(web_search.search_web("x", max_results=3)) == 3


# --------------------------------------------------------------------------- #
# Evidence retriever — helpers
# --------------------------------------------------------------------------- #
class TestRetrieverHelpers:
    def test_lexical_relevance(self):
        from core.evidence_retriever import _lexical_relevance

        # 2 of 2 content words ("cumin", "weight") present -> 1.0.
        assert _lexical_relevance("cumin weight", "cumin reduces body weight") == 1.0
        # None present -> 0.0.
        assert _lexical_relevance("turmeric cancer", "vitamin d and bones") == 0.0
        # Stopwords don't count toward the query terms.
        assert _lexical_relevance("the and of", "anything") == 0.0

    def test_rank_score_blends_tier(self):
        from core.evidence_retriever import _rank_score

        tier1 = Evidence(content="c", source_url="u1", source_name="n",
                         source_tier=1, relevance_score=0.5)
        tier4 = Evidence(content="c", source_url="u2", source_name="n",
                         source_tier=4, relevance_score=0.5)
        # Equal relevance -> the Tier-1 source ranks higher.
        assert _rank_score(tier1) > _rank_score(tier4)

    def test_dedupe_by_url(self):
        from core.evidence_retriever import _dedupe_by_url

        a = Evidence(content="c", source_url="https://x.com/a", source_name="n",
                     source_tier=2, relevance_score=0.5)
        b = Evidence(content="c", source_url="https://X.com/a", source_name="n",
                     source_tier=2, relevance_score=0.9)  # same URL, different case
        c = Evidence(content="c", source_url="https://y.com/b", source_name="n",
                     source_tier=2, relevance_score=0.5)
        deduped = _dedupe_by_url([a, b, c])
        assert [e.source_url for e in deduped] == ["https://x.com/a", "https://y.com/b"]


# --------------------------------------------------------------------------- #
# Evidence retriever — query generation
# --------------------------------------------------------------------------- #
class TestQueryGeneration:
    def _fact(self):
        return AtomicFact(
            text="cumin water causes weight loss",
            fact_type=FactType.CAUSAL,
            original_context="cumin water melts belly fat",
        )

    def test_uses_llm_queries_when_available(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(
            er.providers, "llm_call",
            lambda **_kw: {"queries": ["cumin supplementation weight loss clinical trial"]},
        )
        queries = er._generate_search_queries(self._fact())
        assert queries == ["cumin supplementation weight loss clinical trial"]

    def test_falls_back_to_fact_text_on_budget(self, monkeypatch):
        from core import evidence_retriever as er

        def _boom(**_kw):
            raise er.providers.BudgetWarning("over budget")

        monkeypatch.setattr(er.providers, "llm_call", _boom)
        queries = er._generate_search_queries(self._fact())
        assert queries == ["cumin water causes weight loss"]

    def test_falls_back_on_empty_llm_output(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er.providers, "llm_call", lambda **_kw: {"queries": []})
        queries = er._generate_search_queries(self._fact())
        assert queries == ["cumin water causes weight loss"]


# --------------------------------------------------------------------------- #
# Evidence retriever — full orchestration (all I/O mocked)
# --------------------------------------------------------------------------- #
class TestRetrieveEvidence:
    def _fact(self):
        return AtomicFact(
            text="vitamin c prevents colds",
            fact_type=FactType.CAUSAL,
            original_context="megadose vitamin c stops colds",
        )

    def _fake_article(self, pmid, title, pub_types=None):
        from sources.pubmed import PubMedArticle

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=f"{title}. Full abstract body.",
            summary=f"{title}. Summary.",
            journal="J Immunol",
            publication_date=datetime(2020, 1, 1),
            publication_types=pub_types or ["Journal Article"],
        )

    def _fake_web(self, url, score):
        from sources.web_search import WebResult
        from sources.web_search import _domain_of

        return WebResult(
            title="t", content="vitamin c and the common cold",
            url=url, domain=_domain_of(url), score=score,
        )

    def _patch_sources(self, monkeypatch, articles, web_hits):
        from core import evidence_retriever as er

        # Skip the LLM entirely: queries are fixed.
        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["vitamin c colds"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: list(articles))
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web_hits))
        return er

    def test_builds_fact_evidence_with_mapped_fields(self, monkeypatch):
        articles = [self._fake_article("1", "vitamin c prevents colds", ["Meta-Analysis"])]
        web = [self._fake_web("https://www.cdc.gov/colds", 0.6)]
        er = self._patch_sources(monkeypatch, articles, web)

        out = er.retrieve_evidence([self._fact()])
        assert len(out) == 1
        fe = out[0]
        assert isinstance(fe, FactEvidence)
        assert fe.atomic_fact.text == "vitamin c prevents colds"

        # PubMed meta-analysis -> Tier 1; CDC -> Tier 1. Both present.
        tiers = {e.source_tier for e in fe.evidence}
        assert tiers == {1}
        pubmed_ev = next(e for e in fe.evidence if "pubmed" in e.source_url)
        assert pubmed_ev.summary  # compressed summary carried through
        assert pubmed_ev.content  # full abstract preserved
        assert pubmed_ev.evidence_stance is EvidenceStance.NEUTRAL  # retrieval stays neutral

    def test_caps_results_per_fact(self, monkeypatch):
        articles = [self._fake_article(str(i), f"vitamin c study {i}") for i in range(8)]
        web = [self._fake_web(f"https://who.int/{i}", 0.5) for i in range(5)]
        er = self._patch_sources(monkeypatch, articles, web)

        out = er.retrieve_evidence([self._fact()], per_fact_limit=3)
        assert len(out[0].evidence) == 3

    def test_ranks_relevant_tier1_above_irrelevant_tier4(self, monkeypatch):
        # Tier-1 PubMed hit whose title matches the fact (high relevance) vs a
        # Tier-4 blog with an unrelated snippet.
        articles = [self._fake_article("1", "vitamin c prevents colds", ["Meta-Analysis"])]
        web = [self._fake_web("https://randomblog.example.com/x", 0.1)]
        er = self._patch_sources(monkeypatch, articles, web)

        out = er.retrieve_evidence([self._fact()])
        assert out[0].evidence[0].source_tier == 1

    def test_dedupes_same_url_across_sources(self, monkeypatch):
        # PubMed article and a web hit pointing at the same PubMed URL.
        shared_url = "https://pubmed.ncbi.nlm.nih.gov/1/"
        articles = [self._fake_article("1", "vitamin c prevents colds")]
        web = [self._fake_web(shared_url, 0.9)]
        er = self._patch_sources(monkeypatch, articles, web)

        out = er.retrieve_evidence([self._fact()])
        urls = [e.source_url for e in out[0].evidence]
        assert urls.count(shared_url) == 1

    def test_resilient_when_pubmed_fails(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])

        def _boom(*_a, **_kw):
            raise er.pubmed.PubMedError("network down")

        monkeypatch.setattr(er.pubmed, "search_pubmed", _boom)
        monkeypatch.setattr(
            er.web_search, "search_web",
            lambda q, max_results=5: [self._fake_web("https://who.int/x", 0.5)],
        )
        out = er.retrieve_evidence([self._fact()])
        # Web evidence still came through; no exception bubbled up.
        assert len(out[0].evidence) == 1
        assert out[0].evidence[0].source_url == "https://who.int/x"

    def test_resilient_when_web_fails(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(
            er.pubmed, "search_pubmed",
            lambda q, max_results=8: [self._fake_article("1", "vitamin c prevents colds")],
        )

        def _boom(*_a, **_kw):
            raise er.web_search.WebSearchError("no key")

        monkeypatch.setattr(er.web_search, "search_web", _boom)
        out = er.retrieve_evidence([self._fact()])
        assert len(out[0].evidence) == 1
        assert "pubmed" in out[0].evidence[0].source_url

    def test_empty_facts_returns_empty(self, monkeypatch):
        from core import evidence_retriever as er

        assert er.retrieve_evidence([]) == []

    # --- PubMed reconciliation of web results ------------------------------- #
    def test_web_pubmed_link_dropped_when_duplicate(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        api_article = self._fake_article("25456022", "vitamin c prevents colds")
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [api_article])
        # Web hit points at the same PMID via a scraped PubMed page.
        dup = self._fake_web("https://pubmed.ncbi.nlm.nih.gov/25456022/", 0.9)
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [dup])
        # Re-fetching a PMID we already have would be a bug — fail loudly if called.
        monkeypatch.setattr(
            er.pubmed, "fetch_article_by_pmid",
            lambda pmid: pytest.fail("known PMID should not be re-fetched"),
        )

        out = er.retrieve_evidence([self._fact()])
        urls = [e.source_url for e in out[0].evidence]
        assert urls == ["https://pubmed.ncbi.nlm.nih.gov/25456022/"]  # only the API copy

    def test_web_pubmed_link_refetched_clean_when_new(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        # PubMed API search surfaced nothing for this fact...
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        # ...but the web search found a scraped PubMed page with junk content.
        junk = self._fake_web("https://pubmed.ncbi.nlm.nih.gov/25456022", 0.9)
        junk.content = "Skip to main page content. An official website of the US government."
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [junk])
        clean = self._fake_article("25456022", "vitamin c prevents colds", ["Meta-Analysis"])
        monkeypatch.setattr(er.pubmed, "fetch_article_by_pmid", lambda pmid: clean)

        out = er.retrieve_evidence([self._fact()])
        ev = out[0].evidence
        assert len(ev) == 1
        assert ev[0].content == clean.abstract            # clean abstract, not scraped junk
        assert "Skip to main page content" not in ev[0].content
        assert ev[0].source_tier == 1                     # meta-analysis -> Tier 1
        assert ev[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/25456022/"

    def test_web_pmc_link_converted_and_refetched(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        pmc_hit = self._fake_web("https://pmc.ncbi.nlm.nih.gov/articles/PMC5065707/", 0.9)
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [pmc_hit])
        monkeypatch.setattr(
            er.pubmed, "convert_pmcid_to_pmid",
            lambda pmcid: "5065707" if pmcid == "PMC5065707" else None,
        )
        clean = self._fake_article("5065707", "vitamin c prevents colds")
        monkeypatch.setattr(er.pubmed, "fetch_article_by_pmid", lambda pmid: clean)

        out = er.retrieve_evidence([self._fact()])
        ev = out[0].evidence
        assert len(ev) == 1
        assert ev[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/5065707/"
        assert ev[0].content == clean.abstract

    def test_web_pubmed_link_falls_back_when_refetch_fails(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        hit = self._fake_web("https://pubmed.ncbi.nlm.nih.gov/999/", 0.5)
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [hit])

        def _boom(pmid):
            raise er.pubmed.PubMedError("network down")

        monkeypatch.setattr(er.pubmed, "fetch_article_by_pmid", _boom)

        out = er.retrieve_evidence([self._fact()])
        ev = out[0].evidence
        # Couldn't clean it up, but we keep the web result rather than losing it.
        assert len(ev) == 1
        assert ev[0].source_url == "https://pubmed.ncbi.nlm.nih.gov/999/"

    # --- Relevance floor ---------------------------------------------------- #
    def test_below_floor_results_are_dropped(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        web = [
            self._fake_web("https://www.cdc.gov/relevant", 0.50),   # keep
            self._fake_web("https://www.cdc.gov/noise", 0.05),      # below floor -> drop
        ]
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web))

        out = er.retrieve_evidence([self._fact()])
        urls = [e.source_url for e in out[0].evidence]
        assert urls == ["https://www.cdc.gov/relevant"]
        assert all(e.relevance_score >= er.RELEVANCE_FLOOR for e in out[0].evidence)

    def test_tier1_does_not_rescue_irrelevant_result(self, monkeypatch):
        from core import evidence_retriever as er

        # A Tier-1 source (who.int) with near-zero relevance must still be dropped;
        # tier boosts relevant results, it doesn't rescue irrelevant ones.
        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        web = [
            self._fake_web("https://www.who.int/unrelated", 0.04),  # T1 but irrelevant
            self._fake_web("https://www.cdc.gov/relevant", 0.40),   # T1 and relevant
        ]
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web))

        out = er.retrieve_evidence([self._fact()])
        urls = [e.source_url for e in out[0].evidence]
        assert urls == ["https://www.cdc.gov/relevant"]

    def test_insufficient_evidence_note_when_fewer_than_two(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        web = [self._fake_web("https://www.cdc.gov/only-one", 0.50)]
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web))

        out = er.retrieve_evidence([self._fact()])
        assert len(out[0].evidence) == 1
        assert out[0].retrieval_note == "insufficient evidence found"

    def test_no_note_when_two_or_more_relevant(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        web = [
            self._fake_web("https://www.cdc.gov/a", 0.50),
            self._fake_web("https://www.mayoclinic.org/b", 0.40),
        ]
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web))

        out = er.retrieve_evidence([self._fact()])
        assert len(out[0].evidence) == 2
        assert out[0].retrieval_note == ""

    def test_all_noise_yields_empty_with_note(self, monkeypatch):
        from core import evidence_retriever as er

        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [])
        web = [self._fake_web("https://www.who.int/x", 0.01)]
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: list(web))

        out = er.retrieve_evidence([self._fact()])
        assert out[0].evidence == []
        assert out[0].retrieval_note == "insufficient evidence found"
