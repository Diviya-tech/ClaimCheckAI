"""Weeks 11-12 — error-handling hardening, cache, timing, reasoning integrity.

All offline. These lock down the promise that the pipeline DEGRADES instead of
failing: a dead evidence source, a missing web-search key, a malformed LLM
response, or the token budget running out mid-dossier each produce a dossier
(or a clean HTTP error) with a plain-language note — never a stack trace.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

import main as pipeline
from api import server
from config import providers, settings
from core import dossier_builder, evidence_retriever as er, verdict_engine
from core.cache import DossierCache
from core.dossier_builder import build_dossier, format_dossier
from core.verdict_engine import evaluate
from schemas.models import (
    AtomicFact,
    ClaimExtractionResult,
    Dossier,
    Evidence,
    EvidenceApplicability,
    EvidenceStance,
    FactEvidence,
    FactType,
    SourceFormat,
    Verdict,
)
from sources import pubmed

client = TestClient(server.app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _fact(text="Vitamin C prevents colds."):
    return AtomicFact(text=text, fact_type=FactType.CAUSAL, original_context=text)


def _extraction(text="Vitamin C prevents colds.", claim_found=True):
    return ClaimExtractionResult(
        original_text=text,
        primary_claim=text if claim_found else "",
        atomic_facts=[_fact(text)] if claim_found else [],
        claim_found=claim_found,
        source_format=SourceFormat.TEXT,
    )


def _evidence(name="Cochrane", tier=1, applicability=EvidenceApplicability.UNKNOWN):
    return Evidence(
        content="c", summary="s", source_url=f"https://example.org/{name}",
        source_name=name, source_tier=tier, relevance_score=0.6,
        applicability=applicability,
    )


def _article(pmid, title, mesh=None, abstract=None, is_animal=None):
    text = abstract or f"{title}. Vitamin C prevents colds in this study."
    return pubmed.PubMedArticle(
        pmid=pmid, title=title, abstract=text, summary=text[:80],
        journal="J", publication_date=datetime(2020, 1, 1),
        publication_types=["Journal Article"], mesh_terms=mesh or [],
        is_animal_study=(
            is_animal if is_animal is not None else pubmed.is_animal_study(mesh or [], text)
        ),
    )


def _web(url, score=0.5):
    from sources.web_search import WebResult, _domain_of

    return WebResult(title="t", content="vitamin c prevents colds", url=url,
                     domain=_domain_of(url), score=score)


def _llm_router(monkeypatch, verdict=None, narrative=None):
    """Stub `providers.llm_call` for BOTH premium stages (they share the module).

    `verdict` / `narrative` may be a dict to return or an exception to raise.
    """
    def _call(**kw):
        is_narrative = "narrative_summary" in kw["response_schema"]["properties"]
        resp = narrative if is_narrative else verdict
        if isinstance(resp, Exception):
            raise resp
        return resp

    monkeypatch.setattr(providers, "llm_call", _call)


_STRONG_DIRECT = {
    "fact_verdicts": [{"fact_index": 0, "verdict": "Strongly Supported", "reasoning": "…",
                       "evidence_stances": [{"evidence_index": 0, "stance": "supporting",
                                             "applicability": "direct"}]}],
    "rhetorical_flags": [],
}
_INSUFFICIENT = {
    "fact_verdicts": [{"fact_index": 0, "verdict": "Insufficient Evidence",
                       "reasoning": "…", "evidence_stances": []}],
    "rhetorical_flags": [],
}


@pytest.fixture
def fixed_queries(monkeypatch):
    monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["vitamin c colds"])


@pytest.fixture
def isolated_budget(tmp_path, monkeypatch):
    """Keep provider tests from touching the real usage ledger."""
    monkeypatch.setattr(settings, "DAILY_USAGE_FILE", str(tmp_path / "usage.json"))
    monkeypatch.setattr(settings, "DAILY_TOKEN_BUDGET", 1_000_000)


@pytest.fixture(autouse=True)
def _fresh_cache():
    server.dossier_cache.clear()
    yield
    server.dossier_cache.clear()


# --------------------------------------------------------------------------- #
# Evidence sources down / unconfigured
# --------------------------------------------------------------------------- #
class TestSourceDegradation:
    def test_pubmed_down_falls_back_to_web_with_note(self, monkeypatch, fixed_queries):
        def _boom(*_a, **_kw):
            raise pubmed.PubMedError("NCBI unreachable")

        monkeypatch.setattr(er.pubmed, "search_pubmed", _boom)
        monkeypatch.setattr(er.web_search, "search_web",
                            lambda q, max_results=5: [_web("https://www.cdc.gov/colds")])
        out = er.retrieve_evidence([_fact()])
        assert [e.source_url for e in out[0].evidence] == ["https://www.cdc.gov/colds"]
        assert out[0].source_notes == [er.NOTE_PUBMED_DOWN]

    def test_tavily_key_missing_skips_web_with_note(self, monkeypatch, fixed_queries):
        monkeypatch.setattr(er.pubmed, "search_pubmed",
                            lambda q, max_results=8: [_article("1", "Vitamin C prevents colds")])

        def _no_key(*_a, **_kw):
            raise er.web_search.WebSearchError("TAVILY_API_KEY is not set. Add it to .env")

        monkeypatch.setattr(er.web_search, "search_web", _no_key)
        out = er.retrieve_evidence([_fact()])
        assert len(out[0].evidence) == 1
        assert out[0].source_notes == [er.NOTE_WEB_SKIPPED]

    def test_tavily_down_is_distinguished_from_unconfigured(self, monkeypatch, fixed_queries):
        monkeypatch.setattr(er.pubmed, "search_pubmed",
                            lambda q, max_results=8: [_article("1", "Vitamin C prevents colds")])

        def _down(*_a, **_kw):
            raise er.web_search.WebSearchError("Tavily search failed: 503")

        monkeypatch.setattr(er.web_search, "search_web", _down)
        out = er.retrieve_evidence([_fact()])
        assert out[0].source_notes == [er.NOTE_WEB_DOWN]

    def test_all_sources_down_yields_single_note_and_no_error(self, monkeypatch, fixed_queries):
        def _boom(*_a, **_kw):
            raise pubmed.PubMedError("down")

        def _boom_web(*_a, **_kw):
            raise er.web_search.WebSearchError("down")

        monkeypatch.setattr(er.pubmed, "search_pubmed", _boom)
        monkeypatch.setattr(er.web_search, "search_web", _boom_web)
        out = er.retrieve_evidence([_fact()])
        assert out[0].evidence == []
        assert out[0].source_notes == [er.NOTE_ALL_SOURCES_DOWN]
        assert out[0].retrieval_note == "insufficient evidence found"

    def test_partial_pubmed_failure_is_not_reported_as_down(self, monkeypatch):
        # Two queries, one fails: PubMed still answered, so no note.
        monkeypatch.setattr(er, "_generate_search_queries", lambda f, **_kw: ["q1", "q2"])
        calls = {"n": 0}

        def _flaky(q, max_results=8):
            calls["n"] += 1
            if q == "q1":
                raise pubmed.PubMedError("timeout")
            return [_article("1", "Vitamin C prevents colds")]

        monkeypatch.setattr(er.pubmed, "search_pubmed", _flaky)
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [])
        out = er.retrieve_evidence([_fact()])
        assert out[0].source_notes == []

    def test_no_notes_when_both_sources_healthy(self, monkeypatch, fixed_queries):
        monkeypatch.setattr(er.pubmed, "search_pubmed",
                            lambda q, max_results=8: [_article("1", "Vitamin C prevents colds")])
        monkeypatch.setattr(er.web_search, "search_web",
                            lambda q, max_results=5: [_web("https://www.cdc.gov/colds")])
        out = er.retrieve_evidence([_fact()])
        assert out[0].source_notes == []

    def test_collect_limitations_dedupes_across_facts(self):
        fes = [
            FactEvidence(atomic_fact=_fact("a"), source_notes=[er.NOTE_PUBMED_DOWN]),
            FactEvidence(atomic_fact=_fact("b"), source_notes=[er.NOTE_PUBMED_DOWN]),
            FactEvidence(atomic_fact=_fact("c"), source_notes=[]),
        ]
        assert er.collect_limitations(fes) == [er.NOTE_PUBMED_DOWN]

    def test_source_notes_reach_the_dossier(self, monkeypatch, fixed_queries):
        def _boom(*_a, **_kw):
            raise pubmed.PubMedError("down")

        monkeypatch.setattr(er.pubmed, "search_pubmed", _boom)
        monkeypatch.setattr(er.web_search, "search_web",
                            lambda q, max_results=5: [_web("https://www.cdc.gov/colds")])
        _llm_router(monkeypatch, verdict=_INSUFFICIENT, narrative={"narrative_summary": "Summary."})
        dossier = pipeline.complete_dossier(_extraction())
        assert dossier.narrative_summary == "Summary."
        assert dossier.limitations == [er.NOTE_PUBMED_DOWN]
        assert "LIMITATIONS OF THIS RUN" in format_dossier(dossier)
        assert "PubMed was unavailable" in format_dossier(dossier)


# --------------------------------------------------------------------------- #
# Malformed LLM output: retry once, then a clear error
# --------------------------------------------------------------------------- #
class TestMalformedOutput:
    _SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    def _provider(self, responses):
        calls = []

        class _Stub:
            def complete(self, **kwargs):
                calls.append(kwargs)
                r = responses.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r, 1, 1

        return _Stub(), calls

    def test_retries_once_then_returns_good_result(self, monkeypatch, isolated_budget):
        stub, calls = self._provider(["prose instead of a tool call", {"x": "ok"}])
        monkeypatch.setattr(providers, "_get_provider", lambda _n: stub)
        out = providers.llm_call("hi", model_tier="lightweight", response_schema=self._SCHEMA)
        assert out == {"x": "ok"}
        assert len(calls) == 2

    def test_two_malformed_responses_raise_user_readable_error(self, monkeypatch, isolated_budget):
        stub, calls = self._provider([{"wrong": 1}, "still prose"])
        monkeypatch.setattr(providers, "_get_provider", lambda _n: stub)
        with pytest.raises(providers.MalformedOutputError) as exc_info:
            providers.llm_call("hi", model_tier="lightweight", response_schema=self._SCHEMA)
        assert len(calls) == 2  # exactly one retry
        msg = str(exc_info.value)
        assert "2 times" in msg and "try again" in msg
        assert "Traceback" not in msg

    def test_provider_level_malformed_is_retried_and_billed(self, monkeypatch, isolated_budget):
        stub, calls = self._provider([
            providers.MalformedOutputError("no tool_use block", input_tokens=7, output_tokens=3),
            {"x": "ok"},
        ])
        monkeypatch.setattr(providers, "_get_provider", lambda _n: stub)
        out = providers.llm_call("hi", model_tier="lightweight", response_schema=self._SCHEMA)
        assert out == {"x": "ok"}
        assert providers.get_daily_usage() == 7 + 3 + 1 + 1  # failed attempt still counted

    def test_free_text_calls_are_not_retried(self, monkeypatch, isolated_budget):
        stub, calls = self._provider(["some text"])
        monkeypatch.setattr(providers, "_get_provider", lambda _n: stub)
        assert providers.llm_call("hi", model_tier="lightweight") == "some text"
        assert len(calls) == 1

    def test_malformed_extraction_is_502_with_message(self, monkeypatch):
        def _bad(**_kw):
            raise providers.MalformedOutputError(
                "The lightweight model returned an unusable response 2 times in a row."
            )

        monkeypatch.setattr(server, "check_claim", _bad)
        resp = client.post("/api/check", json={"text": "Vitamin C prevents colds."})
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "unusable response" in detail and "Traceback" not in detail

    def test_malformed_verdict_call_degrades_to_insufficient(self, monkeypatch):
        def _bad(**_kw):
            raise providers.MalformedOutputError("unusable twice")

        monkeypatch.setattr(verdict_engine.providers, "llm_call", _bad)
        fe = FactEvidence(atomic_fact=_fact(), evidence=[_evidence()])
        ev = evaluate(_extraction(), [fe])
        assert ev.verdicts[0].verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert "could not be completed" in ev.verdicts[0].reasoning

    def test_malformed_narrative_call_falls_back_with_note(self, monkeypatch):
        def _bad(**_kw):
            raise providers.MalformedOutputError("unusable twice")

        monkeypatch.setattr(dossier_builder.providers, "llm_call", _bad)
        v = verdict_engine.fallback_verdicts([FactEvidence(atomic_fact=_fact())])
        d = build_dossier("x", _extraction(), v)
        assert "evidence assessment found" in d.narrative_summary
        assert dossier_builder.NOTE_NARRATIVE_FAILED in d.limitations


# --------------------------------------------------------------------------- #
# Token budget exhausted mid-dossier
# --------------------------------------------------------------------------- #
class TestBudgetMidDossier:
    def _retrieval(self, monkeypatch):
        fe = FactEvidence(atomic_fact=_fact(), evidence=[_evidence("Cochrane", 1)])
        monkeypatch.setattr(pipeline, "retrieve_evidence", lambda facts, **_kw: [fe])
        return fe

    def test_budget_at_verdict_stage_completes_dossier_with_note(self, monkeypatch):
        self._retrieval(monkeypatch)

        budget = providers.BudgetWarning("Daily token budget of 100 reached")
        _llm_router(monkeypatch, verdict=budget, narrative=budget)

        dossier = pipeline.complete_dossier(_extraction())
        assert isinstance(dossier, Dossier)
        assert dossier.verdicts[0].verdict is Verdict.INSUFFICIENT_EVIDENCE
        # The retrieved evidence is still shown, as unassessed context.
        assert [e.source_name for e in dossier.verdicts[0].neutral_evidence] == ["Cochrane"]
        assert pipeline.NOTE_VERDICT_BUDGET in dossier.limitations
        assert dossier_builder.NOTE_NARRATIVE_BUDGET in dossier.limitations
        assert dossier.narrative_summary  # deterministic tally, never empty

    def test_budget_only_at_narrative_keeps_real_verdicts(self, monkeypatch):
        self._retrieval(monkeypatch)
        _llm_router(monkeypatch, verdict=_STRONG_DIRECT, narrative=providers.BudgetWarning("out"))
        dossier = pipeline.complete_dossier(_extraction())
        assert dossier.verdicts[0].verdict is Verdict.STRONGLY_SUPPORTED
        assert dossier.limitations == [dossier_builder.NOTE_NARRATIVE_BUDGET]

    def test_budget_before_extraction_is_still_503(self, monkeypatch):
        # Nothing has been computed yet, so there is nothing to "complete".
        def _budget(**_kw):
            raise providers.BudgetWarning("out")

        monkeypatch.setattr(server, "check_claim", _budget)
        resp = client.post("/api/check", json={"text": "Vitamin C prevents colds."})
        assert resp.status_code == 503

    def test_api_returns_degraded_dossier_not_error(self, monkeypatch):
        self._retrieval(monkeypatch)
        monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction())
        budget = providers.BudgetWarning("out")
        _llm_router(monkeypatch, verdict=budget, narrative=budget)
        resp = client.post("/api/check", json={"text": "Vitamin C prevents colds."})
        assert resp.status_code == 200
        body = resp.json()
        assert pipeline.NOTE_VERDICT_BUDGET in body["limitations"]
        assert body["verdicts"][0]["verdict"] == "Insufficient Evidence"


# --------------------------------------------------------------------------- #
# Every error is a user-readable message
# --------------------------------------------------------------------------- #
class TestUserReadableErrors:
    def test_unhandled_exception_is_json_detail_not_traceback(self, monkeypatch):
        def _bug(req):
            raise KeyError("something the mapping never anticipated")

        monkeypatch.setattr(server, "_safe_extract", _bug)
        resp = client.post("/api/check", json={"text": "Vitamin C prevents colds."})
        assert resp.status_code == 500
        body = resp.json()
        assert set(body) == {"detail"}
        assert "unexpected internal error" in body["detail"]
        assert "Traceback" not in resp.text and "KeyError" not in resp.text

    def test_too_long_text_is_413_with_guidance(self):
        resp = client.post("/api/check", json={"text": "x" * (server.MAX_TEXT_CHARS + 1)})
        assert resp.status_code == 413
        assert "too long" in resp.json()["detail"]

    def test_text_at_limit_is_accepted(self, monkeypatch):
        monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction(claim_found=False))
        resp = client.post("/api/extract", json={"text": "x" * server.MAX_TEXT_CHARS})
        assert resp.status_code == 200

    def test_bad_url_is_400_with_reason(self, monkeypatch):
        from input.url_input import URLExtractionError

        def _bad(**_kw):
            raise URLExtractionError("Could not fetch https://x.test (connection refused).")

        monkeypatch.setattr(server, "check_claim", _bad)
        resp = client.post("/api/check", json={"url": "https://x.test"})
        assert resp.status_code == 400
        assert "Could not fetch" in resp.json()["detail"]

    def test_cli_reports_malformed_output_cleanly(self, monkeypatch, capsys):
        def _bad(**_kw):
            raise providers.MalformedOutputError("unusable response 2 times in a row")

        monkeypatch.setattr(pipeline, "check_claim", _bad)
        rc = pipeline.main(["--text", "Vitamin C prevents colds."])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error: unusable response" in err and "Traceback" not in err

    def test_cli_prints_degraded_dossier_on_budget_mid_run(self, monkeypatch, capsys):
        monkeypatch.setattr(pipeline, "check_claim", lambda **_kw: _extraction())
        monkeypatch.setattr(pipeline, "retrieve_evidence", lambda facts, **_kw: [
            FactEvidence(atomic_fact=_fact(), evidence=[_evidence()])])

        budget = providers.BudgetWarning("out")
        _llm_router(monkeypatch, verdict=budget, narrative=budget)
        rc = pipeline.main(["--text", "Vitamin C prevents colds."])
        assert rc == 0
        out = capsys.readouterr().out
        assert "EVIDENCE DOSSIER" in out and "LIMITATIONS OF THIS RUN" in out


# --------------------------------------------------------------------------- #
# In-memory cache
# --------------------------------------------------------------------------- #
class TestDossierCache:
    def _dossier(self, text="Vitamin C prevents colds."):
        return Dossier(original_input=text, claim_extraction=_extraction(text))

    def test_miss_then_hit_returns_same_dossier(self):
        cache = DossierCache()
        assert cache.get("Vitamin C prevents colds.") is None
        d = self._dossier()
        cache.put("Vitamin C prevents colds.", d)
        assert cache.get("Vitamin C prevents colds.") is d
        assert (cache.hits, cache.misses) == (1, 1)

    def test_key_is_normalized_text(self):
        cache = DossierCache()
        d = self._dossier()
        cache.put("  Vitamin C prevents colds.\r\n", d)
        assert cache.get("Vitamin C prevents colds.") is d
        assert cache.get("Vitamin​ C prevents colds.") is d  # zero-width stripped
        assert "vitamin c prevents colds." not in cache  # but case is significant

    def test_lru_eviction(self):
        cache = DossierCache(max_entries=2)
        cache.put("a", self._dossier("a"))
        cache.put("b", self._dossier("b"))
        cache.get("a")  # 'a' is now most recent
        cache.put("c", self._dossier("c"))
        assert "a" in cache and "c" in cache and "b" not in cache

    def test_empty_text_is_never_cached(self):
        cache = DossierCache()
        cache.put("   ", self._dossier())
        assert len(cache) == 0
        assert cache.get("") is None

    def test_api_serves_cached_dossier_on_repeat(self, monkeypatch):
        calls = {"n": 0}

        def _extract(**_kw):
            calls["n"] += 1
            return _extraction()

        monkeypatch.setattr(server, "check_claim", _extract)
        monkeypatch.setattr(server, "complete_dossier", lambda ex, **_kw: self._dossier())
        first = client.post("/api/check", json={"text": "Vitamin C prevents colds."})
        second = client.post("/api/check", json={"text": "Vitamin C prevents colds. "})
        assert first.status_code == second.status_code == 200
        assert first.json()["id"] == second.json()["id"]  # the very same dossier
        assert calls["n"] == 1  # pipeline ran once

    def test_api_does_not_cache_no_claim_or_non_text(self, monkeypatch):
        calls = {"n": 0}

        def _extract(**_kw):
            calls["n"] += 1
            return _extraction(claim_found=False)

        monkeypatch.setattr(server, "check_claim", _extract)
        client.post("/api/check", json={"text": "I love smoothies"})
        client.post("/api/check", json={"text": "I love smoothies"})
        assert calls["n"] == 2  # a 422 is never cached
        assert len(server.dossier_cache) == 0

    def test_run_pipeline_uses_cache(self, monkeypatch):
        server.dossier_cache.clear()
        calls = {"n": 0}

        def _extract(**_kw):
            calls["n"] += 1
            return _extraction()

        monkeypatch.setattr(pipeline, "check_claim", _extract)
        monkeypatch.setattr(pipeline, "complete_dossier", lambda ex, **_kw: self._dossier())
        d1 = pipeline.run_pipeline(text="Vitamin C prevents colds.")
        d2 = pipeline.run_pipeline(text="Vitamin C prevents colds.")
        assert d1 is d2 and calls["n"] == 1
        pipeline.run_pipeline(text="Vitamin C prevents colds.", use_cache=False)
        assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Stage timing
# --------------------------------------------------------------------------- #
class TestStageTiming:
    def test_each_stage_is_timed_and_logged(self, monkeypatch, caplog):
        monkeypatch.setattr(pipeline, "retrieve_evidence", lambda facts, **_kw: [
            FactEvidence(atomic_fact=_fact(), evidence=[])])
        _llm_router(monkeypatch, verdict=_INSUFFICIENT, narrative={"narrative_summary": "ok"})
        timings: dict[str, float] = {}
        with caplog.at_level(logging.INFO, logger="claimcheck.pipeline"):
            pipeline.complete_dossier(_extraction(), timings=timings)
        assert set(timings) == {"retrieve", "evaluate", "assemble"}
        assert all(t >= 0 for t in timings.values())
        logged = [r.getMessage() for r in caplog.records if "[timing]" in r.getMessage()]
        assert any("retrieve" in m for m in logged)
        assert any("assemble" in m for m in logged)

    def test_timer_records_even_when_stage_raises(self, caplog):
        timings: dict[str, float] = {}
        with caplog.at_level(logging.INFO, logger="claimcheck.pipeline"):
            with pytest.raises(RuntimeError):
                with pipeline.stage_timer("boom", timings):
                    raise RuntimeError("x")
        assert "boom" in timings


# --------------------------------------------------------------------------- #
# Reasoning integrity: animal-study filter
# --------------------------------------------------------------------------- #
class TestAnimalStudyFilter:
    @pytest.mark.parametrize(
        "mesh, text, expected",
        [
            (["Animals", "Rats", "Curcumin"], "", True),
            (["Humans", "Animals"], "", False),          # human study with animal arm
            (["Humans", "Female", "Adult"], "", False),
            ([], "Curcumin reduced tumour growth in mice on a high-fat diet.", True),
            ([], "Curcumin suppressed proliferation in the HepG2 cell line.", True),
            ([], "A randomized, placebo-controlled trial in 90 adults.", False),
            ([], "Effects were tested in vitro and then in 40 patients.", False),
            ([], "", False),
        ],
    )
    def test_is_animal_study(self, mesh, text, expected):
        assert pubmed.is_animal_study(mesh, text) is expected

    def test_mesh_terms_are_parsed_from_records(self):
        entry = {
            "MedlineCitation": {
                "PMID": "9",
                "MeshHeadingList": [
                    {"DescriptorName": "Animals"}, {"DescriptorName": "Rats, Wistar"},
                ],
                "Article": {
                    "ArticleTitle": "Cumin in rats",
                    "Abstract": {"AbstractText": ["Cumin lowered weight in Wistar rats."]},
                    "AuthorList": [],
                    "Journal": {"Title": "J"},
                    "PublicationTypeList": ["Journal Article"],
                },
            }
        }
        article = pubmed._parse_one(entry)
        assert article.mesh_terms == ["Animals", "Rats, Wistar"]
        assert article.is_animal_study is True

    def test_retrieval_stamps_non_human_and_ranks_it_below_humans(self, monkeypatch, fixed_queries):
        rat = _article("1", "Vitamin C prevents colds in rats", mesh=["Animals", "Rats"])
        human = _article("2", "Vitamin C prevents colds in adults", mesh=["Humans"])
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: [rat, human])
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [])
        out = er.retrieve_evidence([_fact()])
        names = [e.source_url for e in out[0].evidence]
        assert names.index(human.url) < names.index(rat.url)
        by_url = {e.source_url: e for e in out[0].evidence}
        assert by_url[rat.url].applicability is EvidenceApplicability.NON_HUMAN
        assert by_url[human.url].applicability is EvidenceApplicability.UNKNOWN

    def test_at_most_one_animal_study_per_fact(self, monkeypatch, fixed_queries):
        rats = [_article(str(i), f"Vitamin C prevents colds in rats {i}", mesh=["Animals"])
                for i in range(4)]
        monkeypatch.setattr(er.pubmed, "search_pubmed", lambda q, max_results=8: rats)
        monkeypatch.setattr(er.web_search, "search_web", lambda q, max_results=5: [])
        out = er.retrieve_evidence([_fact()])
        assert len(out[0].evidence) == 1
        # Animal-only evidence doesn't count as "enough to evaluate".
        assert out[0].retrieval_note == "insufficient evidence found"

    def test_verdict_engine_forces_animal_evidence_neutral(self, monkeypatch):
        rat = _evidence("Rat study", 2, EvidenceApplicability.NON_HUMAN)
        fe = FactEvidence(atomic_fact=_fact(), evidence=[rat])
        # Even if the model calls it "supporting"/"direct", code overrides.
        monkeypatch.setattr(verdict_engine.providers, "llm_call", lambda **_kw: {
            "fact_verdicts": [{"fact_index": 0, "verdict": "Strongly Supported", "reasoning": "…",
                               "evidence_stances": [{"evidence_index": 0, "stance": "supporting",
                                                     "applicability": "direct"}]}],
            "rhetorical_flags": [],
        })
        v = evaluate(_extraction(), [fe]).verdicts[0]
        assert v.supporting_evidence == []
        assert v.neutral_evidence[0].evidence_stance is EvidenceStance.NEUTRAL
        assert v.neutral_evidence[0].applicability is EvidenceApplicability.NON_HUMAN
        assert v.verdict is Verdict.PARTIALLY_SUPPORTED  # capped: no direct support left

    def test_prompt_marks_non_human_evidence(self):
        rat = _evidence("Rat study", 2, EvidenceApplicability.NON_HUMAN)
        assert "[NON-HUMAN STUDY]" in verdict_engine._evidence_line(rat)
        assert "[NON-HUMAN STUDY]" not in verdict_engine._evidence_line(_evidence())


# --------------------------------------------------------------------------- #
# Reasoning integrity: applicability dimension
# --------------------------------------------------------------------------- #
class TestEvidenceApplicability:
    def test_schema_requires_applicability_per_evidence_item(self):
        item = verdict_engine._EVAL_SCHEMA["properties"]["fact_verdicts"]["items"]
        stance_item = item["properties"]["evidence_stances"]["items"]
        assert "applicability" in stance_item["required"]
        assert stance_item["properties"]["applicability"]["enum"] == ["direct", "indirect"]

    def test_missing_applicability_defaults_to_indirect(self):
        assert verdict_engine._resolve_applicability(_evidence(), None) is EvidenceApplicability.INDIRECT
        assert verdict_engine._resolve_applicability(_evidence(), "nonsense") is EvidenceApplicability.INDIRECT
        assert verdict_engine._resolve_applicability(_evidence(), "direct") is EvidenceApplicability.DIRECT

    def test_retrieval_non_human_stamp_overrides_model(self):
        rat = _evidence("r", 2, EvidenceApplicability.NON_HUMAN)
        assert verdict_engine._resolve_applicability(rat, "direct") is EvidenceApplicability.NON_HUMAN

    def test_direct_evidence_keeps_strong_verdict(self):
        direct = _evidence("a", 1, EvidenceApplicability.DIRECT)
        v, r = verdict_engine._apply_applicability_cap(Verdict.STRONGLY_SUPPORTED, "why", [direct], [])
        assert v is Verdict.STRONGLY_SUPPORTED and r == "why"

    def test_non_strong_verdicts_are_untouched(self):
        for verdict in (Verdict.PARTIALLY_SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE,
                        Verdict.CONFLICTING_EVIDENCE, Verdict.TOO_VAGUE):
            v, _ = verdict_engine._apply_applicability_cap(verdict, "why", [], [])
            assert v is verdict

    def test_applicability_serializes_in_dossier_json(self):
        d = Dossier(original_input="x", claim_extraction=_extraction())
        ev = _evidence("a", 1, EvidenceApplicability.INDIRECT)
        assert ev.model_dump()["applicability"] == "indirect"
        assert "limitations" in d.model_dump()


# --------------------------------------------------------------------------- #
# Reasoning integrity: evidence-bound rhetoric
# --------------------------------------------------------------------------- #
class TestEvidenceBoundRhetoric:
    _TEXT = "Doctors don't want you to know: cumin water MELTS belly fat in two weeks!"

    def _flags(self, raw):
        return verdict_engine._assemble_flags(raw, original_text=self._TEXT)

    def test_flag_with_verbatim_excerpt_is_kept(self):
        flags = self._flags([{"pattern": "Conspiracy framing", "explanation": "…",
                              "excerpt": "Doctors don't want you to know"}])
        assert [f.pattern for f in flags] == ["Conspiracy framing"]

    def test_match_is_case_quote_and_whitespace_insensitive(self):
        flags = self._flags([{"pattern": "Guaranteed outcome", "explanation": "…",
                              "excerpt": "melts   belly fat"},
                             {"pattern": "Conspiracy framing", "explanation": "…",
                              "excerpt": "doctors don’t want you to know"}])
        assert len(flags) == 2

    def test_paraphrased_or_invented_excerpt_is_dropped(self):
        flags = self._flags([{"pattern": "False urgency", "explanation": "…",
                              "excerpt": "act now before it's banned"},
                             {"pattern": "Emotional manipulation", "explanation": "…",
                              "excerpt": "the claim uses fear"}])
        assert flags == []

    def test_flag_without_excerpt_is_dropped(self):
        flags = self._flags([{"pattern": "Appeal to nature", "explanation": "…", "excerpt": ""}])
        assert flags == []

    def test_sober_claim_yields_no_flags_even_if_model_overreaches(self, monkeypatch):
        text = "Vaccines prevent measles."
        fe = FactEvidence(atomic_fact=_fact(text), evidence=[])
        monkeypatch.setattr(verdict_engine.providers, "llm_call", lambda **_kw: {
            "fact_verdicts": [{"fact_index": 0, "verdict": "Strongly Supported",
                               "reasoning": "…", "evidence_stances": []}],
            "rhetorical_flags": [{"pattern": "Guaranteed outcome", "explanation": "…",
                                  "excerpt": "guaranteed to work"}],
        })
        ev = evaluate(_extraction(text), [fe])
        assert ev.rhetorical_flags == []

    def test_prompt_demands_verbatim_excerpts(self):
        prompt = verdict_engine._SYSTEM_PROMPT.lower()
        assert "verbatim" in prompt
        assert "excerpt does not appear in the text is discarded" in prompt
