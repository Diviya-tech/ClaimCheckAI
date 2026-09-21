"""Weeks 7-8 test suite — verdict engine + dossier builder.

Every test runs offline. The premium LLM call is stubbed via monkeypatch, so no
API key and no network are needed. These lock down the parts that are ours (not
the model's): stance bucketing, verdict/flag mapping, defensive fallbacks, and
dossier assembly + rendering.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core import dossier_builder, verdict_engine
from core.dossier_builder import build_dossier, format_dossier
from core.verdict_engine import evaluate, evaluate_claim
from schemas.models import (
    AtomicFact,
    AtomicVerdict,
    ClaimExtractionResult,
    Dossier,
    Evidence,
    EvidenceApplicability,
    EvidenceStance,
    FactEvidence,
    FactType,
    RhetoricalFlag,
    SourceFormat,
    Verdict,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _fact(text="Cumin water causes belly-fat loss.", ftype=FactType.CAUSAL):
    return AtomicFact(text=text, fact_type=ftype, original_context="melts belly fat")


def _evidence(name="Cochrane", tier=1, rel=0.6, stance=EvidenceStance.NEUTRAL):
    return Evidence(
        content="A systematic review of cumin supplementation.",
        summary="No significant effect on body weight was found.",
        source_url=f"https://example.org/{name.lower()}",
        source_name=name,
        source_tier=tier,
        relevance_score=rel,
        evidence_stance=stance,
        publication_date=datetime(2019, 1, 1, tzinfo=timezone.utc),
    )


def _extraction(claim_found=True, facts=None):
    facts = [_fact()] if facts is None and claim_found else (facts or [])
    return ClaimExtractionResult(
        original_text="Cumin water melts belly fat in two weeks.",
        primary_claim="Cumin water melts belly fat in two weeks." if claim_found else "",
        atomic_facts=facts,
        claim_found=claim_found,
        source_format=SourceFormat.TEXT,
    )


def _stub_llm(response):
    """Return a callable that ignores its kwargs and yields `response`."""
    def _call(**_kwargs):
        return response
    return _call


# --------------------------------------------------------------------------- #
# Verdict engine — stance bucketing & mapping
# --------------------------------------------------------------------------- #
class TestVerdictEngine:
    def test_maps_verdict_stances_and_reasoning(self, monkeypatch):
        fact = _fact()
        fe = FactEvidence(
            atomic_fact=fact,
            evidence=[_evidence("Cochrane", 1), _evidence("RandomBlog", 4, rel=0.2)],
        )
        response = {
            "fact_verdicts": [
                {
                    "fact_index": 0,
                    "verdict": "Strongly Refuted",
                    "reasoning": "A 2019 Cochrane review (T1) found no effect.",
                    "evidence_stances": [
                        {"evidence_index": 0, "stance": "opposing", "applicability": "direct"},
                        {"evidence_index": 1, "stance": "neutral", "applicability": "indirect"},
                    ],
                }
            ],
            "rhetorical_flags": [],
        }
        monkeypatch.setattr(verdict_engine.providers, "llm_call", _stub_llm(response))

        verdicts = evaluate_claim(_extraction(facts=[fact]), [fe])
        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.verdict is Verdict.STRONGLY_REFUTED
        assert "Cochrane" in v.reasoning
        # Opposing evidence bucketed, and its stance stamped on the copy.
        assert [e.source_name for e in v.opposing_evidence] == ["Cochrane"]
        assert v.opposing_evidence[0].evidence_stance is EvidenceStance.OPPOSING
        assert v.opposing_evidence[0].applicability is EvidenceApplicability.DIRECT
        assert [e.source_name for e in v.neutral_evidence] == ["RandomBlog"]
        assert v.supporting_evidence == []

    def test_extracts_rhetorical_flags(self, monkeypatch):
        fact = _fact()
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence()])
        response = {
            "fact_verdicts": [
                {
                    "fact_index": 0,
                    "verdict": "Insufficient Evidence",
                    "reasoning": "Thin evidence.",
                    "evidence_stances": [],
                }
            ],
            "rhetorical_flags": [
                {
                    "pattern": "Guaranteed outcome",
                    "explanation": "Absolute language ('melts') implies a guaranteed result.",
                    "excerpt": "melts belly fat",
                },
                {"pattern": "", "explanation": "dropped", "excerpt": ""},  # blank -> filtered
            ],
        }
        monkeypatch.setattr(verdict_engine.providers, "llm_call", _stub_llm(response))

        evaluation = evaluate(_extraction(facts=[fact]), [fe])
        assert len(evaluation.rhetorical_flags) == 1  # the blank one is dropped
        assert evaluation.rhetorical_flags[0].pattern == "Guaranteed outcome"

    def test_no_claim_short_circuits_without_llm(self, monkeypatch):
        # If the LLM were called here it would blow up — proving we never call it.
        def _boom(**_kwargs):
            raise AssertionError("LLM must not be called when there is no claim.")

        monkeypatch.setattr(verdict_engine.providers, "llm_call", _boom)
        evaluation = evaluate(_extraction(claim_found=False), [])
        assert evaluation.verdicts == []
        assert evaluation.rhetorical_flags == []

    def test_missing_fact_index_defaults_to_insufficient(self, monkeypatch):
        f0, f1 = _fact("Fact zero."), _fact("Fact one.")
        fes = [
            FactEvidence(atomic_fact=f0, evidence=[_evidence()]),
            FactEvidence(atomic_fact=f1, evidence=[_evidence("Mayo", 2)]),
        ]
        # Model only returns a verdict for fact 0.
        response = {
            "fact_verdicts": [
                {
                    "fact_index": 0,
                    "verdict": "Partially Supported",
                    "reasoning": "Some support.",
                    "evidence_stances": [{"evidence_index": 0, "stance": "supporting"}],
                }
            ],
            "rhetorical_flags": [],
        }
        monkeypatch.setattr(verdict_engine.providers, "llm_call", _stub_llm(response))

        verdicts = evaluate_claim(_extraction(facts=[f0, f1]), fes)
        assert verdicts[0].verdict is Verdict.PARTIALLY_SUPPORTED
        # Fact 1 got no verdict from the model -> safe default, evidence preserved.
        assert verdicts[1].verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert [e.source_name for e in verdicts[1].neutral_evidence] == ["Mayo"]

    def test_llm_error_degrades_to_insufficient(self, monkeypatch):
        fact = _fact()
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence()])

        def _raise(**_kwargs):
            raise RuntimeError("API exploded")

        monkeypatch.setattr(verdict_engine.providers, "llm_call", _raise)
        verdicts = evaluate_claim(_extraction(facts=[fact]), [fe])
        assert verdicts[0].verdict is Verdict.INSUFFICIENT_EVIDENCE
        assert "could not be completed" in verdicts[0].reasoning

    def test_budget_warning_propagates(self, monkeypatch):
        fact = _fact()
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence()])

        def _budget(**_kwargs):
            raise verdict_engine.providers.BudgetWarning("no tokens left")

        monkeypatch.setattr(verdict_engine.providers, "llm_call", _budget)
        with pytest.raises(verdict_engine.providers.BudgetWarning):
            evaluate_claim(_extraction(facts=[fact]), [fe])

    def test_unknown_verdict_string_defaults_safely(self, monkeypatch):
        fact = _fact()
        fe = FactEvidence(atomic_fact=fact, evidence=[])
        response = {
            "fact_verdicts": [
                {
                    "fact_index": 0,
                    "verdict": "Definitely True",  # not one of the seven
                    "reasoning": "…",
                    "evidence_stances": [],
                }
            ],
            "rhetorical_flags": [],
        }
        monkeypatch.setattr(verdict_engine.providers, "llm_call", _stub_llm(response))
        verdicts = evaluate_claim(_extraction(facts=[fact]), [fe])
        assert verdicts[0].verdict is Verdict.INSUFFICIENT_EVIDENCE


# --------------------------------------------------------------------------- #
# Dossier builder — assembly
# --------------------------------------------------------------------------- #
class TestDossierAssembly:
    def test_build_dossier_populates_all_fields(self, monkeypatch):
        monkeypatch.setattr(
            dossier_builder.providers,
            "llm_call",
            _stub_llm({"narrative_summary": "In plain terms, the evidence does not support this."}),
        )
        extraction = _extraction()
        verdict = AtomicVerdict(
            atomic_fact=extraction.atomic_facts[0],
            verdict=Verdict.STRONGLY_REFUTED,
            opposing_evidence=[_evidence("Cochrane", 1, stance=EvidenceStance.OPPOSING)],
            reasoning="A Cochrane review found no effect.",
        )
        flags = [RhetoricalFlag(pattern="Guaranteed outcome", explanation="…", excerpt="melts")]

        dossier = build_dossier(
            original_input=extraction.original_text,
            claim_extraction=extraction,
            verdicts=[verdict],
            source_format=SourceFormat.TEXT,
            rhetorical_flags=flags,
        )
        assert isinstance(dossier, Dossier)
        assert dossier.id is not None
        assert dossier.timestamp is not None
        assert len(dossier.verdicts) == 1
        assert dossier.rhetorical_flags[0].pattern == "Guaranteed outcome"
        assert "does not support" in dossier.narrative_summary

    def test_no_claim_skips_llm_and_uses_canned_summary(self, monkeypatch):
        def _boom(**_kwargs):
            raise AssertionError("No narrative LLM call for a no-claim dossier.")

        monkeypatch.setattr(dossier_builder.providers, "llm_call", _boom)
        extraction = _extraction(claim_found=False)
        dossier = build_dossier(
            original_input="I like tea.",
            claim_extraction=extraction,
            verdicts=[],
            source_format=SourceFormat.TEXT,
        )
        assert "No evaluable health claim" in dossier.narrative_summary

    def test_narrative_budget_warning_falls_back(self, monkeypatch):
        def _budget(**_kwargs):
            raise dossier_builder.providers.BudgetWarning("out of tokens")

        monkeypatch.setattr(dossier_builder.providers, "llm_call", _budget)
        extraction = _extraction()
        verdict = AtomicVerdict(
            atomic_fact=extraction.atomic_facts[0],
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            reasoning="Thin evidence.",
        )
        dossier = build_dossier(
            original_input=extraction.original_text,
            claim_extraction=extraction,
            verdicts=[verdict],
            source_format=SourceFormat.TEXT,
        )
        # Deterministic fallback still summarizes the verdict tally.
        assert "Insufficient Evidence" in dossier.narrative_summary


# --------------------------------------------------------------------------- #
# Dossier builder — rendering
# --------------------------------------------------------------------------- #
class TestDossierRendering:
    def _sample_dossier(self):
        extraction = _extraction()
        verdict = AtomicVerdict(
            atomic_fact=extraction.atomic_facts[0],
            verdict=Verdict.STRONGLY_REFUTED,
            opposing_evidence=[_evidence("Cochrane", 1, stance=EvidenceStance.OPPOSING)],
            neutral_evidence=[_evidence("HealthBlog", 4, stance=EvidenceStance.NEUTRAL)],
            reasoning="A 2019 Cochrane review (T1) found no effect on body weight.",
        )
        return Dossier(
            original_input=extraction.original_text,
            claim_extraction=extraction,
            verdicts=[verdict],
            rhetorical_flags=[
                RhetoricalFlag(
                    pattern="Guaranteed outcome",
                    explanation="Absolute wording implies a guaranteed result.",
                    excerpt="melts belly fat",
                )
            ],
            narrative_summary="Overall, the evidence does not support this claim.",
        )

    def test_format_contains_key_sections(self):
        text = format_dossier(self._sample_dossier())
        assert "EVIDENCE DOSSIER" in text
        assert "Strongly Refuted" in text
        assert "[T1]" in text  # tier badge on cited evidence
        assert "Guaranteed outcome" in text  # rhetorical flag surfaced
        assert "Cochrane" in text
        assert "does not support this claim" in text  # narrative
        # Stance markers present for opposing (-) and neutral (.) evidence.
        assert "(-)" in text
        assert "(.)" in text

    def test_format_handles_no_claim(self):
        extraction = _extraction(claim_found=False)
        dossier = Dossier(
            original_input="Just my opinion about tea.",
            claim_extraction=extraction,
            verdicts=[],
            narrative_summary="No evaluable health claim was found.",
        )
        text = format_dossier(dossier)
        assert "no evaluable health claim found" in text.lower()
