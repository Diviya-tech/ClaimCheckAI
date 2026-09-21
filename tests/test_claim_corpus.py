"""Weeks 11-12 — diverse claim corpus + edge-case inputs.

Every test runs offline: the LLM is stubbed with the structured output a
well-behaved model returns for each kind of claim, and the tests lock down what
the PIPELINE does with it — the decomposition invariants, the verdict-engine
guardrails, and how odd inputs (empty, huge, exotic characters, non-English)
are normalized. The prompts are checked for the instructions that produce
these behaviors, so a prompt regression fails here before it reaches a user.
"""

from __future__ import annotations

import pytest

from core import claim_extractor, verdict_engine
from core.claim_extractor import extract_claims
from core.verdict_engine import evaluate
from input.text_input import clean_text
from schemas.models import (
    AtomicFact,
    ClaimExtractionResult,
    Evidence,
    EvidenceApplicability,
    EvidenceStance,
    FactEvidence,
    FactType,
    SourceFormat,
    Verdict,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _stub(monkeypatch, module, response):
    monkeypatch.setattr(module.providers, "llm_call", lambda **_kw: response)


def _fact(text, ftype=FactType.CAUSAL, context="…"):
    return AtomicFact(text=text, fact_type=ftype, original_context=context)


def _extraction(text, claim, facts):
    return ClaimExtractionResult(
        original_text=text,
        primary_claim=claim,
        atomic_facts=facts,
        claim_found=bool(facts),
        source_format=SourceFormat.TEXT,
    )


def _evidence(name, tier=2, applicability=EvidenceApplicability.UNKNOWN, summary="…"):
    return Evidence(
        content=summary,
        summary=summary,
        source_url=f"https://example.org/{name.lower().replace(' ', '-')}",
        source_name=name,
        source_tier=tier,
        relevance_score=0.6,
        applicability=applicability,
    )


def _verdict_response(verdict, stances, reasoning="…", flags=None):
    return {
        "fact_verdicts": [
            {
                "fact_index": 0,
                "verdict": verdict,
                "reasoning": reasoning,
                "evidence_stances": [
                    {"evidence_index": i, "stance": st, "applicability": ap}
                    for i, (st, ap) in enumerate(stances)
                ],
            }
        ],
        "rhetorical_flags": flags or [],
    }


# --------------------------------------------------------------------------- #
# Claim types — extraction
# --------------------------------------------------------------------------- #
class TestClaimTypesExtraction:
    def test_multi_claim_input_decomposes_into_one_fact_per_assertion(self, monkeypatch):
        text = "Turmeric cures cancer, boosts immunity, and reverses diabetes"
        _stub(monkeypatch, claim_extractor, {
            "claim_found": True,
            "primary_claim": "Turmeric cures cancer, boosts immunity, and reverses diabetes.",
            "atomic_facts": [
                {"text": "Turmeric cures cancer.", "fact_type": "causal",
                 "original_context": "Turmeric cures cancer"},
                {"text": "Turmeric boosts immune function.", "fact_type": "causal",
                 "original_context": "boosts immunity"},
                {"text": "Turmeric reverses diabetes.", "fact_type": "causal",
                 "original_context": "reverses diabetes"},
            ],
        })
        result = extract_claims(text)
        assert result.claim_found
        assert len(result.atomic_facts) == 3
        # Every atom is self-contained: names the substance, no bare pronouns.
        for fact in result.atomic_facts:
            assert "turmeric" in fact.text.lower()
            assert not fact.text.lower().startswith(("it ", "this ", "they "))

    def test_no_claim_input_yields_claim_found_false(self, monkeypatch):
        _stub(monkeypatch, claim_extractor, {
            "claim_found": False, "primary_claim": "", "atomic_facts": [],
        })
        result = extract_claims("I love smoothies in the morning")
        assert result.claim_found is False
        assert result.atomic_facts == []
        assert result.primary_claim == ""

    def test_sarcastic_input_is_not_a_claim(self, monkeypatch):
        # Sarcasm asserts the opposite of its literal words; a careful extractor
        # reports no evaluable claim rather than "essential oils cure everything".
        _stub(monkeypatch, claim_extractor, {
            "claim_found": False, "primary_claim": "", "atomic_facts": [],
        })
        result = extract_claims("Yeah sure essential oils cure everything")
        assert result.claim_found is False
        assert result.atomic_facts == []

    def test_ambiguous_claim_still_decomposes_to_testable_atom(self, monkeypatch):
        text = "Moderate alcohol consumption is good for heart health"
        _stub(monkeypatch, claim_extractor, {
            "claim_found": True,
            "primary_claim": text + ".",
            "atomic_facts": [
                {"text": "Moderate alcohol consumption reduces cardiovascular disease risk.",
                 "fact_type": "causal", "original_context": "is good for heart health"},
            ],
        })
        result = extract_claims(text)
        assert result.claim_found
        # "good for heart health" was sharpened into a measurable outcome.
        assert "cardiovascular" in result.atomic_facts[0].text.lower()
        assert result.atomic_facts[0].fact_type is FactType.CAUSAL

    def test_form_mismatch_claim_gets_separate_form_fact(self, monkeypatch):
        # "Turmeric tea cures arthritis": the effect and the FORM are distinct
        # atoms, so a later stage can judge tea-vs-extract on its own.
        _stub(monkeypatch, claim_extractor, {
            "claim_found": True,
            "primary_claim": "Turmeric tea cures arthritis.",
            "atomic_facts": [
                {"text": "Turmeric (curcumin) reduces arthritis symptoms.",
                 "fact_type": "causal", "original_context": "cures arthritis"},
                {"text": "Turmeric tea delivers curcumin at a dose comparable to "
                         "the amounts studied in clinical trials.",
                 "fact_type": "quantitative", "original_context": "Turmeric tea"},
            ],
        })
        result = extract_claims("Turmeric tea cures arthritis")
        texts = [f.text.lower() for f in result.atomic_facts]
        assert any("tea" in t and "dose" in t for t in texts)
        assert any("arthritis" in t for t in texts)

    def test_extractor_prompt_requires_form_decomposition(self):
        prompt = " ".join(claim_extractor._SYSTEM_PROMPT.lower().split())
        assert "form & preparation" in prompt
        assert "turmeric tea" in prompt
        assert "own atomic fact" in prompt

    def test_invariant_rejects_found_claim_with_no_atoms(self, monkeypatch):
        # A model that says "claim found" but decomposes nothing is caught at the
        # boundary instead of flowing downstream.
        _stub(monkeypatch, claim_extractor, {
            "claim_found": True, "primary_claim": "Vaccines prevent measles.",
            "atomic_facts": [],
        })
        with pytest.raises(Exception):
            extract_claims("Vaccines prevent measles")


# --------------------------------------------------------------------------- #
# Claim types — verdicts
# --------------------------------------------------------------------------- #
class TestClaimTypesVerdicts:
    def test_vague_claim_maps_to_too_vague(self, monkeypatch):
        fact = _fact("Eating healthy is good for you.")
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence("WHO", 1)])
        _stub(monkeypatch, verdict_engine, _verdict_response(
            "Too Vague to Evaluate", [("neutral", "indirect")],
            reasoning="'Healthy' and 'good for you' are undefined.",
        ))
        ev = evaluate(_extraction("Eating healthy is good for you", "…", [fact]), [fe])
        assert ev.verdicts[0].verdict is Verdict.TOO_VAGUE
        # Nothing was counted as support for an unfalsifiable statement.
        assert ev.verdicts[0].supporting_evidence == []

    def test_well_supported_claim_with_direct_evidence_is_strongly_supported(self, monkeypatch):
        fact = _fact("Measles vaccination prevents measles infection.")
        fe = FactEvidence(atomic_fact=fact, evidence=[
            _evidence("Cochrane", 1), _evidence("CDC", 1), _evidence("PubMed RCT", 2),
        ])
        _stub(monkeypatch, verdict_engine, _verdict_response(
            "Strongly Supported",
            [("supporting", "direct"), ("supporting", "direct"), ("supporting", "direct")],
        ))
        ev = evaluate(_extraction("Vaccines prevent measles", "…", [fact]), [fe])
        v = ev.verdicts[0]
        assert v.verdict is Verdict.STRONGLY_SUPPORTED
        assert len(v.supporting_evidence) == 3
        assert all(e.applicability is EvidenceApplicability.DIRECT for e in v.supporting_evidence)

    def test_form_mismatch_caps_strongly_supported_at_partially(self, monkeypatch):
        # Extract trials support the effect, but nothing tested TEA: the model's
        # "Strongly Supported" is capped in code, not left to the prompt.
        fact = _fact("Turmeric tea reduces arthritis symptoms.")
        fe = FactEvidence(atomic_fact=fact, evidence=[
            _evidence("Curcumin extract RCT", 2), _evidence("Curcumin meta-analysis", 1),
        ])
        _stub(monkeypatch, verdict_engine, _verdict_response(
            "Strongly Supported",
            [("supporting", "indirect"), ("supporting", "indirect")],
            reasoning="Trials of 1 g curcumin extract improved symptoms.",
        ))
        ev = evaluate(_extraction("Turmeric tea cures arthritis", "…", [fact]), [fe])
        v = ev.verdicts[0]
        assert v.verdict is Verdict.PARTIALLY_SUPPORTED
        assert "capped" in v.reasoning.lower()
        assert "form" in v.reasoning.lower()

    def test_association_claim_prompt_distinguishes_causation(self):
        prompt = verdict_engine._SYSTEM_PROMPT.lower()
        assert '"association" is not "causation"' in prompt
        assert "randomized controlled trials" in prompt

    def test_association_evidence_only_partially_supports_causal_fact(self, monkeypatch):
        # Cohort data "people who drink coffee live longer" is indirect for a
        # causal reading; the engine keeps it at Partially even if a model
        # over-reaches.
        fact = _fact("Drinking coffee increases lifespan.")
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence("NIH-AARP cohort", 2)])
        _stub(monkeypatch, verdict_engine, _verdict_response(
            "Strongly Supported", [("supporting", "indirect")],
            reasoning="A large cohort found lower mortality among coffee drinkers.",
        ))
        ev = evaluate(_extraction("People who drink coffee live longer", "…", [fact]), [fe])
        assert ev.verdicts[0].verdict is Verdict.PARTIALLY_SUPPORTED

    def test_wrong_population_evidence_cannot_strongly_refute(self, monkeypatch):
        # A null result in postmenopausal women is indirect for a claim aimed at
        # everyone; "Strongly Refuted" is downgraded to "Partially Refuted".
        fact = _fact("Soy isoflavones reduce hot-flash frequency in adults.")
        fe = FactEvidence(atomic_fact=fact, evidence=[_evidence("RCT in Japanese men", 2)])
        _stub(monkeypatch, verdict_engine, _verdict_response(
            "Strongly Refuted", [("opposing", "indirect")],
        ))
        ev = evaluate(_extraction("…", "…", [fact]), [fe])
        v = ev.verdicts[0]
        assert v.verdict is Verdict.PARTIALLY_REFUTED
        assert v.opposing_evidence[0].applicability is EvidenceApplicability.INDIRECT

    def test_population_rule_present_in_prompt(self):
        prompt = verdict_engine._SYSTEM_PROMPT.lower()
        assert '"works in population x" is not "works for everyone"' in prompt

    def test_multi_claim_verdicts_are_independent(self, monkeypatch):
        facts = [
            _fact("Turmeric cures cancer."),
            _fact("Turmeric boosts immune function."),
            _fact("Turmeric reverses diabetes."),
        ]
        fes = [FactEvidence(atomic_fact=f, evidence=[_evidence(f"src{i}")]) for i, f in enumerate(facts)]
        _stub(monkeypatch, verdict_engine, {
            "fact_verdicts": [
                {"fact_index": 0, "verdict": "Strongly Refuted", "reasoning": "…",
                 "evidence_stances": [{"evidence_index": 0, "stance": "opposing", "applicability": "direct"}]},
                {"fact_index": 1, "verdict": "Insufficient Evidence", "reasoning": "…",
                 "evidence_stances": [{"evidence_index": 0, "stance": "neutral", "applicability": "indirect"}]},
                {"fact_index": 2, "verdict": "Partially Refuted", "reasoning": "…",
                 "evidence_stances": [{"evidence_index": 0, "stance": "opposing", "applicability": "indirect"}]},
            ],
            "rhetorical_flags": [],
        })
        ev = evaluate(_extraction("…", "…", facts), fes)
        assert [v.verdict for v in ev.verdicts] == [
            Verdict.STRONGLY_REFUTED, Verdict.INSUFFICIENT_EVIDENCE, Verdict.PARTIALLY_REFUTED,
        ]


# --------------------------------------------------------------------------- #
# Edge-case inputs
# --------------------------------------------------------------------------- #
class TestEdgeCaseInputs:
    def test_empty_string_is_rejected_by_normalizer(self):
        with pytest.raises(ValueError):
            clean_text("")

    def test_empty_string_short_circuits_extractor_without_llm(self, monkeypatch):
        def _boom(**_kw):
            raise AssertionError("LLM must not be called for empty input")

        monkeypatch.setattr(claim_extractor.providers, "llm_call", _boom)
        result = extract_claims("")
        assert result.claim_found is False

    def test_extremely_long_text_is_normalized_intact(self):
        text = ("Cumin water melts belly fat. " * 20_000).strip()
        cleaned = clean_text(text)
        assert len(cleaned) == len(text)
        assert cleaned.startswith("Cumin water")

    def test_extremely_long_text_collapses_blank_line_runs(self):
        text = "claim\n" + ("\n" * 5_000) + "more"
        assert clean_text(text) == "claim\n\nmore"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Green tea 🍵 boosts metabolism!!! 💪", "Green tea 🍵 boosts metabolism!!! 💪"),
            ("Café con leche ≠ dañino", "Café con leche ≠ dañino"),
            ("β-carotene → vitamin A", "β-carotene → vitamin A"),
            ("<b>Bold</b> & 'quoted' \"claim\"", "<b>Bold</b> & 'quoted' \"claim\""),
        ],
    )
    def test_special_characters_are_preserved(self, raw, expected):
        assert clean_text(raw) == expected

    def test_zero_width_and_control_characters_are_stripped(self):
        smuggled = "Cumin​ water‍ melts\x00 belly‮ fat"
        assert clean_text(smuggled) == "Cumin water melts belly fat"

    @pytest.mark.parametrize(
        "text",
        [
            "El agua de comino derrite la grasa abdominal en dos semanas.",
            "Le thé vert fait fondre la graisse du ventre.",
            "緑茶は二週間でお腹の脂肪を溶かす。",
            "Зелёный чай сжигает жир на животе за две недели.",
            "الشاي الأخضر يذيب دهون البطن في أسبوعين.",
        ],
    )
    def test_non_english_text_is_preserved_and_extractable(self, text, monkeypatch):
        assert clean_text(text) == text
        _stub(monkeypatch, claim_extractor, {
            "claim_found": True,
            "primary_claim": "Green tea melts belly fat in two weeks.",
            "atomic_facts": [
                {"text": "Green tea causes belly-fat loss.", "fact_type": "causal",
                 "original_context": text},
            ],
        })
        result = extract_claims(text)
        assert result.claim_found
        assert result.original_text == text  # the source language is kept verbatim

    def test_whitespace_only_input_is_rejected(self):
        with pytest.raises(ValueError):
            clean_text(" \n\t​ ")

    def test_non_string_input_is_rejected(self):
        with pytest.raises(TypeError):
            clean_text(None)  # type: ignore[arg-type]
