"""Week 1 test suite.

These tests run offline — they never hit the Anthropic API. The one extractor
test stubs out `providers.llm_call`. The corpus of 50+ live, diverse claim test
cases (per CLAUDE.md) will be added once the extractor is exercised against the
real model.
"""

from __future__ import annotations

import importlib

import pytest

from input.text_input import clean_text
from input.url_input import URLExtractionError, extract_from_url
from schemas.models import (
    AtomicFact,
    ClaimExtractionResult,
    Dossier,
    FactType,
    SourceFormat,
    Verdict,
)


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
class TestCleanText:
    def test_strips_surrounding_whitespace(self):
        assert clean_text("  hello world  \n") == "hello world"

    def test_collapses_excess_blank_lines(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_normalizes_unicode_fullwidth(self):
        # Full-width "ABC" normalizes (NFKC) to ASCII "ABC".
        assert clean_text("ＡＢＣ") == "ABC"

    def test_normalizes_line_endings(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            clean_text("   \n  \t ")

    def test_non_str_raises(self):
        with pytest.raises(TypeError):
            clean_text(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# URL handler — validation path (no network)
# --------------------------------------------------------------------------- #
class TestURLValidation:
    @pytest.mark.parametrize("bad", ["", "not a url", "ftp://x.com", "example.com", 42])
    def test_invalid_urls_raise(self, bad):
        with pytest.raises(URLExtractionError):
            extract_from_url(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class TestModels:
    def test_dossier_gets_uuid_and_timestamp(self):
        extraction = ClaimExtractionResult(
            original_text="x", primary_claim="", atomic_facts=[], claim_found=False
        )
        d1 = Dossier(original_input="x", claim_extraction=extraction)
        d2 = Dossier(original_input="x", claim_extraction=extraction)
        assert d1.id != d2.id  # unique per dossier
        assert d1.timestamp is not None

    def test_atomic_fact_requires_valid_type(self):
        with pytest.raises(Exception):
            AtomicFact(text="x", fact_type="nonsense", original_context="y")  # type: ignore[arg-type]

    def test_evidence_tier_bounds(self):
        from schemas.models import Evidence

        with pytest.raises(Exception):
            Evidence(
                content="c", source_url="u", source_name="n",
                source_tier=5, relevance_score=0.5,
            )

    def test_verdict_enum_has_seven_categories(self):
        assert len(list(Verdict)) == 7

    def test_claim_found_requires_at_least_one_atom(self):
        # claim_found=True with no atomic facts violates the invariant.
        with pytest.raises(Exception):
            ClaimExtractionResult(
                original_text="x",
                primary_claim="Some claim.",
                atomic_facts=[],
                claim_found=True,
            )

    def test_no_claim_forbids_atoms(self):
        # claim_found=False with atoms present also violates the invariant.
        fact = AtomicFact(text="f", fact_type=FactType.CAUSAL, original_context="c")
        with pytest.raises(Exception):
            ClaimExtractionResult(
                original_text="x",
                primary_claim="",
                atomic_facts=[fact],
                claim_found=False,
            )

    def test_valid_found_claim_with_atom_passes(self):
        fact = AtomicFact(text="f", fact_type=FactType.CAUSAL, original_context="c")
        result = ClaimExtractionResult(
            original_text="x",
            primary_claim="Some claim.",
            atomic_facts=[fact],
            claim_found=True,
        )
        assert result.claim_found is True
        assert len(result.atomic_facts) == 1


# --------------------------------------------------------------------------- #
# Daily usage tracking + soft budget guardrail
# --------------------------------------------------------------------------- #
class TestBudgetGuardrail:
    @pytest.fixture
    def fresh_providers(self, tmp_path, monkeypatch):
        """Reload providers with an isolated usage file and tiny budget."""
        from config import settings

        monkeypatch.setattr(settings, "DAILY_USAGE_FILE", str(tmp_path / "usage.json"))
        monkeypatch.setattr(settings, "DAILY_TOKEN_BUDGET", 100)
        providers = importlib.import_module("config.providers")
        return importlib.reload(providers)

    def test_records_and_reads_usage(self, fresh_providers):
        assert fresh_providers.get_daily_usage() == 0
        total = fresh_providers._record_usage(30, 20)
        assert total == 50
        assert fresh_providers.get_daily_usage() == 50

    def test_raises_budget_warning_when_exhausted(self, fresh_providers):
        fresh_providers._record_usage(80, 40)  # 120 > 100 budget
        with pytest.raises(fresh_providers.BudgetWarning):
            fresh_providers.llm_call("hi", model_tier="lightweight")

    def test_allow_over_budget_bypasses_check(self, fresh_providers, monkeypatch):
        fresh_providers._record_usage(80, 40)  # over budget

        # Stub the provider so no real API call happens.
        class _Stub:
            def complete(self, **_kwargs):
                return "ok", 1, 1

        monkeypatch.setattr(fresh_providers, "_get_provider", lambda _name: _Stub())
        out = fresh_providers.llm_call("hi", model_tier="lightweight", allow_over_budget=True)
        assert out == "ok"


# --------------------------------------------------------------------------- #
# Claim extractor (provider stubbed — no API key needed)
# --------------------------------------------------------------------------- #
class TestClaimExtractor:
    def test_empty_text_short_circuits(self):
        from core.claim_extractor import extract_claims

        result = extract_claims("   ")
        assert result.claim_found is False
        assert result.atomic_facts == []

    def test_maps_structured_output_to_model(self, monkeypatch):
        from core import claim_extractor

        fake = {
            "claim_found": True,
            "primary_claim": "Green tea boosts metabolism.",
            "atomic_facts": [
                {
                    "text": "Green tea boosts metabolism by 12%.",
                    "fact_type": "quantitative",
                    "original_context": "boosts your metabolism by 12%",
                }
            ],
        }
        monkeypatch.setattr(claim_extractor.providers, "llm_call", lambda **_kw: fake)

        result = claim_extractor.extract_claims(
            "Green tea boosts your metabolism by 12%.",
            source_format=SourceFormat.TEXT,
        )
        assert result.claim_found is True
        assert result.source_format is SourceFormat.TEXT
        assert len(result.atomic_facts) == 1
        assert result.atomic_facts[0].fact_type is FactType.QUANTITATIVE
        assert result.original_text  # preserved
