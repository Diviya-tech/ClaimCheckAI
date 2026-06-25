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

    def test_system_prompt_requires_self_contained_facts(self):
        # Guards against a prompt regression that would let atoms reference
        # "the effect" / "it" without naming the subject — which destroys
        # downstream retrieval relevance.
        from core.claim_extractor import _SYSTEM_PROMPT

        prompt = _SYSTEM_PROMPT.lower()
        assert "self-contained" in prompt
        assert "the effect" in prompt  # the explicit anti-pattern example
        assert "pronoun" in prompt or "subject" in prompt

    def test_self_contained_fact_flows_through(self, monkeypatch):
        from core import claim_extractor

        # Simulate the model now returning a subject-carrying temporal fact.
        fake = {
            "claim_found": True,
            "primary_claim": "Cumin water melts belly fat in two weeks.",
            "atomic_facts": [
                {
                    "text": "Cumin water causes belly-fat loss.",
                    "fact_type": "causal",
                    "original_context": "Cumin water melts belly fat in two weeks",
                },
                {
                    "text": "Cumin water produces belly-fat loss within two weeks.",
                    "fact_type": "temporal",
                    "original_context": "in two weeks",
                },
            ],
        }
        monkeypatch.setattr(claim_extractor.providers, "llm_call", lambda **_kw: fake)

        result = claim_extractor.extract_claims("Cumin water melts belly fat in two weeks.")
        temporal = result.atomic_facts[1]
        # The temporal atom names the subject ("cumin water") rather than "the effect".
        assert "cumin water" in temporal.text.lower()

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


# --------------------------------------------------------------------------- #
# Image / screenshot handler (LLM stubbed — no API key needed)
# --------------------------------------------------------------------------- #
class TestImageInput:
    def test_missing_file_raises(self):
        from input.image_input import ImageExtractionError, extract_from_image

        with pytest.raises(ImageExtractionError):
            extract_from_image("nope/does_not_exist.png")

    def test_unsupported_format_raises(self, tmp_path):
        from input.image_input import ImageExtractionError, extract_from_image

        bad = tmp_path / "shot.gif"
        bad.write_bytes(b"not really a gif")
        with pytest.raises(ImageExtractionError):
            extract_from_image(str(bad))

    def test_no_claim_sentinel_raises(self, tmp_path, monkeypatch):
        from input import image_input

        img = tmp_path / "shot.png"
        img.write_bytes(b"fake png bytes")
        monkeypatch.setattr(
            image_input.providers, "llm_call", lambda **_kw: image_input._NO_CLAIM_SENTINEL
        )
        with pytest.raises(image_input.ImageExtractionError):
            image_input.extract_from_image(str(img))

    def test_valid_image_returns_clean_claim(self, tmp_path, monkeypatch):
        from input import image_input

        img = tmp_path / "shot.jpg"
        img.write_bytes(b"fake jpg bytes")
        monkeypatch.setattr(
            image_input.providers,
            "llm_call",
            lambda **_kw: "  Green tea melts belly fat in two weeks.  ",
        )
        out = image_input.extract_from_image(str(img))
        assert out == "Green tea melts belly fat in two weeks."

    def test_image_passes_media_type_and_premium_tier(self, tmp_path, monkeypatch):
        from input import image_input

        img = tmp_path / "shot.webp"
        img.write_bytes(b"fake webp bytes")
        captured: dict = {}

        def _fake_llm_call(**kwargs):
            captured.update(kwargs)
            return "Some health claim."

        monkeypatch.setattr(image_input.providers, "llm_call", _fake_llm_call)
        image_input.extract_from_image(str(img))

        assert captured["model_tier"] == "premium"
        assert captured["images"][0]["media_type"] == "image/webp"
        assert captured["images"][0]["data"]  # base64 payload present


# --------------------------------------------------------------------------- #
# Video handler (download / transcription / frames / vision all mocked)
# --------------------------------------------------------------------------- #
class TestVideoInput:
    def test_invalid_url_raises(self):
        from input.video_input import VideoExtractionError, extract_from_video

        with pytest.raises(VideoExtractionError):
            extract_from_video("not-a-url")

    def test_merge_dedups_visual_and_drops_spoken_overlap(self):
        from input.video_input import _merge_texts

        transcription = "celery juice detoxes your liver"
        # Same overlay repeated across frames + one line already spoken.
        visual_texts = [
            "DRINK CELERY JUICE\ncelery juice detoxes your liver",
            "DRINK CELERY JUICE",  # duplicate frame overlay
            "CLEARS ACNE IN 7 DAYS",
        ]
        visual_text, combined = _merge_texts(transcription, visual_texts)

        # Visual de-duped across frames (case-insensitive), order preserved.
        assert visual_text.splitlines() == [
            "DRINK CELERY JUICE",
            "celery juice detoxes your liver",
            "CLEARS ACNE IN 7 DAYS",
        ]
        # Combined keeps spoken text + only the on-screen lines NOT already spoken.
        assert transcription in combined
        assert "DRINK CELERY JUICE" in combined
        assert "CLEARS ACNE IN 7 DAYS" in combined
        # The overlap line appears once (from transcription), not duplicated.
        assert combined.lower().count("celery juice detoxes your liver") == 1

    def test_orchestration_merges_both_channels(self, monkeypatch):
        from input import video_input as v

        monkeypatch.setattr(v, "_ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(v, "_download_video", lambda url, td: "fake.mp4")
        monkeypatch.setattr(v, "_has_audio", lambda p: True)
        monkeypatch.setattr(v, "_extract_audio", lambda p, td: "fake.wav")
        monkeypatch.setattr(v, "_transcribe", lambda p, m=None: "vitamin C prevents colds")
        monkeypatch.setattr(v, "_probe_duration", lambda p: 12.0)
        monkeypatch.setattr(v, "_extract_frames", lambda p, td, d, n: ["f0.png", "f1.png"])
        monkeypatch.setattr(v, "_read_frame_text", lambda f, allow_over_budget=False: "MEGADOSE VITAMIN C")

        result = v.extract_from_video("https://tiktok.com/@x/video/123")
        assert result.transcription == "vitamin C prevents colds"
        assert "MEGADOSE VITAMIN C" in result.visual_text
        assert "vitamin C prevents colds" in result.combined_text
        assert "MEGADOSE VITAMIN C" in result.combined_text

    def test_no_audio_falls_back_to_visual_only(self, monkeypatch):
        from input import video_input as v

        monkeypatch.setattr(v, "_ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(v, "_download_video", lambda url, td: "fake.mp4")
        monkeypatch.setattr(v, "_has_audio", lambda p: False)  # no audio track
        monkeypatch.setattr(v, "_probe_duration", lambda p: 8.0)
        monkeypatch.setattr(v, "_extract_frames", lambda p, td, d, n: ["f0.png"])
        monkeypatch.setattr(v, "_read_frame_text", lambda f, allow_over_budget=False: "TURMERIC CURES CANCER")

        result = v.extract_from_video("https://youtube.com/shorts/abc")
        assert result.transcription == ""  # gracefully empty, not an error
        assert "TURMERIC CURES CANCER" in result.combined_text

    def test_no_speech_and_no_visual_raises(self, monkeypatch):
        from input import video_input as v

        monkeypatch.setattr(v, "_ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(v, "_download_video", lambda url, td: "fake.mp4")
        monkeypatch.setattr(v, "_has_audio", lambda p: False)
        monkeypatch.setattr(v, "_probe_duration", lambda p: 5.0)
        monkeypatch.setattr(v, "_extract_frames", lambda p, td, d, n: ["f0.png"])
        monkeypatch.setattr(v, "_read_frame_text", lambda f, allow_over_budget=False: "")  # no overlay

        with pytest.raises(v.VideoExtractionError):
            v.extract_from_video("https://instagram.com/reel/xyz")


# --------------------------------------------------------------------------- #
# Provider layer forwards images to the backend
# --------------------------------------------------------------------------- #
class TestProviderImagePassthrough:
    def test_llm_call_forwards_images_to_provider(self, monkeypatch):
        from config import providers

        captured: dict = {}

        class _Stub:
            def complete(self, **kwargs):
                captured.update(kwargs)
                return "ok", 1, 1

        monkeypatch.setattr(providers, "_get_provider", lambda _name: _Stub())
        imgs = [{"media_type": "image/png", "data": "abc123"}]
        providers.llm_call("hi", model_tier="premium", images=imgs, allow_over_budget=True)
        assert captured["images"] == imgs
