"""API layer tests (weeks 9-10) — offline.

These exercise the FastAPI adapter only: routing, the 'exactly one input' rule,
status-code mapping, the dossier cache, and Dossier/ClaimExtractionResult
serialization. The whole pipeline is stubbed at the server boundary
(`api.server.check_claim` and `api.server.complete_dossier`), so no network or
API key is touched.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from api import server
from schemas.models import (
    AtomicFact,
    AtomicVerdict,
    ClaimExtractionResult,
    Dossier,
    Evidence,
    EvidenceStance,
    FactType,
    RhetoricalFlag,
    SourceFormat,
    Verdict,
)

client = TestClient(server.app)


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _extraction(claim_found=True):
    facts = (
        [AtomicFact(text="Cumin water causes belly-fat loss.", fact_type=FactType.CAUSAL,
                    original_context="melts belly fat")]
        if claim_found else []
    )
    return ClaimExtractionResult(
        original_text="Cumin water melts belly fat in two weeks.",
        primary_claim="Cumin water melts belly fat in two weeks." if claim_found else "",
        atomic_facts=facts,
        claim_found=claim_found,
        source_format=SourceFormat.TEXT,
    )


def _dossier(extraction):
    evidence = Evidence(
        content="A systematic review of cumin supplementation.",
        summary="No significant effect on body weight.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/25766448/",
        source_name="Annals of Nutrition & Metabolism",
        source_tier=2,
        relevance_score=0.33,
        evidence_stance=EvidenceStance.OPPOSING,
    )
    verdict = AtomicVerdict(
        atomic_fact=extraction.atomic_facts[0],
        verdict=Verdict.PARTIALLY_SUPPORTED,
        opposing_evidence=[evidence],
        reasoning="Trials tested cumin powder, not cumin water.",
    )
    return Dossier(
        original_input=extraction.original_text,
        claim_extraction=extraction,
        verdicts=[verdict],
        rhetorical_flags=[RhetoricalFlag(pattern="Guaranteed outcome", explanation="…", excerpt="melts")],
        narrative_summary="The evidence only partially supports this.",
    )


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Every test starts with an empty dossier cache (it is process-global)."""
    server.dossier_cache.clear()
    yield
    server.dossier_cache.clear()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub the full pipeline at the server boundary; return the extraction used."""
    extraction = _extraction(claim_found=True)
    monkeypatch.setattr(server, "check_claim", lambda **_kw: extraction)
    monkeypatch.setattr(server, "complete_dossier", lambda ex, **_kw: _dossier(ex))
    return extraction


# --------------------------------------------------------------------------- #
# Health + basic routing
# --------------------------------------------------------------------------- #
def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# Input validation -> 400
# --------------------------------------------------------------------------- #
def test_no_input_is_400():
    resp = client.post("/api/check", json={})
    assert resp.status_code == 400
    assert "exactly one" in resp.json()["detail"].lower()


def test_multiple_inputs_is_400():
    resp = client.post("/api/check", json={"text": "x", "url": "https://a.com"})
    assert resp.status_code == 400


def test_blank_text_is_400():
    resp = client.post("/api/check", json={"text": "   "})
    assert resp.status_code == 400


def test_bad_base64_image_is_400(monkeypatch):
    # check_claim shouldn't even be reached — decoding fails first.
    resp = client.post("/api/check", json={"image": "!!!not base64!!!"})
    assert resp.status_code == 400
    assert "base64" in resp.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# /api/extract
# --------------------------------------------------------------------------- #
def test_extract_returns_extraction(monkeypatch):
    monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction(True))
    resp = client.post("/api/extract", json={"text": "Cumin water melts belly fat."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_found"] is True
    assert body["atomic_facts"][0]["fact_type"] == "causal"


def test_extract_no_claim_is_200(monkeypatch):
    # Reporting "no claim" is a valid successful extraction, not an error.
    monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction(False))
    resp = client.post("/api/extract", json={"text": "I like tea."})
    assert resp.status_code == 200
    assert resp.json()["claim_found"] is False


# --------------------------------------------------------------------------- #
# /api/check
# --------------------------------------------------------------------------- #
def test_check_full_dossier(stub_pipeline):
    resp = client.post("/api/check", json={"text": "Cumin water melts belly fat in two weeks."})
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body and body["id"]
    assert "timestamp" in body
    assert body["verdicts"][0]["verdict"] == "Partially Supported"
    assert body["verdicts"][0]["opposing_evidence"][0]["source_tier"] == 2
    assert body["rhetorical_flags"][0]["pattern"] == "Guaranteed outcome"
    assert body["narrative_summary"]


def test_evidence_carries_human_readable_tier(stub_pipeline):
    """The tier's plain-language name ships alongside the number.

    It's a computed field, so a `response_model` that filtered computed fields
    would silently drop it and leave the UI with nothing to render.
    """
    resp = client.post("/api/check", json={"text": "Cumin water melts belly fat."})
    evidence = resp.json()["verdicts"][0]["opposing_evidence"][0]
    assert evidence["source_tier"] == 2
    assert evidence["human_readable_tier"] == "Peer-reviewed Study"


def test_check_no_claim_is_422(monkeypatch):
    monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction(False))
    resp = client.post("/api/check", json={"text": "I like tea."})
    assert resp.status_code == 422
    assert "no evaluable health claim" in resp.json()["detail"].lower()


def test_check_budget_warning_is_503(monkeypatch):
    def _raise(**_kw):
        raise server.providers.BudgetWarning("out of tokens")

    monkeypatch.setattr(server, "check_claim", _raise)
    resp = client.post("/api/check", json={"text": "Cumin water melts belly fat."})
    assert resp.status_code == 503
    assert "budget" in resp.json()["detail"].lower()


def test_check_pipeline_error_is_500(monkeypatch):
    monkeypatch.setattr(server, "check_claim", lambda **_kw: _extraction(True))

    def _boom(ex, **_kw):
        raise RuntimeError("evaluator exploded")

    monkeypatch.setattr(server, "complete_dossier", _boom)
    resp = client.post("/api/check", json={"text": "Cumin water melts belly fat."})
    assert resp.status_code == 500
    assert "pipeline error" in resp.json()["detail"].lower()


def test_check_accepts_data_url_image(stub_pipeline):
    # A well-formed data: URL should decode and flow through to the stubbed pipeline.
    tiny_png = base64.b64encode(b"\x89PNG\r\n\x1a\n fake bytes").decode()
    resp = client.post("/api/check", json={"image": f"data:image/png;base64,{tiny_png}"})
    assert resp.status_code == 200
    assert resp.json()["verdicts"][0]["verdict"] == "Partially Supported"
