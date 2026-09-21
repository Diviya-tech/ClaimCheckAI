"""FastAPI HTTP interface over the ClaimCheck AI pipeline (weeks 9-10).

This is a thin adapter, not a second implementation. Every endpoint calls the
exact same functions the CLI (`main.py`) uses:

    check_claim()      -> input normalization + claim extraction  (from main.py)
    retrieve_evidence()-> stage 3                                 (core.evidence_retriever)
    evaluate()         -> stage 4 (verdicts + rhetorical flags)   (core.verdict_engine)
    build_dossier()    -> stage 5                                 (core.dossier_builder)

Run it with:

    uvicorn api.server:app --reload            # http://localhost:8000
    #  ...or:  python start_server.py

Endpoints:
    GET  /api/health   -> {"status": "ok"}
    POST /api/extract  -> ClaimExtractionResult   (fast: extraction only)
    POST /api/check    -> Dossier                 (full pipeline)

Smart URL routing: `url` accepts *any* link. Social-media video hosts (Instagram,
TikTok, YouTube, X/Twitter) are auto-detected and routed through the video
pipeline (yt-dlp + Whisper + vision); everything else goes through Trafilatura.
Callers who already know they have a video can still say so explicitly via
`video_url`, which always takes the video path.

Status codes:
    400  bad input        (no/many inputs, invalid base64, unreadable URL/image/video)
    422  unprocessable    (input has no evaluable health claim -> /api/check only)
    503  unavailable      (token budget exhausted, or video deps not installed)
    500  pipeline error   (anything unexpected mid-pipeline)
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import providers
from core.dossier_builder import build_dossier
from core.evidence_retriever import retrieve_evidence
from core.verdict_engine import evaluate
from input.image_input import ImageExtractionError
from input.url_input import URLExtractionError
from input.video_input import VideoExtractionError
from main import check_claim
from schemas.models import ClaimExtractionResult, Dossier

logger = logging.getLogger("claimcheck.api")

# Where the frontend runs in development (Next.js default port).
_FRONTEND_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

app = FastAPI(
    title="ClaimCheck AI",
    version="0.9.0",
    summary="Evidence-evaluation engine for health claims.",
    description=(
        "Decomposes a health claim into atomic facts, retrieves evidence from "
        "trusted medical sources, and returns a transparent evidence dossier with "
        "per-fact categorical verdicts. No numeric trust scores."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request model
# --------------------------------------------------------------------------- #
class CheckRequest(BaseModel):
    """One of the four fields must be provided (validated in the endpoint so the
    'exactly one' rule returns a clean 400 rather than a schema 422)."""

    text: str | None = Field(default=None, description="Raw claim text to analyze.")
    url: str | None = Field(
        default=None,
        description=(
            "Any link. Article URLs are read with Trafilatura; links from Instagram, "
            "TikTok, YouTube, or X/Twitter are auto-detected and run through the "
            "video pipeline instead."
        ),
    )
    image: str | None = Field(
        default=None,
        description="Base64-encoded screenshot (optionally a data: URL). PNG/JPG/WEBP.",
    )
    video_url: str | None = Field(
        default=None,
        description=(
            "Video URL (TikTok / Instagram / YouTube / X). Explicit opt-in to the "
            "video pipeline — use it for video links on hosts we don't auto-detect."
        ),
    )


# --------------------------------------------------------------------------- #
# Input plumbing
# --------------------------------------------------------------------------- #
_DATA_URL_RE = re.compile(r"^data:image/(?P<subtype>[a-z0-9.+-]+);base64,", re.IGNORECASE)
_SUBTYPE_TO_SUFFIX = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "webp": ".webp"}

# Hosts whose links are videos, not articles. Trafilatura gets nothing useful
# from these pages (the claim lives in the audio and the on-screen text), so a
# link to one is routed through the video pipeline instead. Subdomains match too
# (vm.tiktok.com, m.youtube.com, www.instagram.com, ...).
_VIDEO_HOSTS = frozenset(
    {
        "instagram.com",
        "instagr.am",
        "tiktok.com",
        "youtube.com",
        "youtube-nocookie.com",
        "youtu.be",
        "x.com",
        "twitter.com",
    }
)


def is_social_video_url(url: str) -> bool:
    """True if `url` points at a social-media video host (see `_VIDEO_HOSTS`)."""
    if not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate:
        return False
    # Users paste "instagram.com/reel/..." as often as the full https:// form;
    # urlparse only fills in `hostname` when there's a scheme.
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    host = (urlparse(candidate).hostname or "").lower()
    return any(host == known or host.endswith(f".{known}") for known in _VIDEO_HOSTS)


def _provided_fields(req: CheckRequest) -> list[str]:
    """Names of the non-empty input fields on the request."""
    present = []
    for name in ("text", "url", "image", "video_url"):
        value = getattr(req, name)
        if value is not None and value.strip():
            present.append(name)
    return present


def _require_exactly_one(req: CheckRequest) -> None:
    provided = _provided_fields(req)
    if len(provided) != 1:
        detail = (
            "Provide exactly one non-empty field: 'text', 'url', 'image', or 'video_url'."
            + (f" Got: {provided}." if provided else " Got none.")
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _decode_image(b64: str) -> tuple[bytes, str]:
    """Decode a base64 image (with or without a data: URL prefix) -> (bytes, suffix)."""
    b64 = b64.strip()
    suffix = ".png"
    match = _DATA_URL_RE.match(b64)
    if match:
        suffix = _SUBTYPE_TO_SUFFIX.get(match.group("subtype").lower(), ".png")
        b64 = b64[match.end():]
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    if not data:
        raise ValueError("Decoded image is empty.")
    return data, suffix


def _extract_from_request(req: CheckRequest) -> ClaimExtractionResult:
    """Normalize the single provided input and run claim extraction.

    Delegates entirely to `check_claim` (the CLI's own function). The only extra
    work is turning a base64 image into a temp file, since `check_claim`/the image
    handler operate on a file path.

    A `url` on a social-media video host is handed to `check_claim(video=...)`, so
    the caller never has to know which pipeline their link needs.
    """
    if req.url is not None and req.url.strip():
        url = req.url.strip()
        if is_social_video_url(url):
            logger.info("Routing %s through the video pipeline (social video host).", url)
            return check_claim(video=url)
        return check_claim(url=url)
    if req.video_url is not None and req.video_url.strip():
        return check_claim(video=req.video_url)
    if req.image is not None and req.image.strip():
        data, suffix = _decode_image(req.image)
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="claimcheck_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            return check_claim(image=tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not remove temp image %s", tmp_path)
    return check_claim(text=req.text)


# The video pipeline needs the ffmpeg *binary* plus a few heavy Python packages.
# When one is missing that's a server-capability problem, not a bad request, so
# it gets a 503 and an actionable install line instead of a wall of traceback.
_FFMPEG_HINT = "Video processing requires ffmpeg. Install it with: winget install ffmpeg"
_VIDEO_PACKAGES = {"yt_dlp": "yt-dlp", "whisper": "openai-whisper", "ffmpeg": "ffmpeg-python"}


def _missing_video_dependency(exc: BaseException) -> str | None:
    """Install hint if `exc` is a missing video dependency, else None."""
    if isinstance(exc, ImportError):
        module = (getattr(exc, "name", "") or "").split(".")[0]
        package = _VIDEO_PACKAGES.get(module)
        if package:
            return (
                f"Video processing requires the '{package}' Python package. "
                f"Install it with: pip install {package}"
            )
        return None
    # `_ensure_ffmpeg()` is the only place that reports the binary as missing.
    if isinstance(exc, VideoExtractionError) and "not found on PATH" in str(exc):
        return _FFMPEG_HINT
    return None


def _safe_extract(req: CheckRequest) -> ClaimExtractionResult:
    """Run extraction, mapping every failure mode to the right HTTP status."""
    try:
        return _extract_from_request(req)
    except (URLExtractionError, ImageExtractionError, VideoExtractionError, ValueError) as exc:
        hint = _missing_video_dependency(exc)
        if hint is not None:
            logger.error("Video dependency missing: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint
            ) from exc
        # Input we couldn't normalize (bad URL, unreadable/undecodable image, etc.).
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ImportError as exc:
        # yt-dlp / whisper / ffmpeg-python are imported lazily inside the video
        # handler, so a missing one surfaces here rather than at import time.
        hint = _missing_video_dependency(exc)
        if hint is not None:
            logger.error("Video dependency missing: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=hint
            ) from exc
        logger.exception("Unexpected import error during extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error during extraction: {exc}",
        ) from exc
    except providers.BudgetWarning as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Daily token budget exhausted: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — surface a clean 500, log the trace
        logger.exception("Unexpected error during extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error during extraction: {exc}",
        ) from exc


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/api/extract", response_model=ClaimExtractionResult)
def extract(req: CheckRequest) -> ClaimExtractionResult:
    """Fast path: normalize input + extract/decompose the claim only.

    Returns the ClaimExtractionResult. A successful call with no health claim is a
    valid 200 result (`claim_found: false`) — reporting that is this endpoint's job.
    """
    _require_exactly_one(req)
    return _safe_extract(req)


@app.post("/api/check", response_model=Dossier)
def check(req: CheckRequest) -> Dossier:
    """Full pipeline: extraction -> retrieval -> verdicts -> dossier."""
    _require_exactly_one(req)
    extraction = _safe_extract(req)

    if not extraction.claim_found:
        # 422 Unprocessable: valid request, but there's no health claim to check.
        # (Integer literal avoids the Starlette constant's rename churn.)
        raise HTTPException(
            status_code=422,
            detail="No evaluable health claim was found in the input.",
        )

    try:
        fact_evidence = retrieve_evidence(extraction.atomic_facts)
        evaluation = evaluate(extraction, fact_evidence)
        return build_dossier(
            original_input=extraction.original_text,
            claim_extraction=extraction,
            verdicts=evaluation.verdicts,
            source_format=extraction.source_format,
            rhetorical_flags=evaluation.rhetorical_flags,
        )
    except providers.BudgetWarning as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Daily token budget exhausted: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during /api/check")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc


@app.get("/")
def root() -> dict[str, object]:
    """Friendly root pointing at the docs and endpoints."""
    return {
        "name": "ClaimCheck AI API",
        "docs": "/docs",
        "endpoints": ["GET /api/health", "POST /api/extract", "POST /api/check"],
    }


# Convenience: `python -m api.server` or `python api/server.py` starts uvicorn.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)
