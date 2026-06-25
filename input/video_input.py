"""Video input handler (Week 4).

Turns a short-form video link (TikTok, Instagram Reel, YouTube, X/Twitter) into
the clean text the claim extractor expects. Health misinformation in video form
lives in *two* channels at once — what's **spoken** and what's **on screen** — so
this handler captures both and merges them:

    1. Download the video with yt-dlp (reasonable quality, we want audio not 4K).
    2. Extract the audio track and transcribe it with local OpenAI Whisper.
    3. Grab a few evenly-spaced key frames and read on-screen text with vision.
    4. Merge transcription + on-screen text, de-duplicating overlap.

Heavy/optional dependencies (yt-dlp, whisper, ffmpeg-python) are imported lazily
so the rest of the package imports cleanly without them. The ffmpeg *binary* must
be installed separately; we check for it up front and fail with instructions.
"""

from __future__ import annotations

import base64
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from config import providers
from input.text_input import clean_text

logger = logging.getLogger("claimcheck.video")

# Tunables (kept module-level so tests/config can see them).
DEFAULT_WHISPER_MODEL = "base"   # ~140MB; upgrade to "medium"/"large" for accuracy
DEFAULT_NUM_FRAMES = 4           # 3-5 evenly-spaced frames for on-screen text
_NO_TEXT_SENTINEL = "NO_TEXT"    # what the vision model returns for a frame with no overlay


class VideoExtractionError(Exception):
    """Raised when a video can't be downloaded, processed, or yields no text."""


@dataclass
class VideoExtraction:
    """Result of processing a video — both channels plus the merged text."""

    transcription: str   # spoken words (Whisper)
    visual_text: str     # de-duplicated on-screen text (vision over key frames)
    combined_text: str   # merged, cleaned text for the claim extractor


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_FRAME_SYSTEM_PROMPT = f"""\
You are reading a single frame from a social-media video.

Return ONLY the text that is overlaid on the frame — titles, captions, and
on-screen words (e.g. "5 FOODS THAT CURE INFLAMMATION"). Transcribe it verbatim.

IGNORE: watermarks, app logos, usernames/handles, follower or view counts,
progress bars, and other UI chrome. Do not describe the image, do not add
commentary, do not fact-check.

If the frame has no readable overlaid text, respond with exactly: {_NO_TEXT_SENTINEL}
"""

_FRAME_USER_PROMPT = "Read the on-screen text in this video frame."


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #
def _ensure_ffmpeg() -> None:
    """Verify the ffmpeg binary is on PATH; raise with install help if not."""
    if shutil.which("ffmpeg") is None:
        raise VideoExtractionError(
            "ffmpeg is required for video processing but was not found on PATH.\n"
            "  Windows:  winget install ffmpeg   (or download from https://ffmpeg.org/download.html)\n"
            "  macOS:    brew install ffmpeg\n"
            "  Linux:    sudo apt install ffmpeg"
        )


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# --------------------------------------------------------------------------- #
# Pipeline steps (each is small + individually mockable for tests)
# --------------------------------------------------------------------------- #
def _download_video(url: str, tmpdir: str) -> str:
    """Download `url` into `tmpdir` and return the path to the video file."""
    import yt_dlp  # noqa: PLC0415

    ydl_opts = {
        # Cap at 480p — we only need audio + legible on-screen text, not 4K.
        "format": "bv*[height<=480]+ba/b[height<=480]/b",
        "outtmpl": str(Path(tmpdir) / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise VideoExtractionError(
            f"Could not download video from {url!r} (unsupported platform, private, "
            f"or network error): {exc}"
        ) from exc

    files = [f for f in Path(tmpdir).iterdir() if f.is_file()]
    if not files:
        raise VideoExtractionError(f"No video file was downloaded from {url!r}.")
    # The merged video is the largest artifact in the temp dir.
    return str(max(files, key=lambda p: p.stat().st_size))


def _has_audio(video_path: str) -> bool:
    import ffmpeg  # noqa: PLC0415

    try:
        info = ffmpeg.probe(video_path)
    except Exception:  # ffmpeg.Error or anything probe throws
        return False
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def _probe_duration(video_path: str) -> float:
    import ffmpeg  # noqa: PLC0415

    try:
        info = ffmpeg.probe(video_path)
        return float(info["format"]["duration"])
    except Exception as exc:
        raise VideoExtractionError(f"Could not read video duration: {exc}") from exc


def _extract_audio(video_path: str, tmpdir: str) -> str:
    """Extract a 16kHz mono WAV (what Whisper wants) and return its path."""
    import ffmpeg  # noqa: PLC0415

    out = str(Path(tmpdir) / "audio.wav")
    try:
        (
            ffmpeg.input(video_path)
            .output(out, ac=1, ar=16000, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as exc:
        raise VideoExtractionError(f"Failed to extract audio: {exc}") from exc
    return out


def _transcribe(audio_path: str, model_name: str = DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe `audio_path` with local Whisper. Downloads the model on first run."""
    import whisper  # noqa: PLC0415

    # The model is cached after the first download (~/.cache/whisper).
    print(
        f"[video] Loading local Whisper '{model_name}' model "
        f"(first run downloads it, ~140MB for 'base')...",
        file=sys.stderr,
    )
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
    except Exception as exc:
        raise VideoExtractionError(f"Whisper transcription failed: {exc}") from exc
    return str(result.get("text", "")).strip()


def _extract_frames(video_path: str, tmpdir: str, duration: float, n: int) -> list[str]:
    """Grab `n` evenly-spaced frames (interior points) and return their paths."""
    import ffmpeg  # noqa: PLC0415

    frames: list[str] = []
    for i in range(n):
        # Interior points at (i+1)/(n+1) of the duration — avoids black
        # first/last frames common in short-form video.
        ts = duration * (i + 1) / (n + 1)
        out = str(Path(tmpdir) / f"frame_{i}.png")
        try:
            (
                ffmpeg.input(video_path, ss=ts)
                .output(out, vframes=1)
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as exc:
            logger.warning("Frame extraction at %.1fs failed: %s", ts, exc)
            continue
        if Path(out).exists():
            frames.append(out)
    return frames


def _read_frame_text(frame_path: str, allow_over_budget: bool = False) -> str:
    """Read on-screen text from one frame via vision. Returns "" if none."""
    encoded = base64.standard_b64encode(Path(frame_path).read_bytes()).decode("utf-8")
    result = providers.llm_call(
        prompt=_FRAME_USER_PROMPT,
        system_prompt=_FRAME_SYSTEM_PROMPT,
        model_tier="premium",
        images=[{"media_type": "image/png", "data": encoded}],
        allow_over_budget=allow_over_budget,
    )
    text = result.strip() if isinstance(result, str) else str(result).strip()
    return "" if not text or text == _NO_TEXT_SENTINEL else text


def _merge_texts(transcription: str, visual_texts: list[str]) -> tuple[str, str]:
    """Merge spoken + on-screen text, de-duplicating overlap.

    Returns (visual_text, combined_text):
      * visual_text  — on-screen lines, de-duplicated across frames (frames often
        share the same overlay).
      * combined_text — transcription plus any on-screen line not already spoken.
    """
    # De-dupe on-screen lines across frames, case-insensitively, preserving order.
    seen: set[str] = set()
    unique_visual: list[str] = []
    for vt in visual_texts:
        for raw in vt.splitlines():
            line = raw.strip()
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_visual.append(line)
    visual_text = "\n".join(unique_visual)

    # Combined: spoken text, then on-screen lines that weren't already spoken.
    norm_transcript = transcription.lower()
    extra = [line for line in unique_visual if line.lower() not in norm_transcript]

    parts: list[str] = []
    if transcription.strip():
        parts.append(transcription.strip())
    parts.extend(extra)
    combined = "\n".join(parts).strip()

    return visual_text, combined


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def extract_from_video(
    url: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    allow_over_budget: bool = False,
) -> VideoExtraction:
    """Download, transcribe, read on-screen text, and merge — for one video URL.

    Args:
        url: A video URL (TikTok, Instagram, YouTube, X/Twitter, etc.).
        num_frames: How many evenly-spaced frames to read for on-screen text.
        whisper_model: Whisper model size ("base" default; "medium"/"large" later).
        allow_over_budget: Forwarded to the LLM layer's soft budget guardrail.

    Returns:
        A VideoExtraction with the transcription, on-screen text, and merged text.

    Raises:
        VideoExtractionError: for invalid URLs, missing ffmpeg, download/transcribe
            failures, or a video with neither speech nor on-screen text.
        providers.BudgetWarning: if the daily token budget is exhausted (per frame).
    """
    if not isinstance(url, str) or not _looks_like_url(url):
        raise VideoExtractionError(f"Not a valid video URL: {url!r}")

    _ensure_ffmpeg()

    tmpdir = tempfile.mkdtemp(prefix="claimcheck_video_")
    try:
        video_path = _download_video(url, tmpdir)

        # --- audio → transcription (graceful if the video has no audio track) ---
        if _has_audio(video_path):
            audio_path = _extract_audio(video_path, tmpdir)
            transcription = _transcribe(audio_path, whisper_model)
        else:
            logger.warning("Video %r has no audio track; using on-screen text only.", url)
            transcription = ""

        # --- key frames → on-screen text ---
        duration = _probe_duration(video_path)
        frames = _extract_frames(video_path, tmpdir, duration, num_frames)
        visual_texts = [_read_frame_text(f, allow_over_budget=allow_over_budget) for f in frames]

        # --- merge ---
        visual_text, combined = _merge_texts(transcription, visual_texts)
        if not combined.strip():
            raise VideoExtractionError(
                f"No speech or on-screen text could be extracted from {url!r}."
            )

        return VideoExtraction(
            transcription=transcription,
            visual_text=visual_text,
            combined_text=clean_text(combined),
        )
    finally:
        # Always clean up downloaded video/audio/frames.
        shutil.rmtree(tmpdir, ignore_errors=True)
