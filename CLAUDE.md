# ClaimCheck AI

## What This Is
A web application that evaluates health claims from social media, blogs, and videos. Not a truth machine — an evidence evaluation engine that decomposes claims, retrieves evidence from trusted medical sources, and produces transparent evidence dossiers.

## Core Design Decisions
- **Combined Approach 2+4**: Claim Decomposition with Per-Atom Verdicts (the engine) wrapped in an Evidence Dossier format (the output). No numerical trust scores.
- **LLM-Agnostic Architecture**: All LLM calls go through a provider abstraction in config/providers.py. Pipeline code never imports anthropic or openai directly. Models are swappable via config.
- **Two-Tier Model Routing**: Cheap model (Haiku/GPT-4o-mini) for claim extraction and classification. Premium model (Sonnet/GPT-4o) for evidence evaluation and verdict generation.
- **Categorical Verdicts**: Strongly Supported, Partially Supported, Insufficient Evidence, Conflicting Evidence, Partially Refuted, Strongly Refuted, Too Vague to Evaluate.
- **Every Dossier gets a UUID** for future shareability.

## Architecture — 5-Stage Pipeline
1. **Input Normalizer** — Accepts text, screenshots (via LLM vision), URLs (via Trafilatura), video links (via yt-dlp + Whisper + vision). All inputs become clean text.
2. **Claim Extractor** — Identifies the primary claim, decomposes into atomic testable facts, classifies each atom (causal, quantitative, prescriptive, temporal).
3. **Evidence Retriever** — Searches curated medical corpus (PubMed, WHO, CDC, Cochrane via Qdrant vector DB) + supplementary web search (Tavily). Zero LLM tokens — all vector search and API calls.
4. **Verdict Engine** — Evaluates each atomic fact against retrieved evidence. Assigns categorical verdicts. Includes rhetorical pattern detection (to be added later as additive module).
5. **Dossier Builder** — Assembles final evidence dossier with per-atom verdict table, cited evidence with source tier badges, rhetorical flags, and narrative summary.

## Source Quality Tiers
- Tier 1: Cochrane systematic reviews, PubMed meta-analyses, WHO guidelines, CDC guidance
- Tier 2: PubMed peer-reviewed studies, NIH MedlinePlus, Mayo Clinic, Cleveland Clinic
- Tier 3: Credentialed health journalism, university health centers
- Tier 4: General web results, blogs, social media (context only, not evidence)

## Tech Stack
- Language: Python
- LLM: Claude API (Sonnet for reasoning, Haiku for lightweight tasks) — swappable via abstraction layer
- RAG: LlamaIndex
- Vector DB: Qdrant
- Medical Sources: PubMed API (Biopython), WHO, CDC, Cochrane
- Web Search: Tavily API
- Web Scraping: Trafilatura
- Video Download: yt-dlp
- Speech-to-Text: OpenAI Whisper (local)
- Frame Extraction: ffmpeg
- Data Validation: Pydantic
- Testing: pytest
- Frontend (later): Next.js + Tailwind CSS

## Token Efficiency Strategy
- Claim decomposition shrinks downstream input by 80-90%
- Evidence retrieval uses zero LLM tokens (vector search + free APIs)
- Only top 3-5 relevant chunks sent to LLM (not 10-15)
- Pre-compressed evidence summaries stored alongside full abstracts
- Structured JSON output eliminates verbose LLM responses
- Two-tier model routing: 60-70% of pipeline runs on cheap model
- Semantic caching for repeated claims (add in iteration phase)

## Project Structure
claimcheck-ai/
├── config/
│   ├── settings.py          # API keys, source tiers, model config
│   └── providers.py         # LLM abstraction layer with cost guardrails
├── input/
│   ├── text_input.py        # Pass-through for pasted text
│   ├── image_input.py       # Screenshot → Vision → text
│   ├── url_input.py         # URL → Trafilatura → text
│   └── video_input.py       # Video → yt-dlp + Whisper + Vision → text
├── core/
│   ├── claim_extractor.py   # Extract + decompose claims into atomic facts
│   ├── evidence_retriever.py
│   ├── source_classifier.py
│   ├── verdict_engine.py
│   └── dossier_builder.py
├── sources/
│   ├── pubmed.py
│   ├── who.py
│   └── web_search.py
├── schemas/
│   └── models.py            # Pydantic models: Claim, AtomicFact, Evidence, Verdict, Dossier (with UUID)
├── tests/
│   └── test_claims.py       # 50+ diverse test cases
├── data/
│   └── test_inputs/
├── main.py                  # Pipeline orchestrator (script for now, FastAPI in weeks 9-10)
└── requirements.txt

## Build Order
- Weeks 1-2: Claim extractor + text/URL input + LLM abstraction + test suite
- Week 3: Screenshot input with vision
- Week 4: Video input with yt-dlp + Whisper
- Weeks 5-6: Evidence retrieval (PubMed + Qdrant + Tavily)
- Weeks 7-8: Verdict engine + dossier builder + rhetorical pattern detection
- Weeks 9-10: Frontend (Next.js) + wrap pipeline in FastAPI
- Weeks 11-12: Testing, edge cases, iteration, semantic caching

## Current Phase: Weeks 9-10
Focus: FastAPI backend + Next.js frontend — wrap the proven pipeline in an HTTP API and a clean consumer UI. The pipeline is untouched; both the CLI and the API call the same functions. Landed:
- `api/server.py` — FastAPI adapter over the SAME functions the CLI uses (no logic duplication). `POST /api/check` (full pipeline -> Dossier JSON), `POST /api/extract` (extraction only -> ClaimExtractionResult JSON, the fast path), `GET /api/health`. Accepts exactly one of `text`/`url`/`image` (base64, data-URL aware)/`video_url`; base64 images are written to a temp file so the path-based image handler is reused unchanged. CORS for localhost:3000. Status mapping: 400 bad input, 422 no evaluable claim (/api/check), 503 budget exhausted, 500 pipeline error. Imports `check_claim` from `main.py`.
- `start_server.py` + `uvicorn api.server:app --reload` — two ways to run the API (http://localhost:8000, interactive docs at /docs). The `main.py` CLI is unchanged and still works.
- `frontend/` — Next.js 16 (App Router, TypeScript, Tailwind v4), light-only professional theme. `app/page.tsx`: minimal search-bar-style input (segmented Text / URL / Screenshot, one field at a time, loading + error states) that POSTs to the API. `app/components/DossierView.tsx`: renders the dossier — primary claim, amber rhetorical-flag cards, color-coded verdict badges (green->red), `[T1]`–`[T4]` tier badges (dark-blue->gray), per-fact reasoning, clickable evidence source URLs, narrative summary, UUID + timestamp shown subtly. `app/lib/{types,api}.ts` mirror the API JSON and wrap `fetch` (base URL via `NEXT_PUBLIC_API_BASE`, default http://localhost:8000).
- `tests/test_api.py` — offline API tests (FastAPI TestClient, pipeline stubbed at the server boundary): health, exactly-one-input (400), no-claim (422), budget (503), pipeline error (500), base64/data-URL decode, Dossier serialization. Live-verified: real `POST /api/extract` returns self-contained atomic facts over HTTP; `npm run build` typechecks clean. requirements.txt gains fastapi, uvicorn[standard], python-multipart, httpx. Dev flow: two terminals — `uvicorn api.server:app --reload` and `cd frontend && npm run dev`.

Completed:
- Weeks 7-8: Verdict engine + dossier builder + rhetorical pattern detection.
  - `core/verdict_engine.py` — `evaluate_claim(claim_extraction, fact_evidence_list)` -> `list[AtomicVerdict]` (spec API) and `evaluate(...)` -> `ClaimEvaluation` (verdicts + rhetorical flags). ONE batched PREMIUM call does the whole job: per-evidence stance (supporting/opposing/neutral — this is where `evidence_stance` gets filled in), tier-weighted + source-diversity reasoning, one of the seven categorical verdicts per fact, a reasoning string, and claim-level rhetorical-flag detection. Verdict/stance enum values feed the output JSON Schema from the models so the schema can't drift. Defensive mapping is keyed off the *input* facts (always one verdict per fact, in order); non-budget LLM errors degrade to Insufficient Evidence; `BudgetWarning` propagates.
  - `core/dossier_builder.py` — `build_dossier(original_input, claim_extraction, verdicts, source_format, rhetorical_flags=..., allow_over_budget=...)` -> `Dossier`; the narrative summary is a second PREMIUM call (plain-language, non-expert), with a deterministic verdict-tally fallback if the call fails or the budget is gone (verdicts are never discarded). `format_dossier(dossier)` renders the terminal dossier: UUID + timestamp, input, primary claim, rhetorical flags, per-atom verdict table with tier badges `[T1]`–`[T4]` and stance markers `(+)/(-)/(.)`, and the narrative.
  - `schemas/models.py` — added `RhetoricalFlag` (pattern/explanation/excerpt); `AtomicVerdict` gained `neutral_evidence` (every retrieved item lands in exactly one of supporting/opposing/neutral, stance-stamped); `Dossier.rhetorical_flags` is now `list[RhetoricalFlag]`.
  - `main.py` — full pipeline end to end: input -> extraction -> retrieval -> verdict evaluation -> dossier, printing the formatted dossier. `--no-evidence` still stops after extraction (debug); `--json` also dumps the dossier as JSON. No-claim inputs still produce a (trivial) dossier for a consistent output shape.
  - Two PREMIUM calls per full run (batched verdicts + narrative); extraction and query-gen stay on the lightweight tier. `tests/test_verdicts.py` covers stance bucketing, flag mapping, defensive fallbacks, and dossier assembly/rendering, all with the LLM stubbed (offline).
- Weeks 5-6: Evidence retrieval — the engine that finds evidence for/against each atomic fact. Spends ZERO LLM tokens on the actual searching; the lightweight LLM is used ONLY to turn an atomic fact into effective search queries.
  - `sources/pubmed.py` — Bio.Entrez client (`search_pubmed`); esearch -> efetch, parses title/abstract/authors/date/PMID/journal, builds an extractive 2-3 sentence summary (no LLM), paces requests to NCBI's rate limit, raises `PubMedError` on failure (empty results are not an error).
  - `sources/web_search.py` — Tavily client (`search_web`); re-ranks results so medical/health domains (WHO, CDC, Mayo, Cleveland Clinic, NIH, Cochrane, NHS) float to the top; raises `WebSearchError` if the key is missing.
  - `core/source_classifier.py` — `classify_source(url, source_name, publication_types)` -> tier (1-4) + justification, via domain matching + PubMed publication-type detection (meta-analysis/systematic review -> Tier 1, else Tier 2).
  - `core/evidence_retriever.py` — orchestrator `retrieve_evidence(atomic_facts)`; per fact: lightweight-LLM query generation (with a token-free fallback to the raw fact text), PubMed (primary) + Tavily (supplementary) search, source classification, ranking that blends lexical relevance with a tier bonus, URL de-dup, top 3-5 kept. Returns `list[FactEvidence]` (each pairs a fact with its ranked `Evidence`). Qdrant vector search slots in here in a later iteration (the 0-1 relevance interface stays the same).
  - `schemas/models.py` — `Evidence` gained `summary` + `evidence_stance` (an `EvidenceStance` enum; retrieval leaves it NEUTRAL — the verdict engine assigns supporting/opposing later); added the `FactEvidence` container.
  - `main.py` — after extraction, runs retrieval and prints evidence per atomic fact grouped by tier badges `[T1]`–`[T4]`; `--no-evidence` opts out.
  - New env vars: `ENTREZ_EMAIL` (required by NCBI), optional `NCBI_API_KEY`.
  - Pre-verdict quality fixes: self-contained atomic facts (each fact carries its own subject/substance/intervention — no bare pronouns) and a relevance floor (`RELEVANCE_FLOOR = 0.15`) applied before ranking so the tier bonus can't rescue topical noise.
- Weeks 1-2: Claim extractor (atomic-fact decomposition + classification), text + URL input handlers, LLM-agnostic provider abstraction with two-tier routing and soft cost guardrails, Pydantic data models with the claim_found <-> atomic_facts invariant, and an offline test suite.
- Week 3: Screenshot input (`input/image_input.py`) — Claude vision reads the claim from an image while ignoring usernames/hashtags/UI/comments; provider abstraction extended with a provider-neutral `images` parameter; `--image` flag. Live-verified.
- Week 4: Video input (`input/video_input.py`) — video URL -> yt-dlp download -> local Whisper ("base") transcription + vision on evenly-spaced key frames -> de-duplicated merge of spoken + on-screen text; checks for the ffmpeg binary; cleans up temp files; `--video` flag.
