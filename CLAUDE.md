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

## Current Phase: Weeks 5-6
Focus: Evidence retrieval — the engine that finds evidence for/against each atomic fact. This stage spends ZERO LLM tokens on the actual searching; the lightweight LLM is used ONLY to turn an atomic fact into effective search queries. Components:
- `sources/pubmed.py` — Bio.Entrez client (`search_pubmed`); esearch -> efetch, parses title/abstract/authors/date/PMID/journal, builds an extractive 2-3 sentence summary (no LLM), paces requests to NCBI's rate limit, raises `PubMedError` on failure (empty results are not an error).
- `sources/web_search.py` — Tavily client (`search_web`); re-ranks results so medical/health domains (WHO, CDC, Mayo, Cleveland Clinic, NIH, Cochrane, NHS) float to the top; raises `WebSearchError` if the key is missing.
- `core/source_classifier.py` — `classify_source(url, source_name, publication_types)` -> tier (1-4) + justification, via domain matching + PubMed publication-type detection (meta-analysis/systematic review -> Tier 1, else Tier 2).
- `core/evidence_retriever.py` — orchestrator `retrieve_evidence(atomic_facts)`; per fact: lightweight-LLM query generation (with a token-free fallback to the raw fact text), PubMed (primary) + Tavily (supplementary) search, source classification, ranking that blends lexical relevance with a tier bonus, URL de-dup, top 3-5 kept. Returns `list[FactEvidence]` (each pairs a fact with its ranked `Evidence`). Qdrant vector search slots in here in a later iteration (the 0-1 relevance interface stays the same).
- `schemas/models.py` — `Evidence` gained `summary` + `evidence_stance` (an `EvidenceStance` enum; retrieval leaves it NEUTRAL — the verdict engine assigns supporting/opposing later); added the `FactEvidence` container.
- `main.py` — after extraction, runs retrieval and prints evidence per atomic fact grouped by tier badges `[T1]`–`[T4]`; `--no-evidence` opts out.
- New env vars: `ENTREZ_EMAIL` (required by NCBI), optional `NCBI_API_KEY`.

Completed:
- Weeks 1-2: Claim extractor (atomic-fact decomposition + classification), text + URL input handlers, LLM-agnostic provider abstraction with two-tier routing and soft cost guardrails, Pydantic data models with the claim_found <-> atomic_facts invariant, and an offline test suite.
- Week 3: Screenshot input (`input/image_input.py`) — Claude vision reads the claim from an image while ignoring usernames/hashtags/UI/comments; provider abstraction extended with a provider-neutral `images` parameter; `--image` flag. Live-verified.
- Week 4: Video input (`input/video_input.py`) — video URL -> yt-dlp download -> local Whisper ("base") transcription + vision on evenly-spaced key frames -> de-duplicated merge of spoken + on-screen text; checks for the ffmpeg binary; cleans up temp files; `--video` flag.
