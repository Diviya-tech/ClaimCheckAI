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

## Build Order (all complete)
- [x] Weeks 1-2: Claim extractor + text/URL input + LLM abstraction + test suite
- [x] Week 3: Screenshot input with vision
- [x] Week 4: Video input with yt-dlp + Whisper
- [x] Weeks 5-6: Evidence retrieval (PubMed + Tavily; Qdrant deferred to a later iteration)
- [x] Weeks 7-8: Verdict engine + dossier builder + rhetorical pattern detection
- [x] Weeks 9-10: Frontend (Next.js) + wrap pipeline in FastAPI
- [x] Weeks 11-12: Testing, error hardening, frontend polish, in-memory cache, docs

## Current Phase: COMPLETE — all 12 weeks landed
The full build plan is done. What remains are unscheduled later iterations (Qdrant vector corpus behind the existing 0-1 relevance interface, semantic caching of paraphrased claims, a persistent dossier store for UUID sharing). Reasoning-integrity guardrails, `.edu` spam demotion, and the weeks 11-12 hardening are all live; see `docs/DECISIONS.md` ADR-015..019.

## Project Structure additions since the plan
- `core/cache.py` — bounded, thread-safe LRU `DossierCache` keyed on normalized claim text; `dossier_cache` is the process-wide instance the API uses.
- `tests/test_claim_corpus.py`, `tests/test_hardening.py` — weeks 11-12 suites (see below).
- `main.py` now owns the stage sequence for BOTH the CLI and the API: `check_claim()` (stages 1-2), `complete_dossier()` (stages 3-5, degrades into `Dossier.limitations`), `run_pipeline()` (all five with the cache in front), `stage_timer()` (per-stage timing logs on `claimcheck.pipeline`).

Completed:
- Weeks 11-12: Testing, error hardening, frontend polish, cache, docs.
  - **Test suite: 230 offline tests** (was 140). `tests/test_claim_corpus.py` covers ambiguous, multi-claim, no-claim, sarcastic, vague, well-supported, form-mismatch, association-vs-causation and wrong-population claims (LLM stubbed; each case locks the pipeline behavior it exercises) plus empty, extremely long, special-character, zero-width/control-character and non-English inputs. `tests/test_hardening.py` covers every degradation path, the malformed-output retry contract, budget exhaustion at each stage, user-readable errors (API + CLI), the cache, stage timing, and the reasoning-integrity guardrails. Prompt regressions are caught by asserting the instructions are still present.
  - **Reasoning integrity (implemented this phase — the earlier commit with that name did not actually contain them):**
    - *Animal-study filter* — `sources/pubmed.py` parses MeSH terms; `is_animal_study()` = `Animals` without `Humans` (keyword fallback for in-vitro / un-indexed). Retrieval stamps `Evidence.applicability = non_human`, ranks such items below human evidence (`_NON_HUMAN_PENALTY`), caps them at one per fact, excludes them from the sufficiency count; the verdict engine forces their stance to `neutral` in code.
    - *Evidence applicability* — new `EvidenceApplicability` enum (`direct` / `indirect` / `non_human` / `unknown`) on `Evidence`; the eval schema requires `applicability` per evidence item; `_apply_applicability_cap()` downgrades `Strongly Supported/Refuted` to `Partially` unless ≥1 `direct` item points that way, annotating the reasoning.
    - *Granular form decomposition* — extractor prompt: a named preparation (tea, water, juice, powder, …) gets its OWN atomic fact ("turmeric tea delivers curcumin at a dose comparable to the trials") separate from the effect.
    - *Evidence-bound rhetoric* — `_assemble_flags(raw, original_text)` drops any flag whose `excerpt` isn't found verbatim (case/curly-quote/whitespace-insensitive) in the claim text; empty excerpts are dropped too.
  - **Error hardening:** `FactEvidence.source_notes` + `Dossier.limitations` are now populated. PubMed down for every query → web-only + note; `TAVILY_API_KEY` missing → PubMed-only + "skipped" note (distinct from "down"); both down → single note. `providers.llm_call` retries a malformed structured response exactly once then raises `MalformedOutputError` (user-readable; API → `502`, CLI → error line; verdict stage → Insufficient; narrative → tally + note). Budget exhausted at the verdict stage → `fallback_verdicts()` with evidence attached unassessed + `NOTE_VERDICT_BUDGET`; at the narrative stage → tally + `NOTE_NARRATIVE_BUDGET`; before extraction → still `503`. `@app.exception_handler(Exception)` guarantees `{"detail": "<sentence>"}` for anything unanticipated. `MAX_TEXT_CHARS = 50_000` → `413`. `api/server.py` calls `logging.basicConfig(INFO)` so pipeline logs reach the uvicorn console.
  - **Frontend:** staged loading indicator (Extracting claim… → Searching evidence… → Evaluating verdicts… → Building dossier…, advanced on wall-clock estimates since the API is one request), four clickable example claims that fill the textbox, "Try another claim" (header + footer of the dossier; resets, scrolls up, refocuses), "Limitations of this run" panel, `Indirect` / `Animal / in-vitro` applicability badges on evidence rows, verdict-card headers stack below `sm`, `break-words`/`min-w-0` throughout for phone width, footer "Evidence sourced from PubMed, WHO, CDC. ClaimCheck AI does not provide medical advice." `types.ts` gains `applicability` and `limitations`. `npm run build` + `eslint` clean. (Narrow-screen layout verified by code review + build only — the Chrome extension couldn't render localhost in the session; worth one manual look at ~400px.)
  - **Cache + timing:** exact same text (after `clean_text` normalization) returns the cached dossier (same UUID, zero tokens); no-claim results and URL/image/video inputs are never cached. Per-stage timing logged as `[timing] <stage> <s>` (`--verbose` on the CLI).
  - **Docs:** README (roadmap all ✅, 230 tests, reasoning rules + guardrail table, status codes, degradation, cache), ARCHITECTURE.md (orchestrator functions, real retriever, guardrail table, malformed retry, §10 Graceful Degradation, §11 Cache & Timing, testing strategy), DECISIONS.md ADR-015..019.
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
