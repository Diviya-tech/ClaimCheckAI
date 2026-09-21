# ClaimCheck AI — Architecture

A deep technical walkthrough of how ClaimCheck AI is built and why the pieces fit
together the way they do. For the *decision rationale* see [`DECISIONS.md`](DECISIONS.md);
for the *research basis* see [`RESEARCH.md`](RESEARCH.md).

---

## 1. System Overview

ClaimCheck is a linear, five-stage pipeline. Each stage has a single
responsibility and a typed input/output contract, so any stage can be developed,
tested, or swapped in isolation.

```
 raw input (text / image / URL / video)
        │
        ▼
 ┌─────────────────────┐
 │ 1. Input Normalizer │  → clean text (str)
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ 2. Claim Extractor  │  → ClaimExtractionResult
 └─────────────────────┘     (primary_claim + list[AtomicFact])
        │
        ▼
 ┌─────────────────────┐
 │ 3. Evidence         │  → list[FactEvidence]
 │    Retriever        │     (0 LLM tokens)
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ 4. Verdict Engine   │  → list[AtomicVerdict]
 └─────────────────────┘
        │
        ▼
 ┌─────────────────────┐
 │ 5. Dossier Builder  │  → Dossier (UUID-stamped)
 └─────────────────────┘
```

**Data flow is one-directional.** Each stage consumes the previous stage's typed
output and produces the next. The typed contracts are the Pydantic models in
[`schemas/models.py`](../schemas/models.py):

```
str → ClaimExtractionResult → FactEvidence[] → AtomicVerdict[] → Dossier
```

The orchestrator ([`main.py`](../main.py)) wires the stages together. **All five
stages run today** — `python main.py --text "…"` takes a claim through
extraction, retrieval, evaluation, and dossier assembly, and prints the finished
dossier. A `FactEvidence` pairs each atomic fact with the ranked evidence
retrieved for it, so stage 4 can evaluate every fact against exactly its own
evidence.

The orchestrator exposes two functions that both the CLI and the HTTP API call,
so the stage sequence exists in exactly one place:

| Function | Stages | Notes |
|----------|--------|-------|
| `check_claim(text=…/url=…/image=…/video=…)` | 1–2 | Input normalization + extraction. Errors here propagate (nothing has been computed yet). |
| `complete_dossier(extraction)` | 3–5 | Retrieval → verdicts → dossier. **Degrades, never fails**: a dead source or a budget cut becomes an entry in `Dossier.limitations` (§10). |
| `run_pipeline(…)` | 1–5 | The two above with the dossier cache (§11) in front. |

Every stage runs inside `stage_timer(...)`, which logs `[timing] <stage> <s>` to
the `claimcheck.pipeline` logger (visible on the API server console, or with
`main.py --verbose`).

---

## 2. Input Normalizer (Stage 1)

**Goal:** turn any of four input modalities into the *same* clean text, so every
downstream stage is input-agnostic.

| Modality | Handler | Technique |
|----------|---------|-----------|
| Text | `input/text_input.py` | Unicode NFKC normalization, line-ending unification, whitespace/blank-line collapsing. |
| URL | `input/url_input.py` | Trafilatura fetch + body extraction (drops nav/ads/comments), then run through the text cleaner. |
| Screenshot | `input/image_input.py` *(wk 3)* | LLM vision reads claim text from the image. |
| Video | `input/video_input.py` *(wk 4)* | yt-dlp download → Whisper transcription (audio) + vision (on-screen text). |

**Design principles:**

- **Single normalization path.** Both the text and URL handlers funnel through
  the same `clean_text()` function, so the output is byte-consistent regardless
  of source. A screenshot and a pasted sentence produce identical-shape input.
- **Fail loudly, fail clean.** `clean_text()` raises `ValueError` on empty input;
  the URL handler raises a domain-specific `URLExtractionError` for invalid URLs,
  failed fetches, or empty extractions — never a raw stack trace into the pipeline.
- **Lazy imports.** Heavy/optional dependencies (Trafilatura) are imported inside
  the handler, so the package imports cleanly even in a minimal environment.

```python
# Both paths converge on the same cleaner:
text  → clean_text(raw)                    → clean str
url   → trafilatura.extract(...) → clean_text(...) → clean str
```

---

## 3. Claim Extractor (Stage 2)

**Goal:** find the single primary health claim and decompose it into atomic,
independently testable facts — each classified by type.

### Why decomposition

A compound claim cannot receive a single honest verdict. "Green tea boosts
metabolism *and* melts belly fat *in two weeks*" is three assertions; collapsing
them loses information the evidence stage needs. Decomposition also shrinks
downstream token usage by 80–90% — we reason over short atoms, not raw text.

### Form & preparation get their own atom

A claim about a *food or drink* is not the same claim as one about a
*concentrated extract* of it, but the evidence base almost always studies the
extract. So when a claim names a preparation — tea, water, juice, powder, gummy,
topical — the extractor emits a **separate atomic fact for the form** alongside
the effect:

```
"Turmeric tea cures arthritis"
  → "Turmeric (curcumin) reduces arthritis symptoms."                         (causal)
  → "Turmeric tea delivers curcumin at a dose comparable to the amounts
     studied in clinical trials."                                            (quantitative)
```

The verdict engine can then say the effect is *Partially Supported* while the
form is *Strongly Refuted* — the honest answer — instead of blurring the two into
one verdict (ADR-017).

### Fact classification

Each atomic fact is one of five types, which later guides how evidence is sought:

| Type | Asserts | Example |
|------|---------|---------|
| `causal` | X causes / prevents / cures Y | "Green tea melts belly fat." |
| `quantitative` | a number, dose, rate, % | "EGCG is up to 80% of green tea's catechins." |
| `prescriptive` | do / take / avoid something | "Take 5000 IU of vitamin D daily." |
| `temporal` | timing, duration, sequence | "Results appear within two weeks." |
| `existential` | something exists / is present | "Green tea contains antioxidants." |

### Structured output schema

The extractor uses the **lightweight** model tier and forces structured output
via tool_use (see §6). The model fills in this JSON Schema, which maps onto
`ClaimExtractionResult` minus the fields we own (`original_text`, `source_format`):

```jsonc
{
  "type": "object",
  "properties": {
    "claim_found":   { "type": "boolean" },
    "primary_claim": { "type": "string" },
    "atomic_facts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text":             { "type": "string" },
          "fact_type":        { "enum": ["causal","quantitative",
                                         "prescriptive","temporal","existential"] },
          "original_context": { "type": "string" }
        },
        "required": ["text", "fact_type", "original_context"],
        "additionalProperties": false
      }
    }
  },
  "required": ["claim_found", "primary_claim", "atomic_facts"],
  "additionalProperties": false
}
```

### Enforced invariant

`claim_found` and `atomic_facts` must agree. A found claim **must** decompose
into at least one atomic fact; a not-found claim **must** carry none. This is
enforced at the *model layer* (a Pydantic `@model_validator`), so it holds no
matter who constructs the result — the extractor, a test, or a future API
handler. A prompt regression that returns `claim_found=True` with zero atoms is
rejected at construction time rather than flowing downstream silently.

```python
@model_validator(mode="after")
def _check_claim_atom_invariant(self):
    if self.claim_found and not self.atomic_facts:
        raise ValueError("claim_found is True but atomic_facts is empty …")
    if not self.claim_found and self.atomic_facts:
        raise ValueError("claim_found is False but atomic_facts is non-empty …")
    return self
```

**Defense in depth:** the *prompt* asks the model to comply (and to strip hedging
from long-form sources while extracting the underlying assertion); the
*validator* guarantees a non-compliant result can't be accepted.

---

## 4. Evidence Retriever (Stage 3) *(weeks 5–6)*

**Goal:** for each atomic fact, retrieve the most relevant, highest-quality
evidence — spending **zero LLM tokens** on the search itself.

- **PubMed** (Bio.Entrez; primary): esearch → efetch, parsed into
  title/abstract/authors/date/publication types/**MeSH terms**, with an
  extractive 2–3 sentence summary (no LLM) stored alongside the full abstract.
- **Supplementary web search** (**Tavily**), re-ranked so WHO/CDC/Mayo/NIH/
  Cochrane/NHS hits float to the top. Web hits that point at a PubMed/PMC
  article are reconciled with the PubMed API record instead of trusted as
  scraped.
- **Query generation** is the one lightweight-LLM call in this stage (1–3
  literature-style queries per fact), with a token-free fallback to the raw fact
  text so retrieval never blocks on the LLM.
- **Relevance floor** (`0.15`) applied *before* ranking so the tier bonus can
  order relevant results but never rescue irrelevant ones; then top 3–5 per fact.
- **Animal / in-vitro studies** are detected token-free — MeSH `Animals` without
  `Humans`, or a keyword fallback ("in mice", "cell line", …) — stamped
  `applicability = non_human`, ranked below human evidence with a flat penalty,
  capped at one per fact, and excluded from the "enough evidence to evaluate"
  count (ADR-015).
- **Source notes.** A source that is down or unconfigured is a *note*, not an
  error: `FactEvidence.source_notes` records "PubMed was unavailable; evidence
  came from web sources only", "web search skipped (no `TAVILY_API_KEY`)", etc.,
  and `collect_limitations()` lifts them into `Dossier.limitations` (§10).

The Qdrant vector corpus described in the original plan slots in behind the same
0–1 relevance interface in a later iteration; today's relevance for PubMed is a
lexical proxy and for Tavily its own score. Tokens are reserved for *reasoning*,
which is stage 4.

### Source hierarchy (Tier 1–4)

Every retrieved source is tagged with a quality tier. Verdicts weigh higher tiers
more heavily, so a meta-analysis can't be outvoted by SEO blogs.

| Tier | Definition | Examples |
|------|------------|----------|
| **Tier 1** | Systematic reviews & official guidelines | Cochrane reviews, PubMed meta-analyses, WHO guidelines, CDC guidance |
| **Tier 2** | Peer-reviewed studies & major institutions | PubMed primary studies, NIH MedlinePlus, Mayo Clinic, Cleveland Clinic |
| **Tier 3** | Credentialed health journalism & academia | University health centers, credentialed health journalism |
| **Tier 4** | General web (context only, *not* evidence) | Blogs, social media, general web results |

Tier 4 is captured for context but never treated as evidence for a verdict.

**`.edu` is not automatically Tier 3.** University domains are routinely hijacked
(abandoned student pages, compromised uploads, open redirects) to host SEO spam.
A `.edu` URL whose *path or query* shows an injected external URL, marketing /
affiliate parameters (`utm_*`, `ref`, `affiliate`, `promo`, …), or a redirect
parameter or segment (`?url=`, `/go/`, `/redirect/`, …) is demoted to Tier 4
with a justification naming the signal. The host itself is never inspected, so
`pharmacy.online.uni.edu` is unaffected (ADR-019).

---

## 5. Verdict Engine (Stage 4)

**Goal:** evaluate each atomic fact against *its own* retrieved evidence and
assign one of seven categorical verdicts, with transparent, evidence-citing
reasoning — plus detect rhetorical manipulation in the original claim.

### One batched premium call

Every atomic fact, all of its evidence, and the claim-level rhetorical analysis
go to the **premium** tier in a **single batched call** (`evaluate(...)`). This is
a deliberate token move: one call amortizes the system prompt and shared claim
context across all atoms instead of paying per-fact. The call is forced through
structured output (§6), and its `enum` values for both the verdict and the
evidence stance are generated *from the Pydantic enums* so the schema can never
drift from the models.

`evaluate_claim(...)` is the spec'd entry point returning `list[AtomicVerdict]`;
`evaluate(...)` returns a richer `ClaimEvaluation` (verdicts **plus** rhetorical
flags) so the dossier builder gets the flags without a second call.

### What it produces per fact

1. **Evidence stance.** Every retrieved item is classified `supporting`,
   `opposing`, or `neutral` relative to the fact — this is where the `evidence_stance`
   placeholder that retrieval left `NEUTRAL` finally gets filled in. Each item
   lands in exactly one of three buckets on the `AtomicVerdict`
   (`supporting_evidence` / `opposing_evidence` / `neutral_evidence`), so the
   dossier can cite every source with how it bore on the verdict.
1b. **Evidence applicability.** Separately from *which way* an item points, the
   model rates *how much it can count*: `direct` (same intervention and form,
   human population the claim is aimed at, the asserted outcome) or `indirect`
   (related, but a different form / dose / population / proxy outcome).
   Retrieval's `non_human` stamp is authoritative and cannot be overridden.
2. **Tier-weighted, diversity-aware reasoning.** The prompt instructs the model
   to weigh higher tiers more (a single T1 meta-analysis outweighs a stack of
   T3/T4), to treat T4 as context only, and to note whether independent
   institutions *converge* (stronger) or a claim rests on a lone source (weaker).
3. **A categorical verdict** — one of the seven below.
4. **A reasoning string** citing specific evidence by source and tier.

| Verdict | Meaning |
|---------|---------|
| `Strongly Supported` | High-tier evidence consistently supports the fact. |
| `Partially Supported` | Some support, with caveats or limited evidence. |
| `Insufficient Evidence` | Not enough quality evidence to judge. |
| `Conflicting Evidence` | Credible evidence points both ways (or cherry-picking). |
| `Partially Refuted` | Some credible evidence contradicts the fact. |
| `Strongly Refuted` | High-tier evidence consistently contradicts the fact. |
| `Too Vague to Evaluate` | The fact isn't specific enough to test. |

### Rhetorical red-flag detection

In the *same* batched call, the engine inspects the **original claim text** for
manipulation patterns — guaranteed/absolute outcomes, conspiracy framing
("doctors don't want you to know"), appeal to nature, anecdote-as-proof, false
urgency, and emotional manipulation — returning a list of structured
`RhetoricalFlag`s (pattern + explanation + the triggering excerpt). Flags describe
**how the claim is argued, not whether it is true**; they are surfaced in the
dossier as context and never alter the evidence verdict.

### Reasoning-integrity guardrails (enforced in code)

The prompt carries six scientific-reasoning rules (absence of evidence ≠
refutation; trial duration ≠ required duration; significant ≠ meaningful;
association ≠ causation; population X ≠ everyone; extract ≠ food). Three of them
are also enforced *deterministically* after the model answers, so an over-read
abstract can't become an over-stated verdict:

| Guardrail | Rule in code | ADR |
|-----------|--------------|-----|
| **Animal-study filter** | Any item retrieval stamped `non_human` is forced to stance `neutral`, whatever the model said. The prompt line is tagged `[NON-HUMAN STUDY]` so the reasoning can say so. | ADR-015 |
| **Applicability cap** | `Strongly Supported` needs ≥1 `direct` supporting item; `Strongly Refuted` needs ≥1 `direct` opposing item. Otherwise the verdict is downgraded to the `Partially` form and the reasoning is annotated with why. Unrated evidence counts as `indirect`, never `direct`. | ADR-016 |
| **Evidence-bound rhetoric** | A `RhetoricalFlag` is kept only if its `excerpt` appears verbatim in the claim text (case-, curly-quote- and whitespace-insensitive). Empty, paraphrased or invented excerpts are dropped. | ADR-018 |

### Defensive by construction

The mapping from model output back to typed models is driven off the *input*
facts, not the model's output, so the result **always has exactly one verdict per
fact, in order**, even if the model skips, reorders, or duplicates entries — a
missing fact defaults to `Insufficient Evidence` with its evidence preserved as
context. Unknown verdict/stance strings coerce to safe defaults. A non-budget LLM
error (including a `MalformedOutputError` after the provider's own retry)
degrades the whole stage to `Insufficient Evidence` rather than crashing the run;
`BudgetWarning` alone propagates to the orchestrator, which converts it into the
same Insufficient-Evidence fallback *plus* a limitation note (§10), so the
evidence already retrieved is still shown.

---

## 5b. Dossier Builder (Stage 5)

Assembles the final `Dossier`: the original input, a per-atom verdict table, the
cited evidence with tier badges and stance markers, the rhetorical flags, and a
plain-language **narrative summary**. **Every dossier gets a UUID and a UTC
timestamp** at construction (`default_factory`) so it can be shared and referenced
later.

- **Narrative summary** is a second **premium** call — the one place the dossier
  builder reasons — turning the decided verdicts into 3–6 sentences a non-expert
  can follow, faithful to the individual verdicts (it won't upgrade *Insufficient
  Evidence* into a clean yes/no).
- **Graceful degradation:** if that call fails or the budget is exhausted, the
  builder falls back to a deterministic summary assembled from the verdict tally
  and appends a note to `Dossier.limitations` saying so. The verdicts — the
  expensive part — are never discarded, so a dossier is *always* produced.
- **`format_dossier(dossier)`** renders the whole thing as clean terminal text:
  tier badges `[T1]`–`[T4]`, stance markers `(+)` / `(-)` / `(.)`, a
  per-verdict cue, and a closing "LIMITATIONS OF THIS RUN" block when there is
  anything to say. `main.py --json` additionally emits the dossier as JSON.

This split — analysis in stage 4, presentation + summary in stage 5 — keeps the
verdict logic independent of how it's rendered, and means two premium calls per
full run (batched verdicts, then narrative).

---

## 6. LLM Abstraction Layer

The single most important architectural boundary in the system.

**Rule:** pipeline code never imports `anthropic` or `openai`. It calls one
function:

```python
from config import providers

result = providers.llm_call(
    prompt="…",
    system_prompt="…",
    model_tier="lightweight",        # or "premium"
    response_schema=SCHEMA,          # optional → structured dict via tool_use
)
```

### How it routes

```
llm_call(model_tier=...)
   │
   ├─ look up MODEL_CONFIG[tier] → {provider, model, max_tokens}
   ├─ check soft token budget (raise BudgetWarning if exhausted)
   ├─ get_provider(provider) → cached LLMProvider instance
   ├─ provider.complete(...) → (content, input_tokens, output_tokens)
   ├─ record usage → daily_usage.json
   └─ return content   (str, or dict when response_schema is given)
```

`MODEL_CONFIG` (in `config/settings.py`) is the only place model names live:

```python
MODEL_CONFIG = {
    "premium":     {"provider": "anthropic", "model": "claude-sonnet-4-6",          "max_tokens": 4096},
    "lightweight": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",  "max_tokens": 2048},
}
```

### Structured output via tool_use

When `response_schema` is provided, the Anthropic provider defines a single
synthetic tool whose `input_schema` *is* the caller's schema, and forces
`tool_choice` to it. The model's tool-call `input` is therefore guaranteed to
match the schema; the provider returns it as a validated dict. No prose parsing,
no regex, no deprecated `output_format`.

**Malformed output is retried once, then surfaced clearly.** "Guaranteed" is not
"always": a model can answer in prose instead of calling the tool, or omit a
required key. `llm_call` checks the shape (`_check_structured`), records the
tokens the failed attempt cost, and retries **exactly once** (a third attempt
would only burn tokens confirming the model can't satisfy the schema right now).
A second failure raises `MalformedOutputError` with a user-readable message
("…returned an unusable response 2 times in a row. This is usually transient —
please try again"), which the API maps to `502` and the CLI prints as a single
line. Free-text calls have nothing to malform and run once.

### Adding a new provider

The abstraction is an ABC with one method. To add OpenAI (or any backend):

1. Implement `LLMProvider.complete(...)`, returning
   `(content, input_tokens, output_tokens)`. For structured output, translate the
   `response_schema` into that provider's native mechanism (e.g. OpenAI function
   calling / JSON mode).
2. Register it: `_PROVIDER_FACTORIES["openai"] = OpenAIProvider`.
3. Point a tier at it in `MODEL_CONFIG` (e.g. `"provider": "openai"`).

**Zero pipeline code changes.** The extractor, retriever, and verdict engine
never know which backend answered.

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, *, prompt, system_prompt, model,
                 max_tokens, response_schema) -> tuple[str | dict, int, int]:
        ...
```

---

## 7. Token Efficiency Strategy

Seven moves, each a structural choice rather than a tweak. Target: **~2,000–3,000
tokens/check** vs. ~8,000–12,000 for a monolithic checker.

1. **Claim decomposition** — reason over atomic facts, not raw input. Shrinks
   downstream input 80–90%.
2. **Zero-token retrieval** — vector search + free APIs, no LLM in the loop for
   lookup (stage 3 spends 0 tokens).
3. **Relevance filtering** — top 3–5 evidence chunks per atom, not 10–15.
4. **Pre-compressed summaries** — send the stored summary, keep the full abstract
   in reserve.
5. **Structured JSON output** — schemas eliminate verbose prose responses.
6. **Two-tier routing** — extraction/classification on the cheap model; only
   evidence evaluation uses premium. 60–70% of work runs cheap.
7. **Semantic caching** *(later)* — repeated/similar claims reuse prior results.

Stage-by-stage token profile:

| Stage | LLM tier | Token cost |
|-------|----------|-----------|
| 1. Input Normalizer | none (vision only for images) | ~0 (text/URL) |
| 2. Claim Extractor | lightweight | low |
| 3. Evidence Retriever | lightweight (query-gen only) → **none** for search | very low; search is **0** |
| 4. Verdict Engine | premium (one batched call) | the bulk of spend |
| 5. Dossier Builder | premium (narrative only) | moderate |

The premium tier touches exactly two stages — evaluating the evidence and writing
the summary — the two places where reasoning depth actually changes the output.
Everything else runs on the cheap tier or spends no tokens at all. (Retrieval's
only LLM use is the lightweight query-generation step, which falls back to the raw
fact text token-free if the budget is gone.)

---

## 8. Cost Guardrail

A **soft** daily token budget protects against runaway cost without hard-stopping
a run mid-pipeline.

- **Where:** `config/providers.py`, enforced inside `llm_call`.
- **State:** rolling per-day usage persisted to `daily_usage.json` (gitignored):
  `{"YYYY-MM-DD": {"input": N, "output": N, "total": N}}`.
- **Default:** `DAILY_TOKEN_BUDGET = 500_000` (override via env var).

### Behavior

```
llm_call(...)
  used = today's total tokens
  if used >= BUDGET and not allow_over_budget:
      log warning
      raise BudgetWarning        # ← caller can catch and continue
  ... make the call ...
  record usage
  if new_total >= BUDGET:
      log warning (but DO NOT raise — this call is already paid for)
  return content
```

`BudgetWarning` is an `Exception` (not a `warnings.Warning`) so it interrupts
control flow and forces a deliberate decision. A caller that wants to proceed
catches it and re-invokes with `allow_over_budget=True`:

```python
try:
    result = providers.llm_call(prompt, model_tier="premium")
except providers.BudgetWarning:
    # decide: stop the batch, alert, or override
    result = providers.llm_call(prompt, model_tier="premium", allow_over_budget=True)
```

**Soft, by design.** A hard limit that abandons a half-finished dossier wastes
the tokens already spent. The soft guardrail surfaces the overage, records every
call's true cost, and leaves the continue/stop decision to the caller.

---

## 9. Testing Strategy

- **Offline-first.** **230 tests** run with **no API key** — every LLM call is
  stubbed via monkeypatch, so CI never spends tokens or flakes on the network.
- **What's covered, by suite:**
  - `test_claims.py` — text normalization, URL validation, image/video handlers,
    the Pydantic models + enforced invariant (both directions), the daily-usage
    ledger + budget guardrail, and the extractor's structured-output → model
    mapping (including a guard that the extractor prompt still demands
    *self-contained* atomic facts).
  - `test_evidence.py` — PubMed/web result parsing, source-tier classification
    (including `.edu` spam demotion), the relevance floor, ranking that blends
    relevance with a tier bonus, and URL de-duplication.
  - `test_verdicts.py` — stance bucketing, verdict/flag mapping, the defensive
    fallbacks (missing fact index, unknown verdict string, LLM error →
    Insufficient, `BudgetWarning` propagation), and dossier assembly + rendering
    (including the deterministic narrative fallback).
  - `test_api.py` — the FastAPI adapter: exactly-one-input, status-code mapping,
    base64 / data-URL images, Dossier serialization, the cache.
  - `test_claim_corpus.py` — the diverse claim corpus: ambiguous, multi-claim,
    no-claim, sarcastic, vague, well-supported, form-mismatch,
    association-vs-causation and wrong-population claims (each locked to the
    pipeline behavior it exercises — decomposition invariants, the applicability
    cap, the prompt rules) plus empty, extremely long, special-character,
    zero-width/control-character and non-English inputs.
  - `test_hardening.py` — degradation paths (PubMed down, Tavily missing vs.
    down, everything down), the malformed-output retry contract, budget
    exhaustion at each stage, "every error is a sentence, never a traceback"
    (API and CLI), the LRU cache, stage timing, and the reasoning-integrity
    guardrails (animal filter, applicability cap, evidence-bound flags).
- **Stubbed, not live.** The corpus tests fix the model's structured output and
  test what the pipeline *does with it*; they are regression tests for our
  code and prompts, not measurements of model quality. A live quality
  benchmark against the real model remains a manual exercise.

The principle: a claim-checker is only as trustworthy as its own test suite.

---

## 10. Graceful Degradation

The pipeline's failure policy: **once a claim has been extracted, a dossier is
always produced.** Everything that can go wrong after that point becomes a
plain-language entry in `Dossier.limitations`, rendered as a "Limitations of
this run" panel (web) or block (terminal), so the reader can weigh the verdicts
knowing what they lack.

| Failure | Where handled | What the reader gets |
|---------|---------------|----------------------|
| PubMed unreachable for every query | `evidence_retriever` → `source_notes` | Web-only evidence + "PubMed was unavailable…" |
| `TAVILY_API_KEY` not set | `evidence_retriever` → `source_notes` | PubMed-only evidence + "web search was skipped (no key)…" |
| Tavily configured but failing | same | PubMed-only evidence + "web search was unavailable…" |
| Both sources down | same | Empty evidence + a single "no evidence source was reachable" note; verdicts Insufficient |
| LLM output malformed (any structured call) | `providers.llm_call` | One automatic retry; then `MalformedOutputError` → extraction: `502` / CLI error line; verdicts: Insufficient; narrative: tally + note |
| Token budget exhausted **before** extraction | `check_claim` raises `BudgetWarning` | `503` / CLI exit 2 — nothing to complete yet |
| Budget exhausted at the **verdict** stage | `complete_dossier` | Facts marked Insufficient with evidence attached *unassessed* + note |
| Budget exhausted at the **narrative** stage | `dossier_builder` | Deterministic tally summary + note |
| Anything unanticipated (API) | `@app.exception_handler(Exception)` | `500` with a one-sentence `detail`; trace goes to the server log only |

Errors *before* extraction — bad URL, unreadable image, empty text, text over
`MAX_TEXT_CHARS` — are real request errors and map to `400` / `413` with the
reason in `detail`.

---

## 11. Dossier Cache & Stage Timing

**Cache** (`core/cache.py`). A bounded (256-entry), thread-safe LRU of `Dossier`s
keyed on the *normalized* claim text (`clean_text`), so trailing whitespace,
smart quotes, zero-width characters and line endings all hit the same entry. A
hit returns the previously built dossier — same UUID, same verdicts — and spends
zero tokens. Only **text** input is cached: a URL can change between visits and
an image/video is a new upload every time. No-claim results (`422`) are never
cached. The cache is per-process and in-memory; semantic caching of *paraphrased*
claims is a later iteration.

**Timing.** Every stage runs under `stage_timer(...)` in `main.py`, which logs
one line per stage to the server console:

```
INFO claimcheck.pipeline: [timing] normalize    0.00s
INFO claimcheck.pipeline: [timing] extract      2.81s
INFO claimcheck.pipeline: [timing] retrieve    14.62s
INFO claimcheck.pipeline: [timing] evaluate    11.05s
INFO claimcheck.pipeline: [timing] assemble     4.37s
INFO claimcheck.pipeline: [timing] total       32.85s
```

`complete_dossier(..., timings={})` also fills the dict for callers that want the
numbers programmatically.
