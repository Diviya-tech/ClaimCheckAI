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
evidence — spending **zero LLM tokens**.

- **Curated medical corpus** indexed in **Qdrant** (vector DB): PubMed abstracts,
  WHO/CDC guidance, Cochrane reviews. Retrieval is pure vector similarity.
- **Supplementary web search** (**Tavily**) when the corpus is thin on a topic.
- **Pre-compressed summaries** stored alongside full abstracts — the summary is
  what gets sent to the verdict stage, not the full text.
- **Relevance filtering** — only the top 3–5 chunks per atom move forward.

This stage is deliberately LLM-free: a vector database does similarity lookup
better and cheaper than an LLM "searching." Tokens are reserved for *reasoning*,
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

### Defensive by construction

The mapping from model output back to typed models is driven off the *input*
facts, not the model's output, so the result **always has exactly one verdict per
fact, in order**, even if the model skips, reorders, or duplicates entries — a
missing fact defaults to `Insufficient Evidence` with its evidence preserved as
context. Unknown verdict/stance strings coerce to safe defaults. A non-budget LLM
error degrades the whole stage to `Insufficient Evidence` rather than crashing the
run; `BudgetWarning` alone propagates, so the caller makes a deliberate spend
decision (per ADR-009).

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
  builder falls back to a deterministic summary assembled from the verdict tally.
  The verdicts — the expensive part — are never discarded, so a dossier is
  *always* produced.
- **`format_dossier(dossier)`** renders the whole thing as clean terminal text:
  tier badges `[T1]`–`[T4]`, stance markers `(+)` / `(-)` / `(.)`, and a
  per-verdict cue. `main.py --json` additionally emits the dossier as JSON for the
  coming FastAPI layer.

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

- **Offline-first.** **111 tests** run with **no API key** — every LLM call is
  stubbed via monkeypatch, so CI never spends tokens or flakes on the network.
- **What's covered, by suite:**
  - `test_claims.py` — text normalization, URL validation, image/video handlers,
    the Pydantic models + enforced invariant (both directions), the daily-usage
    ledger + budget guardrail, and the extractor's structured-output → model
    mapping (including a guard that the extractor prompt still demands
    *self-contained* atomic facts).
  - `test_evidence.py` — PubMed/web result parsing, source-tier classification,
    the relevance floor, ranking that blends relevance with a tier bonus, and
    URL de-duplication.
  - `test_verdicts.py` — stance bucketing, verdict/flag mapping, the defensive
    fallbacks (missing fact index, unknown verdict string, LLM error →
    Insufficient, `BudgetWarning` propagation), and dossier assembly + rendering
    (including the deterministic narrative fallback).
- **What's deferred:** a 50+ case live corpus of diverse real claims, exercised
  against the real model for end-to-end quality regression.

The principle: a claim-checker is only as trustworthy as its own test suite.
