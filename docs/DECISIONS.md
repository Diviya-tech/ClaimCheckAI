# Architecture Decision Records

This log captures the major design decisions behind ClaimCheck AI — the problem
faced, the options weighed, the choice made, and *why*. Each record is meant to
be read by a future contributor (or a future us) asking "why is it built this
way?"

For the research these decisions draw on, see [`RESEARCH.md`](RESEARCH.md).

---

## ADR-001 — Evidence Dossier over a numerical trust score

**Context.** The obvious output for an automated fact-checker is a single
credibility number ("73% true"). It's easy to compute, easy to display, easy to
sort. But it collapses all reasoning into one opaque figure.

**Options considered.**
1. A 0–100 credibility / trust score.
2. A single binary label (true / false).
3. A structured **evidence dossier** — per-fact verdicts + cited evidence + reasoning.

**Decision.** Build the output as an **evidence dossier**, not a score.

**Why.** Research at CHI 2025 ("Show Me the Work") found professional
fact-checkers consider numerical credibility scores *unhelpful and disconnected
from how they actually reason*. A score hides the work; a dossier shows it. Our
product thesis is "show me the work, not a number" — so the output format has to
*be* the work: every atomic fact, its verdict, the evidence behind it, and the
reasoning that connects them. A consumer can then form their own judgment instead
of trusting a black-box figure.

---

## ADR-002 — Claim decomposition with per-atom verdicts

**Context.** Real health claims are compound. "Green tea boosts metabolism and
melts belly fat in two weeks" bundles a causal claim, a stronger causal claim,
and a temporal claim. A single verdict over the whole sentence is *guaranteed* to
be wrong for at least one part.

**Options considered.**
1. Evaluate the whole claim as one unit, one verdict.
2. Decompose into atomic testable facts, evaluate each **independently**.

**Decision.** Decompose every claim into atomic facts and assign a verdict
**per atom** (inspired by EVICheck, IJCAI 2025).

**Why.** A compound claim needs compound evaluation. Decomposition (a) produces
honest, granular verdicts — one atom can be *Strongly Supported* while another is
*Strongly Refuted*; (b) makes the reasoning legible — the user sees exactly which
part fails; and (c) is the foundation of our token efficiency, since we reason
over short atoms instead of raw walls of text (an 80–90% input reduction
downstream). We enforce this structurally: a found claim **must** decompose into
≥1 atom (a Pydantic invariant), so the architecture can't silently skip
decomposition.

---

## ADR-003 — LLM-agnostic provider abstraction

**Context.** Frontier models ship every few months, prices shift, and the best
model for a task changes. If pipeline code imports a specific SDK directly,
swapping models means touching every call site.

**Options considered.**
1. Call the Anthropic SDK directly throughout the pipeline.
2. A thin provider abstraction — one `llm_call(...)` entry point, providers behind
   an interface, models named only in config.

**Decision.** All LLM access goes through `config/providers.py`. Pipeline code
never imports `anthropic` / `openai`; it asks for a model *tier*.

**Why.** The pipeline logic is the durable asset; the LLM is a **replaceable
component**. Treating it as swappable infrastructure means: adopting a new model
is a one-file config change; adding a provider (OpenAI, local model) is one new
`LLMProvider` subclass with zero pipeline edits; and we can A/B providers per
tier. This also keeps a clean seam for testing — the offline suite stubs the
provider and never spends a token.

---

## ADR-004 — Seven categorical verdicts over binary true/false

**Context.** Evidence is rarely a clean yes/no. Sometimes it's mixed, sometimes
thin, sometimes the claim is too vague to test at all. A binary label forces all
of that nuance into two buckets.

**Options considered.**
1. Binary: true / false.
2. Ternary: supported / refuted / not enough evidence (AVeriTeC's core set).
3. A richer **seven-category** scale spanning supported → refuted, plus
   conflicting, insufficient, and too-vague.

**Decision.** Seven categorical verdicts: *Strongly Supported, Partially
Supported, Insufficient Evidence, Conflicting Evidence, Partially Refuted,
Strongly Refuted, Too Vague to Evaluate*.

**Why.** Categories map to how people actually reason about claims (building on
AVeriTeC's categorical scheme — Supported / Refuted / Not Enough Evidence /
Conflicting). Binary is dishonest for the common "partly true, partly
overstated" case. *Conflicting Evidence* explicitly captures cherry-picking;
*Insufficient Evidence* and *Too Vague to Evaluate* let the system admit the
limits of what it can judge instead of forcing a false verdict. Crucially, these
are still **categories, not numbers** (per ADR-001) — graded, but not a fake
precision score.

---

## ADR-005 — Start with the health domain as the vertical

**Context.** Misinformation spans politics, finance, science, health, and more.
A general-purpose checker is tempting but spreads the evidence problem thin.

**Options considered.**
1. Build a domain-general fact-checker.
2. Specialize in one high-stakes vertical first.

**Decision.** Start narrow: **health claims**.

**Why.** Health is where (a) the harm is direct and personal, (b) high-quality,
*structured*, *free* evidence exists (PubMed, Cochrane, WHO, CDC) — which the
whole zero-token retrieval strategy depends on, and (c) source-quality tiers are
unusually clear-cut (a meta-analysis genuinely outranks a blog). A general
checker has no equivalent of PubMed. Nailing one vertical with a real evidence
backbone beats a shallow generalist, and the pipeline architecture generalizes to
other verticals later.

---

## ADR-006 — PubMed + Qdrant for evidence, not general web search

**Context.** Stage 3 needs evidence for each atomic fact. The default instinct is
to web-search and feed results to the LLM.

**Options considered.**
1. LLM-powered web search / retrieval (LLM "looks things up").
2. A curated medical corpus in a vector DB (Qdrant), sourced from PubMed/WHO/CDC/
   Cochrane, with web search (Tavily) only as supplement.

**Decision.** Curated corpus + **Qdrant vector search** as the primary evidence
source; PubMed via Biopython; Tavily only to fill gaps.

**Why.** Three reasons. **Quality:** PubMed/Cochrane/WHO are exactly the
high-tier sources our verdicts should rest on — general web search surfaces SEO
spam alongside science. **Cost:** vector search and these APIs are *free and use
zero LLM tokens*; LLM-powered retrieval burns premium tokens to do worse lookup.
**Control:** a curated, tiered corpus lets us weight evidence by source quality
(ADR-007), which an opaque web search can't. We spend tokens on *reasoning*, not
on *finding*.

---

## ADR-007 — Source quality tiers

**Context.** If all retrieved evidence is treated equally, ten low-quality blog
posts can numerically drown out one Cochrane review.

**Options considered.**
1. Treat all sources equally (count/relevance only).
2. Tag every source with a quality tier and weight verdicts accordingly.

**Decision.** A four-tier hierarchy (Tier 1 systematic reviews/guidelines → Tier
4 general web), with Tier 4 used for context only, never as evidence.

**Why.** Evidence quality *is* the substance of a health verdict. A systematic
review and a wellness blog are not the same kind of object, and a credible system
has to say so. Tiering lets a single Tier-1 meta-analysis outweigh a pile of
Tier-3/4 articles, and lets the dossier show *why* a verdict leans the way it does
(via tier badges on each cited source). It also guards against the failure mode
where SEO volume beats scientific quality.

---

## ADR-008 — Two-tier model routing

**Context.** Using the most capable (expensive) model for every stage is simplest
but wasteful — most stages don't need deep reasoning.

**Options considered.**
1. One premium model for the whole pipeline.
2. One cheap model for the whole pipeline.
3. **Route by task:** cheap model for mechanical work, premium model only where
   reasoning depth changes the answer.

**Decision.** Two tiers — **lightweight** (Haiku) for claim extraction and
classification; **premium** (Sonnet) for evidence evaluation and verdict
generation.

**Why.** Extraction and classification are pattern tasks a small model does well;
verdict generation is where reasoning quality actually matters. Routing by need
keeps 60–70% of the pipeline on the cheap tier without sacrificing verdict
quality — a large cost reduction for a consumer-scale tool. The tier is just a
config key, so re-balancing later is trivial.

---

## ADR-009 — Soft cost guardrails over hard limits

**Context.** A token budget is needed to prevent runaway cost. But a hard cap can
abandon a half-finished dossier mid-pipeline, wasting everything already spent.

**Options considered.**
1. Hard limit: refuse all calls once the budget is hit.
2. No limit: just log usage.
3. **Soft guardrail:** raise a catchable `BudgetWarning` at the budget line; let
   the caller decide to stop or override.

**Decision.** A soft daily budget (`DAILY_TOKEN_BUDGET`, default 500k) enforced in
`llm_call`, raising a catchable `BudgetWarning`; usage tracked in
`daily_usage.json`.

**Why.** Cost control shouldn't destroy work in progress. The soft model surfaces
the overage loudly (it's an `Exception`, not a silent warning), records every
call's true cost, and hands the continue/stop decision to the caller
(`allow_over_budget=True` to proceed). A call that *crosses* the line still
returns its result (already paid for) and warns; only the *next* call is gated.
This gives budget visibility and protection without the brittleness of a hard
kill-switch.

---

## ADR-010 — Script-first development, FastAPI later

**Context.** The end product is a web app with an API. It's tempting to start
with the web framework in place.

**Options considered.**
1. Build FastAPI + frontend from day one.
2. Build the pipeline as a plain script/CLI first; wrap it in FastAPI in weeks
   9–10.

**Decision.** Script-first. `main.py` is a CLI orchestrator now; the HTTP layer
comes after the pipeline is proven.

**Why.** The hard, novel problem is the *pipeline* (decomposition, retrieval,
verdicts) — not request routing. Building the web layer first would mean
designing endpoints around stages that don't exist yet. A CLI gives the fastest
iteration loop, the simplest tests, and a clean separation: when the pipeline is
solid, FastAPI just calls `check_claim(...)`. The orchestrator's function
signature is already shaped to drop into a request handler unchanged.

---

## ADR-011 — One batched verdict call over per-atom calls

**Context.** Stage 4 must evaluate every atomic fact against its own evidence.
The straightforward implementation is one premium LLM call per atom.

**Options considered.**
1. One premium call per atomic fact.
2. One **batched** premium call covering all atoms (each with its evidence) plus
   the claim-level rhetorical analysis.

**Decision.** A single batched call — `evaluate(...)` — returns verdicts for every
atom *and* the rhetorical flags in one round-trip.

**Why.** Per-atom calls re-pay the (large) system prompt and shared claim context
on every fact; batching amortizes them across all atoms, which is a real token
saving on a multi-atom claim (and the whole project is organized around token
efficiency). It also lets the model reason about the atoms *together* — e.g.
recognizing that a causal atom is supported while its temporal sibling is not. The
risk of batching (a malformed or partial response corrupting everything) is
contained by ADR-012's defensive mapping. The verdict/stance `enum`s in the
output schema are generated from the Pydantic enums, so the batch schema can never
silently drift from the models.

---

## ADR-012 — Drive result assembly off the input facts, not the model output

**Context.** A batched call returns a JSON array of per-fact verdicts. Nothing
*guarantees* the model returns exactly one entry per fact, in order — it might
skip a fact, reorder them, duplicate one, emit an out-of-range evidence index, or
return a verdict string outside the seven categories.

**Options considered.**
1. Trust the model output shape; map it directly to `AtomicVerdict`s.
2. Index the model output by `fact_index`, then **iterate over the input facts**
   and look each one up, with safe defaults for anything missing or malformed.

**Decision.** Option 2 — the loop is driven by the input facts, so the result
always has exactly one verdict per fact, in the original order.

**Why.** The verdict stage is the trust core of the product; a silently dropped or
mis-indexed fact would mean a dossier that looks complete but isn't. Making the
input the source of truth turns every model misbehavior into a *safe, visible*
default: a fact with no returned verdict becomes `Insufficient Evidence` (evidence
preserved as context), an unknown verdict string coerces to `Insufficient`, and a
non-budget LLM error degrades the whole stage to `Insufficient` rather than
crashing. `BudgetWarning` is the one exception that propagates — a cost decision
belongs to the caller (ADR-009), not to a silent fallback.

---

## ADR-013 — Rhetorical flags decoupled from evidence verdicts

**Context.** A claim can be manipulative in *form* ("doctors don't want you to
know this ONE trick that MELTS fat!") while being partly true in *substance* — and
vice versa. The rhetorical analysis and the evidence verdict answer different
questions.

**Options considered.**
1. Fold rhetoric into the verdict — let manipulative framing push a fact toward
   "Refuted".
2. Detect rhetorical patterns **separately** and present them as context that
   never changes the evidence verdict.

**Decision.** Rhetorical red flags are detected (in the same batched call, for
token efficiency) but kept in their own `RhetoricalFlag` list on the dossier,
structurally separate from the per-atom verdicts.

**Why.** Conflating the two would make the system dishonest in both directions: it
would penalize a soberly-worded false claim too little and a sensationally-worded
true claim too much. Keeping them separate lets the dossier say two true things at
once — "the evidence partially supports this" *and* "the way it's phrased is
manipulative" — which is exactly the nuance a consumer needs. Each flag carries
the triggering excerpt so the reader can see the pattern for themselves (the
"show me the work" principle, ADR-001, applied to rhetoric).

---

## ADR-014 — Narrative summary as a separate call with a deterministic fallback

**Context.** The dossier ends with a plain-language summary for a non-expert. It
needs the *finalized* verdicts, and it's the last thing produced — after the
expensive verdict call has already been paid for.

**Options considered.**
1. Fold the narrative into the batched verdict call.
2. A separate premium call in the dossier builder, with a deterministic fallback
   if it fails.

**Decision.** A second premium call writes the narrative; if it fails or the
budget is exhausted, the builder falls back to a summary assembled deterministically
from the verdict tally.

**Why.** Separating presentation (stage 5) from analysis (stage 4) keeps the
verdict logic independent of how it's rendered and lets the summary reason over
the *decided* verdicts rather than guessing alongside them. The fallback is the
important half: the verdicts are the costliest artifact in the pipeline, so a
failed *summary* call must never discard them — a dossier is always produced, just
with a plainer summary. This mirrors the "degrade, don't crash" stance of ADR-009.

---

## ADR-015 — Animal and in-vitro studies are context, never evidence

**Context.** PubMed returns rat, mouse, and cell-culture studies alongside human
trials, and for supplement claims they are often the *majority* of hits. A
model reading "curcumin reduced tumour growth in mice" as support for "turmeric
cures cancer" is the single most common way an evidence-based system ends up
over-stating a health claim.

**Options considered.**
1. Drop animal studies at retrieval — never show them.
2. Keep them, tag them, and rely on the prompt to discount them.
3. Keep them, tag them deterministically, **and** enforce in code that they can
   only ever be neutral context.

**Decision.** Option 3. `sources/pubmed.py` parses MeSH headings and flags an
article `is_animal_study` when it carries `Animals` (or a species term) without
`Humans`, with a keyword fallback for un-indexed or in-vitro work. Retrieval
stamps such items `applicability = non_human`, ranks them below human evidence,
keeps at most one per fact, and excludes them from the "enough evidence to
evaluate" threshold. The verdict engine forces their stance to `neutral`
regardless of what the model returned.

**Why.** Dropping them (option 1) would hide real context the reader deserves to
see — "the only evidence is in rats" is itself a finding. Trusting the prompt
alone (option 2) leaves the guardrail one paraphrase away from failing. Enforcing
it in code means the dossier can *show* the rat study, labelled as such, while
guaranteeing it never becomes the reason a human claim is called supported.

---

## ADR-016 — Applicability as a second dimension of evidence, with a cap on "Strongly"

**Context.** Stance (supporting / opposing / neutral) says which *way* evidence
points. It says nothing about how much it can *count*: a capsule-extract trial in
postmenopausal women can point the same way as "turmeric tea cures arthritis" and
still be unable to establish it. The six reasoning rules in the prompt describe
these gaps (form, dose, population, proxy outcome) but a rule in prose does not
change a verdict.

**Options considered.**
1. Leave applicability to the reasoning text; trust the model to pick the right
   verdict.
2. Add `applicability` (`direct` / `indirect` / `non_human`) to every evidence
   item and make the strongest verdicts *conditional* on direct evidence.

**Decision.** Option 2. The model rates each item `direct` (same intervention and
form, a human population the claim targets, the asserted outcome) or `indirect`.
After the model answers, `Strongly Supported` is downgraded to `Partially
Supported` unless at least one *direct* supporting item exists, and `Strongly
Refuted` to `Partially Refuted` unless at least one *direct* opposing item does;
the reasoning is annotated with the reason. Unrated evidence counts as indirect.

**Why.** "Strongly" is the verdict a reader acts on, so it is the one that must
be hardest to reach. Making it depend on a structured field the model fills per
item — rather than on the model remembering a rule while writing a paragraph —
turns the form/population rules from advice into a constraint, while leaving the
model full latitude on everything below "Strongly". The field also surfaces in the
UI ("Indirect", "Animal / in-vitro" badges), so the reader sees the gap, not just
its consequence.

---

## ADR-017 — Granular decomposition: the form of a substance is its own atomic fact

**Context.** Health claims on social media are overwhelmingly about *preparations*
— turmeric tea, cumin water, celery juice, ACV gummies — while the literature
studies *standardized extracts* at known doses. Decomposing "turmeric tea cures
arthritis" into a single causal atom forces one verdict to answer two different
questions: does curcumin affect arthritis, and does tea deliver curcumin?

**Options considered.**
1. One atom; rely on the verdict reasoning to mention the form gap.
2. Emit a separate atomic fact for the form ("turmeric tea delivers curcumin at a
   dose comparable to the amounts studied in clinical trials"), evidenced and
   judged on its own.

**Decision.** Option 2. The extractor prompt now treats a named preparation as
part of the claim and emits a distinct (usually quantitative) atom for it.

**Why.** This is ADR-002 (per-atom verdicts) applied to the gap that actually
sinks most supplement claims. With two atoms the dossier can say the honest
thing — effect *Partially Supported*, form *Strongly Refuted* — and the reader
sees exactly where the claim breaks. It also gives the retriever a query aimed at
dose/bioavailability literature instead of hoping the effect query surfaces it.
The cost is one more atom per form-bearing claim, which is small next to the
batched verdict call.

---

## ADR-018 — Rhetorical flags must be bound to text the reader can find

**Context.** Rhetorical red flags (ADR-013) are the dossier's most
accusatory output: "conspiracy framing", "emotional manipulation". A flag whose
excerpt is a paraphrase ("the claim uses fear") or an inference the words don't
carry ("act now before it's banned" on a claim that never said so) damages
trust in every other flag and, by association, in the verdicts.

**Options considered.**
1. Ask the prompt for verbatim excerpts and trust it.
2. Ask for verbatim excerpts **and** drop, in code, any flag whose excerpt does
   not appear in the claim text.

**Decision.** Option 2. `_assemble_flags` keeps a flag only if its `excerpt`
occurs in the original text (case-, curly-quote- and whitespace-insensitive).
Flags with an empty excerpt are dropped too.

**Why.** A red flag is an *observation about the text*, so the text is its only
admissible evidence — the same "show me the work" standard the verdicts are held
to (ADR-001). The check is cheap, deterministic, and fails safe: the worst case
is a missed flag, never an invented one. It also removes the incentive for the
model to flag a soberly worded claim just because the category list was long.

---

## ADR-019 — `.edu` earns Tier 3 only when the URL looks like university content

**Context.** The source classifier promoted any `.edu` host to Tier 3
("university health center"). University domains are routinely hijacked —
abandoned personal pages, compromised CMS uploads, open redirects — precisely
*because* their credibility transfers to whatever is hosted there. An
SEO-spam page about a supplement, sitting on `people.someuni.edu/~old/`, would
outrank the same page on a blog.

**Options considered.**
1. Drop the blanket `.edu` rule; list trusted university health centers
   individually.
2. Keep the rule but demote a `.edu` URL whose path or query carries obvious
   spam signals.

**Decision.** Option 2. For `.edu` hosts only, the classifier scans the path and
query string for an injected external URL (raw, percent-encoded, a bare `www.`,
or a TLD-like token), marketing / affiliate parameters (`utm_*`, `gclid`, `ref`,
`affiliate`, `promo`, …), and redirect parameters or path segments (`?url=`,
`?redirect=`, `/go/`, `/redirect/`, …). Any hit demotes the source to Tier 4 with
a justification naming the signal. The host itself is never inspected.

**Why.** An allow-list (option 1) would be perpetually incomplete and would drop
legitimate university content we haven't listed yet. The URL heuristics target
the mechanics of hijacking rather than any particular spammer, cost nothing, and
keep the classifier's transparency: the dossier states *why* a university link
was not trusted. The one ordering caveat is deliberate — a title or PubMed
publication type marking a systematic review still wins (rule 1 of the
classifier), because that signal is stronger than the host.
