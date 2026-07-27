# Research Foundations

ClaimCheck AI is not designed from intuition — it synthesizes findings from recent
academic work in automated fact verification and fact-checker tooling. This
document summarizes the four research threads that shaped the design and how each
maps onto a concrete decision.

> These are the works that informed the architecture. Where a finding drove a
> specific choice, the relevant ADR is linked — see [`DECISIONS.md`](DECISIONS.md).

---

## 1. AVeriTeC (University of Cambridge) — categorical verdicts

**The work.** AVeriTeC is a dataset and shared task for real-world claim
verification with evidence retrieved from the web. Rather than scoring claims
numerically, it labels them with a small set of **categorical verdicts**:

- **Supported**
- **Refuted**
- **Not Enough Evidence**
- **Conflicting Evidence / Cherry-picking**

**Why it matters here.** AVeriTeC validates that *categories*, not numbers, are
the right output shape for a verdict — and that the category set must include
honest "can't tell" states (*Not Enough Evidence*) and the cherry-picking case
(*Conflicting Evidence*). A binary true/false scheme can't express either.

**What ClaimCheck took.** Our seven-category verdict scale is a refinement of
AVeriTeC's set: we keep Supported / Refuted / Not-Enough / Conflicting, split
support and refutation into *Strongly* vs *Partially* for granularity, and add
*Too Vague to Evaluate* for unfalsifiable claims. → **ADR-004**.

---

## 2. EVICheck (IJCAI 2025) — evidence-driven atomic reasoning

**The work.** EVICheck approaches verification through **evidence-driven,
independent reasoning over fine-grained criteria**: a claim is broken into atomic
components, and each is assessed on its own against retrieved evidence rather than
judged holistically.

**Why it matters here.** It establishes that **atomic decomposition** plus
**independent per-unit evaluation** produces more faithful, more legible verdicts
than scoring a compound claim as a single block. The granularity is what makes
the reasoning auditable.

**What ClaimCheck took.** The core engine — Claim Decomposition with Per-Atom
Verdicts. Every claim becomes a set of atomic facts; each fact is independently
retrieved-for and judged. This is also the lever behind our token efficiency
(reasoning over short atoms, not raw text). → **ADR-002**.

---

## 3. CHI 2025 "Show Me the Work" — scores are unhelpful to experts

**The work.** A CHI 2025 study of professional fact-checkers examined how they
respond to automated credibility outputs. A central finding: practitioners found
**numerical credibility scores unhelpful and disconnected from how they actually
reason** about claims. What they wanted was the *evidence and reasoning*, not a
distilled figure.

**Why it matters here.** This is the empirical case against the default design
(a trust score). If the people best at this task reject numerical scores, building
the product around one is building the wrong thing.

**What ClaimCheck took.** The output is an **Evidence Dossier**, not a score —
each atomic fact shown with its verdict, its cited evidence (tier-badged), and the
reasoning connecting them. "Show me the work" is, almost literally, the product
spec. → **ADR-001**.

---

## 4. Community Notes bridging algorithm — convergence as signal

**The work.** X/Twitter's Community Notes uses a *bridging* algorithm: a note is
surfaced not by raw vote count but when it earns agreement **across normally
disagreeing groups**. Convergence among diverse, independent raters is treated as
the credibility signal — agreement that crosses divides means more than agreement
within a bubble.

**Why it matters here.** It reframes credibility as **convergence across
diverse, independent sources** rather than volume from one. For evidence, that
maps directly: agreement across independent high-tier sources (a Cochrane review
*and* a WHO guideline *and* multiple PubMed studies) is a far stronger signal than
ten articles echoing one press release.

**What ClaimCheck took.** The principle informs two things, both now live: our
**source-quality tiers** (the verdict engine is prompted to weight independent,
high-tier *convergence* and discount echo-chamber volume — **ADR-007**), and the
**rhetorical red-flag detection** shipped in the verdict engine (weeks 7–8), which
surfaces manipulation patterns — conspiracy framing, guaranteed outcomes,
anecdote-as-proof — as dossier context, kept structurally separate from the
evidence verdict so form and substance are judged independently (**ADR-013**).

---

## Synthesis — how the four combine

ClaimCheck AI is not any one of these papers; it's a **unified approach** that
takes one core idea from each:

| Source | Contribution | Where it lives |
|--------|--------------|----------------|
| **AVeriTeC** | Categorical verdicts with honest "can't tell" + conflicting states | 7-category verdict scale (ADR-004) |
| **EVICheck** | Atomic decomposition + independent per-unit evaluation | The claim-extractor → per-atom verdict engine (ADR-002) |
| **CHI 2025** | Reject numerical scores; show evidence and reasoning | The Evidence Dossier output format (ADR-001) |
| **Community Notes** | Convergence across diverse sources as the credibility signal | Source-quality tiers (ADR-007) + rhetorical red-flag detection (ADR-013) |

The result is the combined thesis stated in the README:

> **Claim Decomposition with Per-Atom Verdicts** (the engine, from EVICheck +
> AVeriTeC) wrapped in an **Evidence Dossier** (the output, from CHI 2025), with
> credibility judged by **convergence across tiered, independent sources** (from
> Community Notes).

No numerical trust scores. Atomic, evidence-grounded, categorical, and
transparent — because that's what the research says actually helps people reason
about what they're seeing.

---

### References

- **AVeriTeC** — *A Dataset for Real-world Claim Verification with Evidence from the Web*, University of Cambridge.
- **EVICheck** — evidence-driven claim verification with fine-grained atomic criteria, IJCAI 2025.
- **"Show Me the Work"** — study of professional fact-checkers and automated credibility outputs, CHI 2025.
- **Community Notes** — X/Twitter bridging-based note ranking algorithm (open-sourced).

*Citations reflect the research that informed this project's design; consult the
primary sources for full methodology and results.*
