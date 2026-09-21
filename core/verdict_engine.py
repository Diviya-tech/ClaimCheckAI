"""Verdict engine — stage 4 of the pipeline (weeks 7-8).

Takes the extracted claim and the evidence retrieved per atomic fact, and reasons
about each fact against its evidence to assign a categorical verdict. This is the
stage that turns "here is a claim and some abstracts" into "here is what the
evidence actually says about it."

Unlike retrieval (which is deliberately token-free), evaluation is genuine
reasoning, so it runs on the PREMIUM model tier. To stay token-efficient it does
the whole job in ONE batched call: every atomic fact, all of its evidence, and
the claim-level rhetorical analysis are evaluated together, and the model returns
a single structured object we map back onto typed models.

For each atomic fact the engine:
  a. classifies each piece of evidence as supporting / opposing / neutral,
  b. weighs the evidence by source tier (a Tier-1 systematic review outweighs a
     stack of Tier-3/4 blog posts) and by whether independent sources converge,
  c. assigns one of the seven categorical verdicts, and
  d. writes a plain reasoning string that cites specific evidence.

A verdict is a statement about the EVIDENCE, not about the world, so the prompt
draws a hard line between evidence that contradicts a fact and evidence that is
merely silent on it. If nothing retrieved actually tested the parameter claimed
(the timeframe, the dose, the preparation), the verdict is Insufficient Evidence
— "no study looked" is not "the study found nothing". Only evidence that is
incompatible with a fact can refute it.

Separately (same call) it inspects the ORIGINAL claim text for rhetorical red
flags — guaranteed outcomes, conspiracy framing, appeal to nature, anecdote-as-
proof, false urgency, emotional manipulation. Flags describe how the claim is
argued; they never change the evidence verdict.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from config import providers
from schemas.models import (
    AtomicVerdict,
    ClaimExtractionResult,
    Evidence,
    EvidenceApplicability,
    EvidenceStance,
    FactEvidence,
    RhetoricalFlag,
    Verdict,
)

logger = logging.getLogger("claimcheck.verdict_engine")

# How much of each evidence summary/abstract to send to the model. Retrieval
# already stores a pre-compressed 2-3 sentence summary; this only bites when a
# web result has no summary and we fall back to raw content.
_EVIDENCE_SNIPPET_CHARS = 600


# --------------------------------------------------------------------------- #
# Return container
# --------------------------------------------------------------------------- #
class ClaimEvaluation(BaseModel):
    """Everything the verdict stage produces in its single batched call.

    `evaluate_claim` returns just the verdicts (the spec'd API); `evaluate`
    returns this richer object so the dossier builder also gets the claim-level
    rhetorical flags without a second premium call.
    """

    verdicts: list[AtomicVerdict] = Field(default_factory=list)
    rhetorical_flags: list[RhetoricalFlag] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompt + structured-output schema
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """\
You are the verdict stage of an evidence-evaluation engine for health claims. You
do NOT decide what is true in the abstract — you decide what the SUPPLIED evidence
says about each claim, and you show your work.

You are given a primary claim broken into atomic facts. Each atomic fact comes
with the evidence retrieved for it: numbered items, each tagged with a source
quality tier (T1 best, T4 weakest) and a short summary.

Source quality tiers:
  T1 — systematic reviews, meta-analyses, WHO/CDC guidelines (strongest).
  T2 — peer-reviewed primary studies, major medical institutions (NIH, Mayo).
  T3 — credentialed health journalism, university health centers.
  T4 — general web, blogs, social media — CONTEXT ONLY, never treat as evidence.

For EACH atomic fact, do all of the following:

1. STANCE: Classify every evidence item as one of:
   - "supporting": its findings back up the fact.
   - "opposing": its findings contradict or refute the fact.
   - "neutral": related/background, but it does not itself confirm or deny the fact
     (e.g. it studies a different population, a different dose, or is T4 context).

1b. APPLICABILITY: Separately from stance, rate how directly each item applies
   to THIS fact:
   - "direct": it tested the same intervention in the same form (tea vs extract
     vs capsule), in a human population the claim is aimed at, and measured the
     outcome the fact asserts.
   - "indirect": it is about the same topic but differs in form, dose, route,
     population, or measures a proxy outcome. Indirect evidence can qualify a
     verdict but cannot on its own establish or refute the fact.
   Items marked [NON-HUMAN STUDY] were run in animals or cells. Their stance must
   be "neutral": an effect in rats or a petri dish is context for a human claim,
   never support or refutation of it. Say so in the reasoning when they are the
   only evidence.

2. WEIGH BY TIER: Higher tiers carry more weight. A single T1 meta-analysis
   outweighs several T3/T4 items pointing the other way. Never let T4 sources
   alone drive a "supported" or "refuted" verdict.

3. SOURCE DIVERSITY: Note whether independent institutions converge on the same
   finding (stronger) or whether the support comes from a single source or from
   sources that merely echo each other (weaker).

4. VERDICT: Assign EXACTLY ONE categorical verdict:
   - "Strongly Supported": strong, consistent higher-tier evidence supports it;
     little or no credible opposition. Requires at least one DIRECT supporting
     item — indirect evidence alone caps the verdict at Partially Supported.
   - "Partially Supported": real support, but qualified — limited, lower-tier,
     narrow conditions, or only part of the fact holds.
   - "Insufficient Evidence": not enough quality evidence was found to evaluate
     this specific fact in EITHER direction. Use this when:
       * no retrieved study addresses the specific parameters of the fact — its
         timeframe, dosage, preparation/form, population, or magnitude;
       * the retrieved studies tested related but DIFFERENT conditions;
       * retrieval returned very few relevant results, or none at all.
   - "Conflicting Evidence": credible evidence of comparable weight on BOTH sides.
   - "Partially Refuted": evidence leans against the fact but with caveats — it
     does not directly disprove it. Example: studies found the effect only at a
     much longer timeframe or a much higher dose, which makes the claimed
     timeframe or dose implausible without ever testing it directly.
   - "Strongly Refuted": ONLY when the evidence DIRECTLY CONTRADICTS the fact — a
     study measured the specific thing claimed and found it false, or a systematic
     review explicitly concludes the claimed effect does not exist. The evidence
     must be INCOMPATIBLE with the fact, not merely silent on it. Requires at
     least one DIRECT opposing item — indirect evidence alone caps the verdict at
     Partially Refuted.
   - "Too Vague to Evaluate": the fact itself is too vague, subjective, or
     unfalsifiable to test against evidence, regardless of what was retrieved.

5. REASONING: 2-4 sentences in plain language explaining how the verdict follows
   from the evidence. Reference specific items by their source and tier (e.g.
   "a 2019 Cochrane review (T1) found no effect"). Be honest about limitations —
   and when the verdict is "Insufficient Evidence", say plainly WHAT was never
   tested rather than implying the claim was disproven.

SIX SCIENTIFIC REASONING RULES
Apply these six reasoning rules when evaluating evidence. If any of these
distinctions are relevant to the current claim, mention them in your reasoning.
They are the difference between reporting what a study SHOWS and over-reading it.

Make the point, never the citation: these rules are internal guidance, so the
reasoning must NOT refer to them by number or name ("Rule 6", "per reasoning rule
2"). The reader never sees this list. Write the distinction itself in plain words
— "the trials used 3 g of cumin powder, not cumin water, which is far more
dilute" rather than "Rule 6 applies".

1. "No evidence found" is NOT "Refuted". Absence of evidence is not evidence of
   absence. If no study addressed the specific claim, the verdict is
   "Insufficient Evidence", not a Refuted verdict.
   CRITICAL: distinguish evidence that directly contradicts a claim from evidence
   that simply does not address it. If no retrieved study specifically tested the
   exact parameter claimed (the timeframe, dosage form, or magnitude), the verdict
   for that parameter is "Insufficient Evidence". A claim can only be "Strongly
   Refuted" when evidence is directly INCOMPATIBLE with it — not when evidence is
   merely absent. Before assigning ANY Refuted verdict, ask yourself: does the
   evidence say this is WRONG, or does it merely say we don't have proof it's
   RIGHT? If the latter, use "Insufficient Evidence". The same rule governs
   stance: an item that never measured what the fact claims is "neutral", not
   "opposing".

2. "The study lasted 12 weeks" is NOT "12 weeks are required". A study's duration
   is a DESIGN CHOICE, not a finding. If the shortest trial ran 8 weeks, that
   means nobody tested shorter — it does NOT mean shorter timeframes don't work.
   Never convert a trial's length, dose, or population into a claimed threshold
   the study never tested.

3. "Statistically significant" is NOT "clinically meaningful". A p-value below
   0.05 means the result is unlikely to be due to chance. It does NOT mean the
   effect is large enough to matter in practice. A statistically significant
   0.5 kg weight loss over 12 weeks is real but practically meaningless to someone
   expecting to "melt belly fat". When a significant effect is too small to matter
   for what the claim promises, say so and let it qualify the verdict.

4. "Association" is NOT "causation". Observational studies showing correlation
   (people who drink green tea weigh less) do NOT prove the tea caused the weight
   loss — the association may run the other way or come from a third factor. Only
   randomized controlled trials can establish causation. Flag explicitly whether
   the evidence is associational or causal, especially when judging a causal fact.

5. "Works in population X" is NOT "works for everyone". A study on postmenopausal
   women in Iran does not automatically apply to young men in Texas. Note when
   evidence comes from a specific population (age, sex, health status, region)
   that may not generalize to whoever the claim is aimed at.

6. "Ingredient/extract X" is NOT "an ordinary food or drink containing X". A study
   using 500 mg of concentrated cumin extract in capsules does NOT show that cumin
   water — which contains a fraction of that dose — has the same effect. Flag when
   the studied FORM (extract, capsule, concentration, dose, route) differs from
   the form the claim describes, and let that gap qualify the verdict.

SEPARATELY, analyze the ORIGINAL claim text for rhetorical red flags — the WAY it
is argued, independent of whether it is true. Only report patterns actually
present. Every flag MUST quote the triggering phrase VERBATIM from the claim
text in `excerpt` — copy it character for character; do not paraphrase,
summarize, or infer a tone the words themselves do not carry. A flag whose
excerpt does not appear in the text is discarded. Look for:
  - Guaranteed / absolute outcomes ("melts fat", "cures", "guaranteed", "always").
  - Conspiracy framing ("doctors don't want you to know", "they're hiding this").
  - Appeal to nature ("it's natural, so it's safe/effective").
  - Anecdote as proof ("I lost 10 lbs, so it works").
  - False urgency ("act now", "before it's banned").
  - Emotional manipulation (fear, shame, miracle framing).
If the claim is soberly worded, return an empty rhetorical_flags list.

Return your analysis strictly matching the provided schema.
"""

# Enum value lists are pulled from the models so the schema can never drift from
# the Verdict / EvidenceStance definitions.
_VERDICT_VALUES = [v.value for v in Verdict]
_STANCE_VALUES = [s.value for s in EvidenceStance]
# The model only ever chooses between direct and indirect; non_human is stamped
# by retrieval and unknown is the pre-assessment default.
_APPLICABILITY_VALUES = [
    EvidenceApplicability.DIRECT.value,
    EvidenceApplicability.INDIRECT.value,
]

_EVAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "fact_verdicts": {
            "type": "array",
            "description": "One entry per atomic fact, in any order (keyed by fact_index).",
            "items": {
                "type": "object",
                "properties": {
                    "fact_index": {
                        "type": "integer",
                        "description": "The 0-based index of the atomic fact this verdict is for.",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": _VERDICT_VALUES,
                        "description": "One of the seven categorical verdicts.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "2-4 sentences citing specific evidence by source and tier.",
                    },
                    "evidence_stances": {
                        "type": "array",
                        "description": "Stance for each evidence item, keyed by its evidence_index.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_index": {
                                    "type": "integer",
                                    "description": "0-based index of the evidence item within this fact.",
                                },
                                "stance": {
                                    "type": "string",
                                    "enum": _STANCE_VALUES,
                                },
                                "applicability": {
                                    "type": "string",
                                    "enum": _APPLICABILITY_VALUES,
                                    "description": (
                                        "direct: same form, human population, tested "
                                        "outcome; indirect: related but different "
                                        "form/dose/population/outcome."
                                    ),
                                },
                            },
                            "required": ["evidence_index", "stance", "applicability"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["fact_index", "verdict", "reasoning", "evidence_stances"],
                "additionalProperties": False,
            },
        },
        "rhetorical_flags": {
            "type": "array",
            "description": "Rhetorical red flags in the original claim text (empty if none).",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "explanation": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
                "required": ["pattern", "explanation", "excerpt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["fact_verdicts", "rhetorical_flags"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate_claim(
    claim_extraction: ClaimExtractionResult,
    fact_evidence_list: list[FactEvidence],
    allow_over_budget: bool = False,
) -> list[AtomicVerdict]:
    """Evaluate each atomic fact against its evidence and return per-fact verdicts.

    This is the spec'd entry point. It runs the full evaluation and hands back
    just the list of `AtomicVerdict`. Use `evaluate` instead when you also need
    the claim-level rhetorical flags (the dossier builder does).

    Args:
        claim_extraction: The extractor's result (primary claim + atomic facts).
        fact_evidence_list: Retrieved evidence, one `FactEvidence` per atomic fact.
        allow_over_budget: Forwarded to the single premium LLM call.

    Returns:
        One `AtomicVerdict` per atomic fact (empty if there was no claim).

    Raises:
        providers.BudgetWarning: if the daily budget is exhausted and
            `allow_over_budget` is False. (Non-budget LLM errors degrade to
            Insufficient-Evidence verdicts rather than raising.)
    """
    return evaluate(
        claim_extraction, fact_evidence_list, allow_over_budget=allow_over_budget
    ).verdicts


def evaluate(
    claim_extraction: ClaimExtractionResult,
    fact_evidence_list: list[FactEvidence],
    allow_over_budget: bool = False,
) -> ClaimEvaluation:
    """Full evaluation in ONE batched premium call: verdicts + rhetorical flags.

    Returns an empty evaluation when there is no evaluable claim. On a non-budget
    LLM failure it degrades to Insufficient-Evidence verdicts (with no flags) so a
    transient error never crashes the pipeline; `BudgetWarning` still propagates
    so the caller can make a deliberate spend decision.
    """
    if not claim_extraction.claim_found or not fact_evidence_list:
        return ClaimEvaluation(verdicts=[], rhetorical_flags=[])

    prompt = _build_prompt(claim_extraction, fact_evidence_list)
    try:
        result = providers.llm_call(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            model_tier="premium",
            response_schema=_EVAL_SCHEMA,
            allow_over_budget=allow_over_budget,
        )
    except providers.BudgetWarning:
        raise  # deliberate control-flow signal — let the caller decide
    except Exception as exc:  # noqa: BLE001 — never let one bad call sink the run
        logger.warning(
            "Verdict evaluation failed (%s); degrading to Insufficient Evidence.", exc
        )
        return ClaimEvaluation(
            verdicts=fallback_verdicts(fact_evidence_list), rhetorical_flags=[]
        )

    assert isinstance(result, dict)  # structured call always returns a dict
    verdicts = _assemble_verdicts(fact_evidence_list, result.get("fact_verdicts", []))
    flags = _assemble_flags(
        result.get("rhetorical_flags", []),
        original_text=f"{claim_extraction.original_text} {claim_extraction.primary_claim}",
    )
    return ClaimEvaluation(verdicts=verdicts, rhetorical_flags=flags)


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def _build_prompt(
    claim_extraction: ClaimExtractionResult, fact_evidence_list: list[FactEvidence]
) -> str:
    """Render the claim, atomic facts, and their evidence into one batched prompt."""
    lines: list[str] = [
        f"PRIMARY CLAIM: {claim_extraction.primary_claim}",
        "",
        "Evaluate each atomic fact below against ONLY the evidence listed under it.",
        "",
    ]
    for i, fe in enumerate(fact_evidence_list):
        fact = fe.atomic_fact
        lines.append(f"FACT {i} [{fact.fact_type.value}]: {fact.text}")
        lines.append(f'  original phrasing: "{fact.original_context}"')
        if fe.retrieval_note:
            lines.append(f"  retrieval note: {fe.retrieval_note}")
        if not fe.evidence:
            lines.append("  EVIDENCE: none found")
        else:
            lines.append("  EVIDENCE:")
            for j, ev in enumerate(fe.evidence):
                lines.append(f"    [{j}] {_evidence_line(ev)}")
        lines.append("")
    return "\n".join(lines)


def _evidence_line(ev: Evidence) -> str:
    """One compact line describing a piece of evidence for the prompt."""
    snippet = (ev.summary or ev.content or "").strip()
    if len(snippet) > _EVIDENCE_SNIPPET_CHARS:
        snippet = snippet[:_EVIDENCE_SNIPPET_CHARS].rstrip() + "…"
    pub = ev.publication_date.date().isoformat() if ev.publication_date else "n.d."
    tag = " [NON-HUMAN STUDY]" if ev.applicability is EvidenceApplicability.NON_HUMAN else ""
    return f"(T{ev.source_tier}) {ev.source_name} ({pub}){tag}: {snippet}"


# --------------------------------------------------------------------------- #
# Mapping the structured output back onto typed models
# --------------------------------------------------------------------------- #
def _assemble_verdicts(
    fact_evidence_list: list[FactEvidence], raw_verdicts: list[dict]
) -> list[AtomicVerdict]:
    """Turn the model's fact_verdicts into typed AtomicVerdicts, defensively.

    We drive the loop off the *input* facts (not the model output) so the result
    always has exactly one verdict per fact, in order, even if the model skips,
    reorders, or duplicates entries.
    """
    by_index = {}
    for raw in raw_verdicts:
        idx = raw.get("fact_index")
        if isinstance(idx, int) and idx not in by_index:
            by_index[idx] = raw

    verdicts: list[AtomicVerdict] = []
    for i, fe in enumerate(fact_evidence_list):
        raw = by_index.get(i)
        if raw is None:
            logger.warning("No verdict returned for fact %d; marking Insufficient.", i)
            verdicts.append(
                AtomicVerdict(
                    atomic_fact=fe.atomic_fact,
                    verdict=Verdict.INSUFFICIENT_EVIDENCE,
                    neutral_evidence=list(fe.evidence),
                    reasoning="The evaluator returned no verdict for this fact.",
                )
            )
            continue
        verdicts.append(_build_verdict(fe, raw))
    return verdicts


def _build_verdict(fe: FactEvidence, raw: dict) -> AtomicVerdict:
    """Build one AtomicVerdict, bucketing its evidence by assigned stance.

    Two guardrails are enforced here in code rather than trusted to the prompt:
      * animal / in-vitro evidence is always NEUTRAL (the animal-study filter);
      * a "Strongly" verdict needs at least one DIRECT item pointing that way,
        otherwise it is downgraded to the "Partially" verdict (the applicability
        cap). The reasoning is annotated so the reader sees why.
    """
    stance_by_idx: dict[int, str] = {}
    applicability_by_idx: dict[int, str] = {}
    for s in raw.get("evidence_stances", []):
        idx = s.get("evidence_index")
        if isinstance(idx, int):
            stance_by_idx[idx] = s.get("stance", EvidenceStance.NEUTRAL.value)
            applicability_by_idx[idx] = s.get("applicability", "")

    supporting: list[Evidence] = []
    opposing: list[Evidence] = []
    neutral: list[Evidence] = []
    for j, ev in enumerate(fe.evidence):
        stance = _coerce_stance(stance_by_idx.get(j))
        applicability = _resolve_applicability(ev, applicability_by_idx.get(j))
        if applicability is EvidenceApplicability.NON_HUMAN:
            stance = EvidenceStance.NEUTRAL  # rats never support a human claim
        stamped = ev.model_copy(
            update={"evidence_stance": stance, "applicability": applicability}
        )
        if stance is EvidenceStance.SUPPORTING:
            supporting.append(stamped)
        elif stance is EvidenceStance.OPPOSING:
            opposing.append(stamped)
        else:
            neutral.append(stamped)

    verdict = _coerce_verdict(raw.get("verdict"))
    reasoning = (raw.get("reasoning") or "").strip()
    verdict, reasoning = _apply_applicability_cap(verdict, reasoning, supporting, opposing)

    return AtomicVerdict(
        atomic_fact=fe.atomic_fact,
        verdict=verdict,
        supporting_evidence=supporting,
        opposing_evidence=opposing,
        neutral_evidence=neutral,
        reasoning=reasoning,
    )


def _resolve_applicability(ev: Evidence, raw_value: str | None) -> EvidenceApplicability:
    """Retrieval's NON_HUMAN stamp is authoritative; otherwise take the model's call."""
    if ev.applicability is EvidenceApplicability.NON_HUMAN:
        return EvidenceApplicability.NON_HUMAN
    try:
        value = EvidenceApplicability(raw_value)
    except ValueError:
        return EvidenceApplicability.INDIRECT  # unrated evidence never counts as direct
    if value is EvidenceApplicability.UNKNOWN:
        return EvidenceApplicability.INDIRECT
    return value


_STRONG_TO_PARTIAL = {
    Verdict.STRONGLY_SUPPORTED: Verdict.PARTIALLY_SUPPORTED,
    Verdict.STRONGLY_REFUTED: Verdict.PARTIALLY_REFUTED,
}
_CAP_NOTE = (
    " (Verdict capped at \"{partial}\": none of the {stance} evidence tested the "
    "claim's own form and population directly.)"
)


def _apply_applicability_cap(
    verdict: Verdict,
    reasoning: str,
    supporting: list[Evidence],
    opposing: list[Evidence],
) -> tuple[Verdict, str]:
    """Downgrade a Strongly verdict that rests only on indirect evidence."""
    partial = _STRONG_TO_PARTIAL.get(verdict)
    if partial is None:
        return verdict, reasoning
    pool = supporting if verdict is Verdict.STRONGLY_SUPPORTED else opposing
    if any(e.applicability is EvidenceApplicability.DIRECT for e in pool):
        return verdict, reasoning
    logger.info("Capping %s -> %s: no direct evidence.", verdict.value, partial.value)
    stance = "supporting" if verdict is Verdict.STRONGLY_SUPPORTED else "opposing"
    return partial, reasoning + _CAP_NOTE.format(partial=partial.value, stance=stance)


def _coerce_stance(value: str | None) -> EvidenceStance:
    try:
        return EvidenceStance(value)
    except ValueError:
        return EvidenceStance.NEUTRAL


def _coerce_verdict(value: str | None) -> Verdict:
    try:
        return Verdict(value)
    except ValueError:
        logger.warning("Unrecognized verdict %r; defaulting to Insufficient.", value)
        return Verdict.INSUFFICIENT_EVIDENCE


def _assemble_flags(raw_flags: list[dict], original_text: str = "") -> list[RhetoricalFlag]:
    """Map raw flags to typed ones, keeping only those bound to the claim text.

    A flag is evidence-bound when its `excerpt` actually appears in the original
    text (case- and whitespace-insensitive). Flags with no excerpt, or with an
    excerpt the model paraphrased or invented, are dropped: a red flag the reader
    can't find in the claim is an accusation, not an observation.
    """
    haystack = _normalize_for_match(original_text)
    flags: list[RhetoricalFlag] = []
    for f in raw_flags:
        pattern = (f.get("pattern") or "").strip()
        excerpt = (f.get("excerpt") or "").strip()
        if not pattern:
            continue
        needle = _normalize_for_match(excerpt)
        if not needle or (haystack and needle not in haystack):
            logger.info("Dropping unbound rhetorical flag %r (excerpt %r).", pattern, excerpt)
            continue
        flags.append(
            RhetoricalFlag(
                pattern=pattern,
                explanation=(f.get("explanation") or "").strip(),
                excerpt=excerpt,
            )
        )
    return flags


_QUOTE_CHARS = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _normalize_for_match(text: str) -> str:
    """Lowercase, fold curly quotes, and collapse whitespace for substring checks."""
    return re.sub(r"\s+", " ", (text or "").translate(_QUOTE_CHARS)).strip().lower()


def fallback_verdicts(fact_evidence_list: list[FactEvidence]) -> list[AtomicVerdict]:
    """Insufficient-Evidence verdicts used when the premium call can't run.

    Public because the orchestrator uses the same degradation when the token
    budget is exhausted at the verdict stage: the evidence is attached as
    neutral (unassessed) so the reader still sees what was found.
    """
    return [
        AtomicVerdict(
            atomic_fact=fe.atomic_fact,
            verdict=Verdict.INSUFFICIENT_EVIDENCE,
            neutral_evidence=list(fe.evidence),
            reasoning="Automated evaluation could not be completed for this fact "
            "(the reasoning model was unavailable); no verdict was reached.",
        )
        for fe in fact_evidence_list
    ]
