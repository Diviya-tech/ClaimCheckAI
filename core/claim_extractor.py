"""Claim extractor — stage 2 of the pipeline.

Takes clean text and uses the *lightweight* model tier to:
  1. Identify the single primary health claim (if any).
  2. Decompose it into atomic, independently testable facts.
  3. Classify each fact (causal / quantitative / prescriptive / temporal / existential).
  4. Flag when there is no evaluable health claim at all.

Structured output is produced via tool_use (see config/providers.py), so the
model returns a dict guaranteed to match the schema below, which we then load
into a validated `ClaimExtractionResult`.

This decomposition is the project's core token-efficiency lever: it shrinks the
input for every downstream stage by 80-90%.
"""

from __future__ import annotations

from config import providers
from schemas.models import ClaimExtractionResult, SourceFormat

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = """\
You are the claim-extraction stage of an evidence-evaluation engine for health claims.

Your job is NOT to judge whether claims are true. Your job is to identify and
structure them precisely so a later stage can gather evidence.

Given a piece of text, do the following:

1. PRIMARY CLAIM: Identify the single most important health-related claim the
   text is making. If the text makes several, choose the central one and capture
   the others as atomic facts under it. State it as one clear sentence.

2. DECOMPOSE: Break the primary claim into atomic facts. An atomic fact is the
   smallest independently testable assertion — one that could be supported or
   refuted on its own by evidence. Split compound claims ("X cures Y and boosts Z")
   into separate facts. Do not invent facts the text does not assert.

   INVARIANT: Whenever claim_found is true you MUST return at least one atomic
   fact. A found claim that yields zero atomic facts is an invalid result. If a
   claim is real but you genuinely cannot break it into anything testable, emit a
   single atomic fact that restates the core assertion.

   SELF-CONTAINED: Each atomic fact must be self-contained — include the subject,
   substance, or intervention in EVERY fact. A fact must be independently
   understandable without reading the other facts, because each one is searched
   and evaluated on its own. Never use bare pronouns or references like "it",
   "they", "this", "the effect", or "the result" without specifying what they
   refer to. For example, from "Cumin water melts belly fat in two weeks" do NOT
   emit a temporal fact "the effect occurs within two weeks" — emit "Cumin water
   produces belly-fat loss within two weeks". A reader (or a search engine) must
   never have to guess what the fact is about.

   FORM & PREPARATION: When the claim names a specific preparation or form of a
   substance — a tea, water, juice, smoothie, powder, oil, gummy, topical cream,
   "raw", "organic" — the form is part of the claim and gets its OWN atomic
   fact, separate from the effect. The evidence base usually studies a
   standardized extract or capsule at a known dose, so a later stage must be
   able to judge "does turmeric TEA deliver what the turmeric-extract trials
   delivered?" independently of "does curcumin affect arthritis?". From
   "Turmeric tea cures arthritis" emit BOTH:
     - "Turmeric (curcumin) reduces arthritis symptoms." (causal)
     - "Turmeric tea delivers curcumin at a dose comparable to the amounts
       studied in clinical trials." (quantitative)
   Never collapse the form into the effect; a claim about a food or drink is not
   the same claim as one about a concentrated extract of it.

   HEDGED & LONG-FORM SOURCES: Articles and balanced journalism often wrap claims
   in uncertainty ("may help", "might reduce", "evidence is mixed", "modest
   effects", "some studies suggest"). Do NOT fold that hedging into a summary
   sentence and stop. Strip the hedging and extract the UNDERLYING factual
   assertion as the atomic fact (e.g. "green tea may boost metabolism" ->
   atomic fact "green tea boosts metabolism", type causal). The hedging is for
   the later evidence-evaluation stage to weigh — your job is to surface the
   testable assertion so it CAN be weighed. Capture the hedging phrasing in
   `original_context`, not in the fact text.

3. CLASSIFY each atomic fact as exactly one of:
   - "causal": asserts X causes, prevents, cures, or triggers Y.
   - "quantitative": asserts a number, magnitude, dose, rate, or percentage.
   - "prescriptive": tells the reader to do / take / avoid something.
   - "temporal": asserts timing, duration, or sequence ("within 2 weeks").
   - "existential": asserts something exists or is present ("contains compound X").

4. NO CLAIM: If the text contains no evaluable health claim (e.g. it is an
   opinion, a question, an ad with no factual assertion, or unrelated content),
   set claim_found to false, leave primary_claim empty, and return no atomic facts.

For each atomic fact include `original_context`: the phrase or sentence from the
source text it was distilled from. Be faithful to the source — never add claims.
"""

_USER_PROMPT_TEMPLATE = """\
Extract and decompose the primary health claim from the following text.

--- TEXT START ---
{text}
--- TEXT END ---
"""

# --------------------------------------------------------------------------- #
# Structured-output schema (drives tool_use in the provider layer)
# --------------------------------------------------------------------------- #
# This is the JSON Schema the model must fill in. It maps onto
# ClaimExtractionResult minus the fields we set ourselves (original_text,
# source_format).
_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claim_found": {
            "type": "boolean",
            "description": "False if the text contains no evaluable health claim.",
        },
        "primary_claim": {
            "type": "string",
            "description": "The single main health claim as one sentence, or empty string if none.",
        },
        "atomic_facts": {
            "type": "array",
            "description": "The primary claim decomposed into atomic testable facts.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The atomic, testable fact in plain language.",
                    },
                    "fact_type": {
                        "type": "string",
                        "enum": [
                            "causal",
                            "quantitative",
                            "prescriptive",
                            "temporal",
                            "existential",
                        ],
                        "description": "Classification of what the fact asserts.",
                    },
                    "original_context": {
                        "type": "string",
                        "description": "The source phrase this fact was distilled from.",
                    },
                },
                "required": ["text", "fact_type", "original_context"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claim_found", "primary_claim", "atomic_facts"],
    "additionalProperties": False,
}


def extract_claims(
    text: str,
    source_format: SourceFormat = SourceFormat.TEXT,
    allow_over_budget: bool = False,
) -> ClaimExtractionResult:
    """Extract and decompose the primary health claim from `text`.

    Args:
        text: Clean input text (already normalized by an input handler).
        source_format: Which input handler produced the text.
        allow_over_budget: Forwarded to the LLM layer's soft budget guardrail.

    Returns:
        A validated ClaimExtractionResult.

    Raises:
        providers.BudgetWarning: if the daily token budget is exhausted and
            `allow_over_budget` is False.
    """
    if not text or not text.strip():
        # Empty input can never contain a claim — no need to spend tokens.
        return ClaimExtractionResult(
            original_text=text,
            primary_claim="",
            atomic_facts=[],
            claim_found=False,
            source_format=source_format,
        )

    result = providers.llm_call(
        prompt=_USER_PROMPT_TEMPLATE.format(text=text),
        system_prompt=_SYSTEM_PROMPT,
        model_tier="lightweight",
        response_schema=_EXTRACTION_SCHEMA,
        allow_over_budget=allow_over_budget,
    )

    # `result` is the validated tool_use dict. Add the fields we own, then let
    # Pydantic do the final validation/coercion into the typed model.
    assert isinstance(result, dict)  # structured calls always return a dict
    return ClaimExtractionResult(
        original_text=text,
        source_format=source_format,
        **result,
    )


# --------------------------------------------------------------------------- #
# Example calls (run this module directly to smoke-test the extractor).
# Requires a real ANTHROPIC_API_KEY in .env.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    examples = [
        # 1. A compound causal + quantitative claim.
        "Drinking green tea every morning boosts your metabolism by 12% and "
        "melts away belly fat in just two weeks.",
        # 2. A prescriptive supplement claim.
        "Doctors don't want you to know this: taking 5000 IU of vitamin D daily "
        "prevents almost all winter colds.",
        # 3. No evaluable health claim (opinion / chit-chat).
        "Honestly, I just think mornings are the best time to relax with a cup of "
        "tea and read a good book.",
    ]

    for i, example in enumerate(examples, start=1):
        print(f"\n{'=' * 70}\nEXAMPLE {i}\n{'-' * 70}")
        print(f"INPUT: {example}\n")
        extraction = extract_claims(example)
        print(f"claim_found:   {extraction.claim_found}")
        print(f"primary_claim: {extraction.primary_claim or '(none)'}")
        for j, fact in enumerate(extraction.atomic_facts, start=1):
            print(f"  [{j}] ({fact.fact_type.value}) {fact.text}")
            print(f"      from: {fact.original_context!r}")
