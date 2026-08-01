// TypeScript mirrors of the Pydantic models the API returns
// (schemas/models.py). Kept intentionally close to the JSON shape.

export type Verdict =
  | "Strongly Supported"
  | "Partially Supported"
  | "Insufficient Evidence"
  | "Conflicting Evidence"
  | "Partially Refuted"
  | "Strongly Refuted"
  | "Too Vague to Evaluate";

export type EvidenceStance = "supporting" | "opposing" | "neutral";

export type FactType =
  | "causal"
  | "quantitative"
  | "prescriptive"
  | "temporal"
  | "existential";

export interface Evidence {
  content: string;
  summary: string;
  source_url: string;
  source_name: string;
  source_tier: number; // 1..4
  relevance_score: number; // 0..1
  evidence_stance: EvidenceStance;
  publication_date: string | null;
}

export interface AtomicFact {
  text: string;
  fact_type: FactType;
  original_context: string;
}

export interface AtomicVerdict {
  atomic_fact: AtomicFact;
  verdict: Verdict;
  supporting_evidence: Evidence[];
  opposing_evidence: Evidence[];
  neutral_evidence: Evidence[];
  reasoning: string;
}

export interface RhetoricalFlag {
  pattern: string;
  explanation: string;
  excerpt: string;
}

export interface ClaimExtractionResult {
  original_text: string;
  primary_claim: string;
  atomic_facts: AtomicFact[];
  claim_found: boolean;
  source_format: string;
}

export interface Dossier {
  id: string;
  timestamp: string;
  original_input: string;
  claim_extraction: ClaimExtractionResult;
  verdicts: AtomicVerdict[];
  rhetorical_flags: RhetoricalFlag[];
  narrative_summary: string;
}

// The three input modes the UI offers (video is supported by the API too,
// but the minimal UI focuses on the three most common consumer inputs).
export type InputMode = "text" | "url" | "image";

export interface CheckRequest {
  text?: string;
  url?: string;
  image?: string; // base64 data URL
  video_url?: string;
}
