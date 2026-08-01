import type {
  AtomicVerdict,
  Dossier,
  Evidence,
  RhetoricalFlag,
  Verdict,
} from "../lib/types";

// --- Verdict badge colors (exact strings so Tailwind's scanner keeps them) ---
const VERDICT_BADGE: Record<Verdict, string> = {
  "Strongly Supported": "bg-green-100 text-green-800 border-green-300",
  "Partially Supported": "bg-emerald-50 text-emerald-700 border-emerald-200",
  "Insufficient Evidence": "bg-gray-100 text-gray-700 border-gray-300",
  "Conflicting Evidence": "bg-yellow-100 text-yellow-800 border-yellow-300",
  "Partially Refuted": "bg-orange-100 text-orange-800 border-orange-300",
  "Strongly Refuted": "bg-red-100 text-red-800 border-red-300",
  "Too Vague to Evaluate": "bg-gray-100 text-gray-600 border-gray-300",
};

// Left accent stripe on each verdict card.
const VERDICT_ACCENT: Record<Verdict, string> = {
  "Strongly Supported": "border-l-green-500",
  "Partially Supported": "border-l-emerald-400",
  "Insufficient Evidence": "border-l-gray-400",
  "Conflicting Evidence": "border-l-yellow-400",
  "Partially Refuted": "border-l-orange-400",
  "Strongly Refuted": "border-l-red-500",
  "Too Vague to Evaluate": "border-l-gray-400",
};

// --- Source tier badges: T1 dark blue -> T4 gray ---
const TIER_BADGE: Record<number, string> = {
  1: "bg-blue-900 text-white",
  2: "bg-blue-600 text-white",
  3: "bg-blue-200 text-blue-900",
  4: "bg-gray-300 text-gray-800",
};

const TIER_LABEL: Record<number, string> = {
  1: "Systematic reviews & official guidelines (Cochrane, WHO, CDC, meta-analyses)",
  2: "Peer-reviewed studies & major medical institutions (PubMed, NIH, Mayo)",
  3: "Credentialed health journalism & university health centers",
  4: "General web / social media — context only, not evidence",
};

const STANCE_STYLE: Record<string, { label: string; className: string }> = {
  supporting: { label: "Supports", className: "text-green-700 bg-green-50" },
  opposing: { label: "Refutes", className: "text-red-700 bg-red-50" },
  neutral: { label: "Context", className: "text-gray-600 bg-gray-100" },
};

function TierBadge({ tier }: { tier: number }) {
  const cls = TIER_BADGE[tier] ?? TIER_BADGE[4];
  return (
    <span
      title={TIER_LABEL[tier] ?? ""}
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-semibold ${cls}`}
    >
      T{tier}
    </span>
  );
}

function formatDate(iso: string | null): string {
  if (!iso) return "n.d.";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "n.d." : d.toISOString().slice(0, 10);
}

function EvidenceRow({ ev }: { ev: Evidence }) {
  const stance = STANCE_STYLE[ev.evidence_stance] ?? STANCE_STYLE.neutral;
  return (
    <li className="rounded-md border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <TierBadge tier={ev.source_tier} />
        <span
          className={`rounded px-1.5 py-0.5 text-xs font-medium ${stance.className}`}
        >
          {stance.label}
        </span>
        <span className="font-medium text-foreground">{ev.source_name}</span>
        <span className="text-muted">· {formatDate(ev.publication_date)}</span>
        <span className="text-muted">· relevance {ev.relevance_score.toFixed(2)}</span>
      </div>
      {ev.summary && (
        <p className="mt-2 text-sm leading-relaxed text-gray-700">{ev.summary}</p>
      )}
      {ev.source_url && (
        <a
          href={ev.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-1 inline-block break-all text-xs text-blue-700 hover:underline"
        >
          {ev.source_url}
        </a>
      )}
    </li>
  );
}

function VerdictCard({ verdict, index }: { verdict: AtomicVerdict; index: number }) {
  const badge = VERDICT_BADGE[verdict.verdict] ?? VERDICT_BADGE["Insufficient Evidence"];
  const accent = VERDICT_ACCENT[verdict.verdict] ?? VERDICT_ACCENT["Insufficient Evidence"];
  // Cite all evidence, most authoritative first.
  const evidence = [
    ...verdict.supporting_evidence,
    ...verdict.opposing_evidence,
    ...verdict.neutral_evidence,
  ].sort((a, b) => a.source_tier - b.source_tier);

  return (
    <article
      className={`rounded-lg border border-border border-l-4 bg-surface p-5 shadow-sm ${accent}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <span className="text-xs font-medium uppercase tracking-wide text-muted">
            Claim {index + 1} · {verdict.atomic_fact.fact_type}
          </span>
          <h3 className="mt-1 text-base font-semibold leading-snug text-foreground">
            {verdict.atomic_fact.text}
          </h3>
        </div>
        <span
          className={`shrink-0 rounded-full border px-3 py-1 text-sm font-semibold ${badge}`}
        >
          {verdict.verdict}
        </span>
      </div>

      {verdict.reasoning && (
        <p className="mt-3 text-sm leading-relaxed text-gray-700">
          {verdict.reasoning}
        </p>
      )}

      {evidence.length > 0 ? (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            Evidence ({evidence.length})
          </p>
          <ul className="space-y-2">
            {evidence.map((ev, i) => (
              <EvidenceRow key={`${ev.source_url}-${i}`} ev={ev} />
            ))}
          </ul>
        </div>
      ) : (
        <p className="mt-4 text-sm italic text-muted">No evidence retrieved for this fact.</p>
      )}
    </article>
  );
}

function FlagCard({ flag }: { flag: RhetoricalFlag }) {
  return (
    <div className="rounded-md border border-amber-300 bg-amber-50 p-3">
      <div className="flex items-center gap-2">
        <span aria-hidden className="text-amber-600">
          ⚑
        </span>
        <span className="font-semibold text-amber-900">{flag.pattern}</span>
      </div>
      {flag.excerpt && (
        <p className="mt-1 text-sm italic text-amber-800">“{flag.excerpt}”</p>
      )}
      {flag.explanation && (
        <p className="mt-1 text-sm text-amber-900/90">{flag.explanation}</p>
      )}
    </div>
  );
}

export default function DossierView({ dossier }: { dossier: Dossier }) {
  const { claim_extraction: extraction } = dossier;

  return (
    <section className="w-full space-y-6">
      {/* Primary claim */}
      <header className="rounded-lg border border-border bg-surface p-5 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">
          Claim assessed
        </p>
        <h2 className="mt-1 text-lg font-semibold leading-snug text-foreground">
          {extraction.primary_claim || dossier.original_input}
        </h2>
      </header>

      {/* Rhetorical red flags */}
      {dossier.rhetorical_flags.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-800">
            Rhetorical red flags ({dossier.rhetorical_flags.length})
          </h3>
          {dossier.rhetorical_flags.map((flag, i) => (
            <FlagCard key={i} flag={flag} />
          ))}
        </div>
      )}

      {/* Per-atom verdicts */}
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-gray-800">
          Per-claim verdicts ({dossier.verdicts.length})
        </h3>
        {dossier.verdicts.map((v, i) => (
          <VerdictCard key={i} verdict={v} index={i} />
        ))}
      </div>

      {/* Narrative summary */}
      {dossier.narrative_summary && (
        <div className="rounded-lg border border-blue-200 bg-blue-50/60 p-5">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-blue-900">
            Summary
          </h3>
          <p className="text-sm leading-relaxed text-gray-800">
            {dossier.narrative_summary}
          </p>
        </div>
      )}

      {/* Subtle provenance */}
      <footer className="border-t border-border pt-3 text-xs text-muted">
        Dossier {dossier.id} · generated{" "}
        {new Date(dossier.timestamp).toLocaleString()}
      </footer>
    </section>
  );
}
