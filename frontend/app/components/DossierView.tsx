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

// --- Source tier badges: tier 1 dark blue -> tier 4 gray ---
const TIER_BADGE: Record<number, string> = {
  1: "bg-blue-900 text-white",
  2: "bg-blue-600 text-white",
  3: "bg-blue-200 text-blue-900",
  4: "bg-gray-300 text-gray-800",
};

// Tooltip text. The badge itself carries the API's `human_readable_tier`; this
// is the longer "why does this rank here" line for anyone who hovers.
const TIER_TITLE: Record<number, string> = {
  1: "Systematic reviews & official guidelines (Cochrane, WHO, CDC, meta-analyses)",
  2: "Peer-reviewed studies & major medical institutions (PubMed, NIH, Mayo)",
  3: "Credentialed health journalism & university health centers",
  4: "General web / social media — context only, not evidence",
};

// Fallback label, only used if an older API build omits `human_readable_tier`.
const TIER_FALLBACK_LABEL: Record<number, string> = {
  1: "Systematic Review / Meta-analysis",
  2: "Peer-reviewed Study",
  3: "Medical Journalism",
  4: "General Web Source",
};

// Plain-language explanation of the tier hierarchy, shown in the collapsed
// "Why is this considered high-quality evidence?" section below the verdicts.
const TIER_EXPLAINER: { name: string; tier: number; text: string }[] = [
  {
    name: "Systematic Reviews and Meta-analyses",
    tier: 1,
    text: "(highest) These analyze multiple studies together. One systematic review represents the combined findings of many individual studies.",
  },
  {
    name: "Peer-reviewed Studies",
    tier: 2,
    text: "Individual studies published in medical journals and reviewed by other scientists before publication.",
  },
  {
    name: "Medical Journalism",
    tier: 3,
    text: "Reporting by credentialed health journalists or university health centers. Useful for context but not primary evidence.",
  },
  {
    name: "General Web Sources",
    tier: 4,
    text: "Blogs, social media, and unverified web content. Shown for context only — not weighted as evidence.",
  },
];

const STANCE_STYLE: Record<string, { label: string; className: string }> = {
  supporting: { label: "Supports", className: "text-green-700 bg-green-50" },
  opposing: { label: "Refutes", className: "text-red-700 bg-red-50" },
  neutral: { label: "Context", className: "text-gray-600 bg-gray-100" },
};

// Applicability is shown only when it qualifies the evidence: "direct" is the
// unremarkable default, so it gets no badge.
const APPLICABILITY_STYLE: Record<string, { label: string; title: string; className: string }> = {
  indirect: {
    label: "Indirect",
    title: "Related evidence, but a different form, dose, population, or outcome than the claim",
    className: "text-amber-800 bg-amber-50 border border-amber-200",
  },
  non_human: {
    label: "Animal / in-vitro",
    title: "Studied in animals or cells — context for a human claim, never support or refutation",
    className: "text-purple-800 bg-purple-50 border border-purple-200",
  },
};

function TierBadge({ tier, label }: { tier: number; label?: string }) {
  const cls = TIER_BADGE[tier] ?? TIER_BADGE[4];
  return (
    <span
      title={TIER_TITLE[tier] ?? ""}
      className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-semibold ${cls}`}
    >
      {label || TIER_FALLBACK_LABEL[tier] || TIER_FALLBACK_LABEL[4]}
    </span>
  );
}

/* Collapsed by default: the tier names are self-explanatory at a glance, and
   this is here for the reader who wants to know why one badge outranks another.
   Native <details> keeps it keyboard-accessible with no client-side state. */
function EvidenceQualityNote() {
  return (
    <details className="group rounded-lg border border-border bg-surface px-4 py-3">
      <summary className="cursor-pointer list-none text-sm font-medium text-blue-800 hover:underline [&::-webkit-details-marker]:hidden">
        <span aria-hidden className="mr-1.5 inline-block transition-transform group-open:rotate-90">
          ›
        </span>
        Why is this considered high-quality evidence?
      </summary>
      <div className="mt-3 border-t border-border pt-3">
        <p className="text-sm text-gray-700">
          ClaimCheck rates evidence quality on four tiers:
        </p>
        <ul className="mt-2 space-y-2">
          {TIER_EXPLAINER.map((t) => (
            <li key={t.tier} className="flex flex-wrap items-baseline gap-x-2 text-sm">
              <TierBadge tier={t.tier} label={t.name} />
              <span className="flex-1 leading-relaxed text-gray-700">{t.text}</span>
            </li>
          ))}
        </ul>
      </div>
    </details>
  );
}

function formatDate(iso: string | null): string {
  if (!iso) return "n.d.";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "n.d." : d.toISOString().slice(0, 10);
}

function EvidenceRow({ ev }: { ev: Evidence }) {
  const stance = STANCE_STYLE[ev.evidence_stance] ?? STANCE_STYLE.neutral;
  const applicability = APPLICABILITY_STYLE[ev.applicability];
  return (
    <li className="min-w-0 rounded-md border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <TierBadge tier={ev.source_tier} label={ev.human_readable_tier} />
        <span
          className={`rounded px-1.5 py-0.5 text-xs font-medium ${stance.className}`}
        >
          {stance.label}
        </span>
        {applicability && (
          <span
            title={applicability.title}
            className={`rounded px-1.5 py-0.5 text-xs font-medium ${applicability.className}`}
          >
            {applicability.label}
          </span>
        )}
        <span className="min-w-0 break-words font-medium text-foreground">{ev.source_name}</span>
        <span className="text-muted">· {formatDate(ev.publication_date)}</span>
        <span className="text-muted">· relevance {ev.relevance_score.toFixed(2)}</span>
      </div>
      {ev.summary && (
        <p className="mt-2 break-words text-sm leading-relaxed text-gray-700">{ev.summary}</p>
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
      className={`min-w-0 rounded-lg border border-border border-l-4 bg-surface p-4 shadow-sm sm:p-5 ${accent}`}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
        <div className="min-w-0">
          <span className="text-xs font-medium uppercase tracking-wide text-muted">
            Claim {index + 1} · {verdict.atomic_fact.fact_type}
          </span>
          <h3 className="mt-1 break-words text-base font-semibold leading-snug text-foreground">
            {verdict.atomic_fact.text}
          </h3>
        </div>
        <span
          className={`self-start whitespace-nowrap rounded-full border px-3 py-1 text-sm font-semibold sm:shrink-0 ${badge}`}
        >
          {verdict.verdict}
        </span>
      </div>

      {verdict.reasoning && (
        <p className="mt-3 break-words text-sm leading-relaxed text-gray-700">
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

function ResetButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-4 py-2 text-sm font-medium text-gray-800 transition-colors hover:border-blue-400 hover:text-blue-800"
    >
      <span aria-hidden>↺</span> Try another claim
    </button>
  );
}

/* What degraded THIS run (a source that was down, the budget cutting the
   verdict stage short). Shown prominently — a reader weighing the verdicts
   needs to know what they're missing before they read the summary. */
function LimitationsPanel({ notes }: { notes: string[] }) {
  if (notes.length === 0) return null;
  return (
    <div
      role="note"
      className="rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-900"
    >
      <p className="font-semibold">Limitations of this run</p>
      <ul className="mt-1.5 list-disc space-y-1 pl-5">
        {notes.map((n, i) => (
          <li key={i} className="break-words">{n}</li>
        ))}
      </ul>
    </div>
  );
}

export default function DossierView({
  dossier,
  onReset,
}: {
  dossier: Dossier;
  onReset?: () => void;
}) {
  const { claim_extraction: extraction } = dossier;
  const limitations = dossier.limitations ?? [];

  return (
    <section className="w-full min-w-0 space-y-6">
      {/* Primary claim */}
      <header className="rounded-lg border border-border bg-surface p-4 shadow-sm sm:p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted">
              Claim assessed
            </p>
            <h2 className="mt-1 break-words text-lg font-semibold leading-snug text-foreground">
              {extraction.primary_claim || dossier.original_input}
            </h2>
          </div>
          {onReset && (
            <div className="sm:shrink-0">
              <ResetButton onClick={onReset} />
            </div>
          )}
        </div>
      </header>

      <LimitationsPanel notes={limitations} />

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
        {/* One explainer for every evidence list above, rather than repeating
            the same four tiers under each verdict card. */}
        <EvidenceQualityNote />
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

      {/* Subtle provenance + a second way out at the bottom of a long page */}
      <footer className="flex flex-col gap-3 border-t border-border pt-3 text-xs text-muted sm:flex-row sm:items-center sm:justify-between">
        <span className="break-all">
          Dossier {dossier.id} · generated{" "}
          {new Date(dossier.timestamp).toLocaleString()}
        </span>
        {onReset && <ResetButton onClick={onReset} />}
      </footer>
    </section>
  );
}
