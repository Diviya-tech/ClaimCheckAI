"use client";

import { useEffect, useRef, useState } from "react";
import DossierView from "./components/DossierView";
import { ApiError, checkClaim, fileToDataUrl } from "./lib/api";
import type { CheckRequest, Dossier, InputMode } from "./lib/types";

const MODES: { id: InputMode; label: string; hint: string }[] = [
  { id: "text", label: "Text", hint: "Paste a health claim" },
  { id: "url", label: "URL", hint: "Link to an article" },
  { id: "image", label: "Screenshot", hint: "Upload a screenshot" },
];

// One-click starters. Chosen to show the range of verdicts the engine
// produces, not to flatter it: a well-supported claim, a form-mismatch claim,
// a correlation-vs-causation claim, and a classic social-media miracle.
const EXAMPLE_CLAIMS = [
  "Vaccines prevent measles",
  "Turmeric tea cures arthritis",
  "People who drink coffee live longer",
  "Cumin water melts belly fat in two weeks",
];

// The API is a single request, so the UI can't observe stage boundaries. The
// stages below advance on wall-clock estimates that track the real pipeline
// (extraction ~3s on the lightweight model; retrieval ~10-20s of PubMed/Tavily
// traffic; one batched premium verdict call; one narrative call). If a stage
// runs long the label simply waits on the last one — it never claims "done".
const LOADING_STAGES: { label: string; at: number }[] = [
  { label: "Extracting claim…", at: 0 },
  { label: "Searching evidence…", at: 4_000 },
  { label: "Evaluating verdicts…", at: 22_000 },
  { label: "Building dossier…", at: 40_000 },
];

function useLoadingStage(loading: boolean): number {
  // `stage` only advances via timers while a request is in flight; when the
  // request ends the timers are cleared and the value is ignored (the caller
  // reads 0), so a new request always starts from the first stage.
  const [stage, setStage] = useState(0);
  useEffect(() => {
    if (!loading) return;
    const timers = [
      setTimeout(() => setStage(0), 0),
      ...LOADING_STAGES.slice(1).map((s, i) =>
        setTimeout(() => setStage(i + 1), s.at),
      ),
    ];
    return () => timers.forEach(clearTimeout);
  }, [loading]);
  return loading ? stage : 0;
}

export default function Home() {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dossier, setDossier] = useState<Dossier | null>(null);

  const stage = useLoadingStage(loading);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function switchMode(next: InputMode) {
    setMode(next);
    setError(null);
  }

  async function buildRequest(): Promise<CheckRequest> {
    if (mode === "url") return { url: url.trim() };
    if (mode === "image") {
      if (!imageFile) throw new Error("Choose a screenshot to analyze.");
      return { image: await fileToDataUrl(imageFile) };
    }
    return { text: text.trim() };
  }

  function inputIsEmpty(): boolean {
    if (mode === "url") return url.trim() === "";
    if (mode === "image") return imageFile === null;
    return text.trim() === "";
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (loading || inputIsEmpty()) return;

    setLoading(true);
    setError(null);
    setDossier(null);
    try {
      const req = await buildRequest();
      const result = await checkClaim(req);
      setDossier(result);
    } catch (err) {
      if (err instanceof ApiError && err.status === 422) {
        setError(
          "No evaluable health claim was found in that input. Try a more specific health claim.",
        );
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  function fillExample(claim: string) {
    setMode("text");
    setText(claim);
    setError(null);
    textareaRef.current?.focus();
  }

  function reset() {
    setDossier(null);
    setError(null);
    setText("");
    setUrl("");
    setImageFile(null);
    setMode("text");
    window.scrollTo({ top: 0, behavior: "smooth" });
    // Focus after the scroll starts so the caret lands in the visible box.
    setTimeout(() => textareaRef.current?.focus(), 150);
  }

  return (
    <main className="mx-auto flex min-h-full w-full max-w-3xl flex-col px-4 py-10 sm:py-16">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
          ClaimCheck AI
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-muted sm:text-base">
          Paste a health claim and get a transparent evidence dossier — atomic
          facts, per-fact verdicts, and cited medical sources.{" "}
          <span className="whitespace-nowrap">Not a trust score.</span>
        </p>
      </div>

      {/* Input card */}
      <form
        onSubmit={handleSubmit}
        className="mt-8 rounded-2xl border border-border bg-surface p-4 shadow-sm"
      >
        {/* Mode selector */}
        <div className="mb-3 inline-flex rounded-lg bg-gray-100 p-1 text-sm">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => switchMode(m.id)}
              className={`rounded-md px-3 py-1.5 font-medium transition-colors ${
                mode === m.id
                  ? "bg-surface text-foreground shadow-sm"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Input area (one at a time — like a search bar, not a form) */}
        {mode === "text" && (
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. Cumin water melts belly fat in two weeks"
            rows={3}
            disabled={loading}
            className="w-full resize-none rounded-lg border border-border bg-white px-4 py-3 text-base text-foreground outline-none placeholder:text-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-60"
          />
        )}

        {mode === "url" && (
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/some-health-article"
            disabled={loading}
            className="w-full rounded-lg border border-border bg-white px-4 py-3 text-base text-foreground outline-none placeholder:text-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-60"
          />
        )}

        {mode === "image" && (
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-border bg-white px-4 py-8 text-center hover:border-blue-400">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={loading}
              onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
              className="hidden"
            />
            <span className="text-sm font-medium text-foreground">
              {imageFile ? imageFile.name : "Click to upload a screenshot"}
            </span>
            <span className="mt-1 text-xs text-muted">PNG, JPG, or WEBP</span>
          </label>
        )}

        {/* Submit */}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
          <span className="text-xs text-muted">
            {MODES.find((m) => m.id === mode)?.hint}
          </span>
          <button
            type="submit"
            disabled={loading || inputIsEmpty()}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
          >
            {loading && (
              <span
                aria-hidden
                className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
              />
            )}
            {loading ? "Analyzing…" : "Check claim"}
          </button>
        </div>
      </form>

      {/* Example claims — only while there's nothing else on screen */}
      {!loading && !dossier && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs text-muted">Try:</span>
          {EXAMPLE_CLAIMS.map((claim) => (
            <button
              key={claim}
              type="button"
              onClick={() => fillExample(claim)}
              className="rounded-full border border-border bg-surface px-3 py-1 text-xs text-gray-700 transition-colors hover:border-blue-400 hover:text-blue-800"
            >
              {claim}
            </button>
          ))}
        </div>
      )}

      {/* Staged loading indicator */}
      {loading && (
        <div
          role="status"
          aria-live="polite"
          className="mt-8 rounded-lg border border-border bg-surface p-5 text-center shadow-sm"
        >
          <div className="mx-auto h-8 w-8 animate-spin rounded-full border-[3px] border-blue-100 border-t-blue-600" />
          <p className="mt-3 text-sm font-medium text-foreground">
            {LOADING_STAGES[stage].label}
          </p>
          <ol className="mx-auto mt-3 flex max-w-md flex-wrap justify-center gap-x-4 gap-y-1 text-xs">
            {LOADING_STAGES.map((s, i) => (
              <li
                key={s.label}
                className={
                  i < stage
                    ? "text-green-700"
                    : i === stage
                      ? "font-semibold text-blue-700"
                      : "text-gray-400"
                }
              >
                {i < stage ? "✓ " : `${i + 1}. `}
                {s.label.replace("…", "")}
              </li>
            ))}
          </ol>
          <p className="mt-3 text-xs text-muted">
            Searching PubMed and weighing sources can take up to a minute.
          </p>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Result */}
      {dossier && !loading && (
        <div className="mt-8">
          <DossierView dossier={dossier} onReset={reset} />
        </div>
      )}

      <footer className="mt-auto space-y-1 pt-10 text-center text-xs text-muted">
        <p>Evidence sourced from PubMed, WHO, CDC. ClaimCheck AI does not provide medical advice.</p>
        <p>An evidence dossier builder, not a truth machine.</p>
      </footer>
    </main>
  );
}
