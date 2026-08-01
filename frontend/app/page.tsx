"use client";

import { useState } from "react";
import DossierView from "./components/DossierView";
import { ApiError, checkClaim, fileToDataUrl } from "./lib/api";
import type { CheckRequest, Dossier, InputMode } from "./lib/types";

const MODES: { id: InputMode; label: string; hint: string }[] = [
  { id: "text", label: "Text", hint: "Paste a health claim" },
  { id: "url", label: "URL", hint: "Link to an article" },
  { id: "image", label: "Screenshot", hint: "Upload a screenshot" },
];

export default function Home() {
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dossier, setDossier] = useState<Dossier | null>(null);

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
        <div className="mt-3 flex items-center justify-between">
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

      {/* Loading message */}
      {loading && (
        <p className="mt-6 text-center text-sm text-muted">
          Analyzing claim — decomposing into facts, retrieving evidence, and
          weighing sources. This can take up to a minute.
        </p>
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
          <DossierView dossier={dossier} />
        </div>
      )}

      <footer className="mt-auto pt-10 text-center text-xs text-muted">
        ClaimCheck AI · an evidence dossier builder, not a truth machine.
      </footer>
    </main>
  );
}
