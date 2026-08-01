import type { CheckRequest, Dossier } from "./types";

// The FastAPI backend. Override at build/run time with NEXT_PUBLIC_API_BASE.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data?.detail) {
      return typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail);
    }
  } catch {
    // fall through to a generic message
  }
  return `Request failed (${res.status} ${res.statusText}).`;
}

/** POST /api/check — run the full pipeline and return the evidence dossier. */
export async function checkClaim(input: CheckRequest): Promise<Dossier> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    });
  } catch {
    throw new ApiError(
      `Could not reach the ClaimCheck API at ${API_BASE}. Is the backend running (uvicorn api.server:app --reload)?`,
      0,
    );
  }

  if (!res.ok) {
    throw new ApiError(await readErrorDetail(res), res.status);
  }
  return (await res.json()) as Dossier;
}

/** Read a File as a base64 data URL (e.g. "data:image/png;base64,...."). */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Could not read the image file."));
    reader.readAsDataURL(file);
  });
}
