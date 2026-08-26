import type { AskResponse, AskResult } from "./ask-types";

const BACKEND_BASE_URL = "https://furiosa-multimodal-rag.onrender.com";
const ASK_API_URL = `${BACKEND_BASE_URL}/ask`;
const HEALTH_API_URL = `${BACKEND_BASE_URL}/health`;

// Hosted inference can take tens of seconds (routing + generation),
// and a sleeping Render instance needs time for a cold start.
const REQUEST_TIMEOUT_MS = 180_000;
const HEALTH_TIMEOUT_MS = 120_000;

export async function askBackend(question: string, documentId: string): Promise<AskResult> {
  try {
    const res = await fetch(ASK_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_id: documentId }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });

    if (!res.ok) {
      let detail: string | undefined;
      try {
        const body = (await res.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        // Keep upstream implementation details out of the browser response.
      }
      return detail ? { ok: false, status: res.status, detail } : { ok: false, status: res.status };
    }

    const data = (await res.json()) as AskResponse;
    return { ok: true, data };
  } catch {
    // Network failure, DNS error, or timeout — surface nothing internal.
    return { ok: false, status: null };
  }
}

export async function checkBackendHealth(): Promise<{ ok: boolean }> {
  try {
    const res = await fetch(HEALTH_API_URL, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    return { ok: res.ok };
  } catch {
    return { ok: false };
  }
}
