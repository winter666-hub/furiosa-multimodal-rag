const SESSION_ID_KEY = "paper-rag-session-id";

export function getSessionId(): string {
  try {
    const existing = window.sessionStorage.getItem(SESSION_ID_KEY);
    if (existing) return existing;

    const created = window.crypto.randomUUID();
    window.sessionStorage.setItem(SESSION_ID_KEY, created);
    return created;
  } catch {
    // Storage may be unavailable in privacy-restricted browser contexts.
    return window.crypto.randomUUID();
  }
}
