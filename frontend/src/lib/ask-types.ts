export interface AskSource {
  page: number;
  chunk: string;
}

export interface AskResponse {
  question: string;
  document_id?: string;
  answer: string;
  route: string;
  routing_reason?: string;
  vision_used: boolean;
  vision_available: boolean;
  fallback_used: boolean;
  sources: AskSource[];
  latency_ms: {
    total: number;
    routing?: number;
  };
}

export type AskResult =
  { ok: true; data: AskResponse } | { ok: false; status: number | null; detail?: string };

export interface CurrentDocument {
  documentId: string;
  filename: string;
  pages: number;
  status: "ready";
  cacheHit?: boolean;
}
