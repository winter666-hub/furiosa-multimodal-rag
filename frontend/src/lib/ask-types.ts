export interface AskSource {
  page: number;
  chunk: string;
  chunk_id: string;
  excerpt: string;
  retrieval_score: number;
  rerank_score?: number | null;
  page_width?: number | null;
  page_height?: number | null;
  highlights: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
  }>;
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
