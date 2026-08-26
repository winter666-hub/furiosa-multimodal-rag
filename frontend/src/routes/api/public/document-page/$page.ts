import { createFileRoute } from "@tanstack/react-router";

const BACKEND_BASE_URL = "https://furiosa-multimodal-rag.onrender.com";

// "Attention Is All You Need" (arXiv:1706.03762) — the only document
// served by the demo backend.
const MAX_DOCUMENT_PAGE = 15;
const UPSTREAM_TIMEOUT_MS = 120_000;

// Same-origin proxy for the backend's rendered page images, so the
// browser never calls the Render API directly (CORS-safe, URL hidden).
export const Route = createFileRoute("/api/public/document-page/$page")({
  server: {
    handlers: {
      GET: async ({ params }) => {
        const page = Number(params.page);
        if (!Number.isInteger(page) || page < 1 || page > MAX_DOCUMENT_PAGE) {
          return new Response("Invalid page number", { status: 400 });
        }

        try {
          const upstream = await fetch(
            `${BACKEND_BASE_URL}/document/page/${page}`,
            { signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) },
          );

          if (!upstream.ok || !upstream.body) {
            return new Response("Page not available", {
              status: upstream.status === 404 ? 404 : 502,
            });
          }

          return new Response(upstream.body, {
            headers: {
              "Content-Type": "image/png",
              "Cache-Control": "public, max-age=86400, immutable",
            },
          });
        } catch {
          return new Response("Document service unavailable", { status: 502 });
        }
      },
    },
  },
});
