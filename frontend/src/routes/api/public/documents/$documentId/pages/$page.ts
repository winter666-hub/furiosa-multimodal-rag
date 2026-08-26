import { createFileRoute } from "@tanstack/react-router";

const BACKEND_BASE_URL = "https://furiosa-multimodal-rag.onrender.com";
const UPSTREAM_TIMEOUT_MS = 120_000;
const DOCUMENT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;

export const Route = createFileRoute("/api/public/documents/$documentId/pages/$page")({
  server: {
    handlers: {
      GET: async ({ params }) => {
        const page = Number(params.page);
        if (!DOCUMENT_ID_PATTERN.test(params.documentId) || !Number.isInteger(page) || page < 1) {
          return new Response("Invalid document or page number", { status: 400 });
        }

        try {
          const upstream = await fetch(
            `${BACKEND_BASE_URL}/documents/${encodeURIComponent(params.documentId)}/pages/${page}`,
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
