import { createFileRoute } from "@tanstack/react-router";

const BACKEND_BASE_URL = "https://furiosa-multimodal-rag.onrender.com";
const UPSTREAM_TIMEOUT_MS = 180_000;

export const Route = createFileRoute("/api/public/documents")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        try {
          const formData = await request.formData();
          const file = formData.get("file");
          if (!(file instanceof File)) {
            return Response.json({ detail: "A PDF file is required." }, { status: 400 });
          }

          const upstreamForm = new FormData();
          upstreamForm.set("file", file, file.name);
          const upstream = await fetch(`${BACKEND_BASE_URL}/documents`, {
            method: "POST",
            body: upstreamForm,
            signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
          });
          const body = await upstream.text();
          return new Response(body, {
            status: upstream.status,
            headers: { "Content-Type": upstream.headers.get("content-type") ?? "application/json" },
          });
        } catch {
          return Response.json({ detail: "Document service unavailable." }, { status: 502 });
        }
      },
    },
  },
});
