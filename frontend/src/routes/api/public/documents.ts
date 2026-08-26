import { createFileRoute } from "@tanstack/react-router";
import { paperRagProxyHeaders } from "@/lib/proxy-headers.server";

const BACKEND_BASE_URL = "https://furiosa-multimodal-rag.onrender.com";
const UPSTREAM_TIMEOUT_MS = 180_000;
const DEFAULT_PROXY_LIMIT_MB = 27;

function proxyLimitBytes(): number {
  const configured = Number(process.env.MAX_UPLOAD_PROXY_MB ?? DEFAULT_PROXY_LIMIT_MB);
  const megabytes =
    Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_PROXY_LIMIT_MB;
  return megabytes * 1024 * 1024;
}

export const Route = createFileRoute("/api/public/documents")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const contentLength = request.headers.get("content-length");
        if (contentLength !== null) {
          const parsedLength = Number(contentLength);
          if (!Number.isFinite(parsedLength) || parsedLength < 0) {
            return Response.json({ detail: "Invalid Content-Length." }, { status: 400 });
          }
          if (parsedLength > proxyLimitBytes()) {
            return Response.json({ detail: "PDF upload is too large" }, { status: 413 });
          }
        }

        const contentType = request.headers.get("content-type");
        if (!contentType?.toLowerCase().startsWith("multipart/form-data")) {
          return Response.json({ detail: "A PDF file is required." }, { status: 400 });
        }
        try {
          const upstream = await fetch(`${BACKEND_BASE_URL}/documents`, {
            method: "POST",
            headers: {
              "Content-Type": contentType,
              Accept: request.headers.get("accept") ?? "application/json",
              ...paperRagProxyHeaders(request.headers),
            },
            body: request.body,
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
