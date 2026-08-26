# Render Deployment

## Service Type

Web Service

## Runtime

Python 3.11.9

## Build Command

```bash
pip install .
```

## Start Command

```bash
uvicorn furiosa_rag.web.app:app --host 0.0.0.0 --port $PORT
```

## Health Check Path

```text
/health
```

`/health` performs no model initialization, PDF access, or Furiosa API request.

## Source Page Preview

The frontend can render any one-based `sources[].page` from `/ask` through:

```text
GET /document/page/{page_number}
```

For example, source page `5` maps directly to `/document/page/5`. The response
is an in-memory PNG (`image/png`) rendered at 120 DPI. Up to eight rendered pages
are held in a bounded process-local LRU cache; no PNG files are written to disk.

## Uploaded Paper API

Upload an arbitrary research paper:

```text
POST /documents
Content-Type: multipart/form-data
field: file
```

Use the returned SHA-256 `document_id` for all subsequent operations:

```text
GET  /documents/{document_id}
POST /ask  {"document_id":"...", "question":"..."}
GET  /documents/{document_id}/pages/{page_number}
```

Uploaded documents are isolated under:

```text
/tmp/furiosa-rag/documents/<document_id>/document.pdf
/tmp/furiosa-rag/documents/<document_id>/cache/
```

The filename is metadata only and is never used as a storage path. IDs are the
SHA-256 hash of PDF bytes, so uploading identical bytes reuses the same document
and embedding cache. Page numbers in answer sources and preview URLs are both
one-based.

## Required Environment Variables

```text
FURIOSA_API_KEY
FURIOSA_REQUEST_TIMEOUT
FURIOSA_LLM_BASE_URL
FURIOSA_LLM_MODEL
FURIOSA_EMBEDDING_BASE_URL
FURIOSA_EMBEDDING_MODEL
FURIOSA_RERANKER_BASE_URL
FURIOSA_RERANKER_MODEL
```

Never put secret values in source files or Render build logs.

## Optional Environment Variables

```text
DEPLOYMENT_MODE=hosted_only
ALLOWED_ORIGINS=https://example.lovable.app,http://localhost:5173
DEMO_PDF_PATH=/path/to/attention_is_all_you_need.pdf
DEMO_PDF_URL=https://public.example/attention_is_all_you_need.pdf
MAX_PDF_UPLOAD_MB=25
DOCUMENT_STORAGE_ROOT=/tmp/furiosa-rag/documents
DATABASE_URL=<Render PostgreSQL internal URL; omit to disable logging>
PAPER_RAG_PROXY_SECRET=<long-random-proxy-secret>
UPLOAD_RATE_LIMIT_REQUESTS=3
UPLOAD_RATE_LIMIT_WINDOW_SECONDS=600
ASK_RATE_LIMIT_REQUESTS=20
ASK_RATE_LIMIT_WINDOW_SECONDS=600
MAX_CONCURRENT_UPLOADS=1
MAX_CONCURRENT_ASKS=3
MAX_DOCUMENTS=20
DOCUMENT_TTL_HOURS=6
MAX_DOCUMENT_STORAGE_MB=500
```

Set the exact same `PAPER_RAG_PROXY_SECRET` as a server-side secret in both the
Cloudflare Worker frontend and the Render backend. It is separate from
`FURIOSA_API_KEY` and must never be exposed to browser code. The Worker reads
Cloudflare's `CF-Connecting-IP` and sends it to Render using internal headers;
Render accepts that address only when the accompanying proxy token matches with
a constant-time comparison. Direct Render requests and requests with missing or
invalid tokens are limited by Render's immediate peer address. Ordinary
`X-Forwarded-For` and `X-Real-IP` values are not trusted for limiter identity.

The upload and ask limits are process-local controls for the single Render
instance used by this demo. A rate-limited request returns HTTP 429 with a
`Retry-After` header. Requests above the concurrency caps fail immediately with
HTTP 503 instead of entering an unbounded queue.

Before accepting a new PDF, the document store removes expired directories and
then evicts the least-recently-accessed documents until the count and storage
caps have room. Documents currently being uploaded, indexed, or queried are
protected from cleanup. Existing metadata without access timestamps remains
readable and falls back to its directory modification time.

`DEPLOYMENT_MODE` defaults to `hosted_only`. In this mode the web service never
constructs or calls the Direct NPU Vision backend. A `VISUAL_REQUIRED` route is
reported to the caller and answered through Text RAG with `fallback_used=true`.

The repository intentionally ignores `data/` and PDF files. Recommended Render
values are:

```text
DEMO_PDF_PATH=/tmp/attention_is_all_you_need.pdf
DEMO_PDF_URL=<public PDF URL>
```

On the first `/ask`, the service uses an existing file at `DEMO_PDF_PATH` or
downloads `DEMO_PDF_URL` to a temporary file, verifies the `%PDF-` signature,
and atomically moves it into place. A failed download leaves no partial final
file and `/ask` returns HTTP 503. `/health` never triggers the download.

Do not commit the paper merely to make deployment work. Ensure that the public
URL and its use comply with the document host's terms and the paper's license.

If `ALLOWED_ORIGINS` is absent, the CORS allow-list is empty. Do not use `*` as a
production default.

## Render Free Plan Notes

- The service may sleep while idle, causing a cold start on the next request.
- Treat the local filesystem as ephemeral, not persistent storage.
- The PDF may be downloaded again after an instance restart or redeployment.
- Embedding cache files may disappear after restart or redeployment.
- Uploaded PDFs and their per-document caches disappear after restart/redeploy.
- Use the same cache conditions when comparing benchmark strategies.

The limiter, concurrency counters, active-document tracking, and lock map are
process-local. They assume one Render instance and reset on restart. A
multi-instance production deployment must enforce limits in shared
infrastructure such as Cloudflare Rate Limiting, Durable Objects, Redis, or
another external store. The local filesystem is still ephemeral.

The Cloudflare frontend upload proxy accepts `MAX_UPLOAD_PROXY_MB` (default 27)
as a fast `Content-Length` limit and forwards the raw multipart stream. It does
not buffer the body with `request.formData()`. Requests without a
`Content-Length` header rely on the backend's streaming 25 MB hard limit; the
proxy intentionally does not buffer such requests merely to measure them.

Configure the Cloudflare Worker secrets separately from public build variables:

```text
PAPER_RAG_PROXY_SECRET=<the same value configured on Render>
MAX_UPLOAD_PROXY_MB=27
```

## Optional PostgreSQL Conversation Logging

Attach a Render PostgreSQL database to the backend and set its internal connection URL as:

```text
DATABASE_URL=<Render PostgreSQL internal URL>
```

When `DATABASE_URL` is present, backend startup creates the `chat_logs` table and its
`session_id`, `created_at`, and `document_id` indexes if they do not already exist. `/health`
does not initialize the database. When the variable is absent, logging is disabled and the app
continues to operate normally. A database initialization or insert failure is also best-effort:
the answer still succeeds and logs contain only a fixed warning without connection details.

Only the question, generated answer, routing/fallback metadata, source references, timing data,
document ID/filename, and a browser-tab session UUID are stored. API keys, proxy secrets, client
IP addresses, request headers/cookies, PDFs, full extracted document text, and embeddings are not
stored in this table. `DATABASE_URL` belongs only on Render; do not configure it as a Cloudflare
Worker variable or expose it to browser code.

This initial implementation does not include a history API, admin UI, or automatic retention
cleanup. Define and implement a retention/deletion policy before treating the log as a long-term
dataset, and restrict database access to authorized operators.
