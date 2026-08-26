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
```

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

The initial version does not use persistent object storage or a database and
does not run automatic TTL cleanup. Before higher-volume use, add bounded
`MAX_DOCUMENTS`/`DOCUMENT_TTL_HOURS` cleanup or external durable storage.
