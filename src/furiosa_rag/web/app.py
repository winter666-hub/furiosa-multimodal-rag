"""Thin hosted-only FastAPI wrapper around the existing RAG components."""

from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.models import RagAnswer
from furiosa_rag.pdf_images import PdfPageRenderer
from furiosa_rag.pipeline import TextRagPipeline
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.router import AdaptiveQueryRouter, LLMQueryRouter, QueryRoute, QueryRouter
from furiosa_rag.web.documents import (
    DocumentNotFoundError,
    DocumentStore,
    DocumentTooLargeError,
    DocumentValidationError,
    RegisteredDocument,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class SourceResponse(BaseModel):
    page: int
    chunk: str


class AskResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    document_id: str | None = None
    answer: str
    route: str
    routing_reason: str
    vision_used: bool
    vision_available: bool
    fallback_used: bool
    sources: list[SourceResponse]
    latency_ms: dict[str, float]


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    status: str
    cache_hit: bool | None = None


def parse_allowed_origins(value: str | None) -> list[str]:
    if not value:
        return []
    return list(dict.fromkeys(origin.strip() for origin in value.split(",") if origin.strip()))


def _endpoint(settings: Settings, name: str) -> ModelEndpoint:
    return next(endpoint for endpoint in settings.endpoints if endpoint.name == name)


def ensure_demo_pdf(
    path_value: str | Path,
    url: str | None,
    *,
    timeout: float,
) -> Path | None:
    """Return an existing demo PDF or atomically download it from an HTTP(S) URL."""
    if not str(path_value).strip():
        return None
    destination = Path(path_value)
    if destination.is_file():
        return destination
    if not url or not url.strip():
        return None
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return None

    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(url.strip(), timeout=timeout) as response:
                magic = response.read(5)
                if magic != b"%PDF-":
                    raise ValueError("downloaded document is not a PDF")
                temporary.write(magic)
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        return destination
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        return None


@lru_cache(maxsize=1)
def get_demo_pdf_path() -> Path | None:
    settings = Settings.from_env()
    return ensure_demo_pdf(
        os.getenv("DEMO_PDF_PATH", ""),
        os.getenv("DEMO_PDF_URL"),
        timeout=settings.request_timeout,
    )


@lru_cache(maxsize=8)
def render_page_png(pdf_path: str, page_number: int) -> bytes:
    """Render and cache up to eight one-based source pages in memory."""
    return PdfPageRenderer(dpi=120.0).render_png(pdf_path, page_number)


def _max_upload_bytes() -> int:
    try:
        megabytes = int(os.getenv("MAX_PDF_UPLOAD_MB", "25"))
    except ValueError as exc:
        raise RuntimeError("MAX_PDF_UPLOAD_MB must be an integer") from exc
    if megabytes <= 0:
        raise RuntimeError("MAX_PDF_UPLOAD_MB must be greater than zero")
    return megabytes * 1024 * 1024


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    root = os.getenv("DOCUMENT_STORAGE_ROOT", "/tmp/furiosa-rag/documents")
    return DocumentStore(root, max_upload_bytes=_max_upload_bytes())


class DocumentPipelineFactory:
    def __init__(
        self,
        embedding: FuriosaEmbedding,
        reranker: FuriosaReranker,
        llm: FuriosaLlm,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker
        self.llm = llm

    def create(self, document: RegisteredDocument) -> TextRagPipeline:
        return TextRagPipeline(
            self.embedding,
            self.reranker,
            self.llm,
            cache=DocumentEmbeddingCache(document.cache_dir),
        )

    def prepare(self, document: RegisteredDocument) -> None:
        if document.cache_hit:
            return
        pipeline = self.create(document)
        pipeline._retrieve(  # Reuse the existing extraction/chunking/embedding cache path.
            document.pdf_path,
            "document indexing",
            rebuild_cache=False,
        )


class HostedOnlyRagService:
    """Apply deployment policy without changing research routers or pipelines."""

    def __init__(
        self,
        router: QueryRouter,
        text_pipeline: TextRagPipeline,
        pdf_path: str | Path,
        *,
        document_store: DocumentStore | None = None,
        pipeline_factory: DocumentPipelineFactory | None = None,
    ) -> None:
        self.router = router
        self.text_pipeline = text_pipeline
        self.pdf_path = Path(pdf_path) if str(pdf_path).strip() else None
        self.document_store = document_store
        self.pipeline_factory = pipeline_factory

    def ask(self, question: str, document_id: str | None = None) -> AskResponse:
        active_document: RegisteredDocument | None = None
        if document_id is not None:
            if self.document_store is None or self.pipeline_factory is None:
                raise DocumentNotFoundError("document not found")
            record = self.document_store.get(document_id)
            active_document = RegisteredDocument(
                record,
                self.document_store.pdf_path(document_id),
                self.document_store.cache_dir(document_id),
                cache_hit=True,
            )
            pipeline = self.pipeline_factory.create(active_document)
            pdf_path = active_document.pdf_path
        else:
            if self.pdf_path is None or not self.pdf_path.is_file():
                raise FileNotFoundError("configured demo PDF is unavailable")
            pipeline = self.text_pipeline
            pdf_path = self.pdf_path

        total_started = time.perf_counter_ns()
        routing_started = time.perf_counter_ns()
        decision = self.router.route(question)
        routing_latency = (time.perf_counter_ns() - routing_started) / 1_000_000

        # hosted_only intentionally uses Text RAG even when visual evidence was requested.
        result: RagAnswer = pipeline.answer(pdf_path, question)
        total_latency = (time.perf_counter_ns() - total_started) / 1_000_000
        latency = {
            key: float(value)
            for key, value in result.latency_ms.items()
            if not isinstance(value, bool)
        }
        latency["routing"] = routing_latency
        latency["total"] = total_latency
        fallback_used = decision.route is QueryRoute.VISUAL_REQUIRED
        return AskResponse(
            question=question,
            document_id=document_id,
            answer=result.answer,
            route=decision.route.value,
            routing_reason=decision.reason,
            vision_used=False,
            vision_available=False,
            fallback_used=fallback_used,
            sources=[
                SourceResponse(
                    page=source.chunk.page_number,
                    chunk=source.chunk.chunk_id,
                )
                for source in result.sources
            ],
            latency_ms=latency,
        )

    def prepare_document(self, document: RegisteredDocument) -> None:
        if self.pipeline_factory is None:
            raise RuntimeError("uploaded document pipeline is unavailable")
        self.pipeline_factory.prepare(document)


@lru_cache(maxsize=1)
def get_service() -> HostedOnlyRagService:
    mode = os.getenv("DEPLOYMENT_MODE", "hosted_only").strip().casefold()
    if mode != "hosted_only":
        raise RuntimeError(f"unsupported deployment mode: {mode}")
    settings = Settings.from_env()
    pdf_path = get_demo_pdf_path()
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    llm = FuriosaLlm(_endpoint(settings, "llm"), client)
    embedding = FuriosaEmbedding(_endpoint(settings, "embedding"), client)
    reranker = FuriosaReranker(_endpoint(settings, "reranker"), client)
    router = AdaptiveQueryRouter(LLMQueryRouter(_endpoint(settings, "llm"), client))
    pipeline = TextRagPipeline(
        embedding,
        reranker,
        llm,
    )
    return HostedOnlyRagService(
        router,
        pipeline,
        pdf_path or "",
        document_store=get_document_store(),
        pipeline_factory=DocumentPipelineFactory(embedding, reranker, llm),
    )


app = FastAPI(title="Furiosa Multimodal RAG", version="0.1.0")
allowed_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    service: Annotated[HostedOnlyRagService, Depends(get_service)],
) -> DocumentResponse:
    if service.document_store is None:
        raise HTTPException(status_code=503, detail="document storage unavailable")
    try:
        document = await service.document_store.register(file)
        await run_in_threadpool(service.prepare_document, document)
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail="PDF upload is too large") from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail="invalid PDF") from exc
    except FuriosaApiError as exc:
        raise HTTPException(status_code=502, detail="upstream model service unavailable") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="document processing failed") from exc
    return DocumentResponse(
        document_id=document.record.document_id,
        filename=document.record.filename,
        pages=document.record.pages,
        status=document.record.status,
        cache_hit=document.cache_hit,
    )


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    response_model_exclude_none=True,
)
def document_metadata(
    document_id: str,
    store: Annotated[DocumentStore, Depends(get_document_store)],
) -> DocumentResponse:
    try:
        record = store.get(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        pages=record.pages,
        status=record.status,
    )


@app.get("/documents/{document_id}/pages/{page_number}", response_class=Response)
async def uploaded_document_page(
    document_id: str,
    page_number: int,
    store: Annotated[DocumentStore, Depends(get_document_store)],
) -> Response:
    if page_number <= 0:
        raise HTTPException(status_code=422, detail="page_number must be greater than zero")
    try:
        store.get(document_id)
        pdf_path = store.pdf_path(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    try:
        content = await run_in_threadpool(render_page_png, str(pdf_path), page_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="document page not found") from exc
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail="page preview unavailable") from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/document/page/{page_number}", response_class=Response)
async def document_page(
    page_number: int,
    pdf_path: Annotated[Path | None, Depends(get_demo_pdf_path)],
) -> Response:
    if page_number <= 0:
        raise HTTPException(status_code=422, detail="page_number must be greater than zero")
    if pdf_path is None or not pdf_path.is_file():
        raise HTTPException(status_code=503, detail="demo document unavailable")
    try:
        content = await run_in_threadpool(render_page_png, str(pdf_path), page_number)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="demo document unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="document page not found") from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail="page preview unavailable") from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
async def ask(
    request: AskRequest,
    service: Annotated[HostedOnlyRagService, Depends(get_service)],
) -> AskResponse:
    try:
        return await run_in_threadpool(service.ask, request.question, request.document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="demo document unavailable") from exc
    except FuriosaApiError as exc:
        raise HTTPException(status_code=502, detail="upstream model service unavailable") from exc
    except (TimeoutError, OSError) as exc:
        raise HTTPException(status_code=502, detail="upstream service unavailable") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="request processing failed") from exc
