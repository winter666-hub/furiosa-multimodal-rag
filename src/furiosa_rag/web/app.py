"""Thin hosted-only FastAPI wrapper around the existing RAG components."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from furiosa_rag.cache import DocumentEmbeddingCache
from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.models import RagAnswer
from furiosa_rag.pdf_images import PdfPageRenderer, RenderPixelLimitError, find_text_highlights
from furiosa_rag.pipeline import (
    DocumentTooLargeToIndexError,
    RagConfig,
    TextRagPipeline,
    clean_internal_citations,
)
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.router import AdaptiveQueryRouter, LLMQueryRouter, QueryRoute, QueryRouter
from furiosa_rag.web.database import (
    ChatLogRecord,
    ChatLogRepository,
    create_chat_log_repository,
)
from furiosa_rag.web.documents import (
    DocumentNotFoundError,
    DocumentStore,
    DocumentTooLargeError,
    DocumentValidationError,
    RegisteredDocument,
)
from furiosa_rag.web.limits import ConcurrencyLimiter, RateLimiter, client_ip

logger = logging.getLogger(__name__)
RATE_LIMIT_DETAIL = "Too many requests. Please try again later."
BUSY_DETAIL = "The demo is currently busy. Please try again shortly."


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    session_id: uuid.UUID | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class HighlightResponse(BaseModel):
    x: float
    y: float
    width: float
    height: float


class SourceResponse(BaseModel):
    page: int
    chunk: str
    chunk_id: str
    excerpt: str
    retrieval_score: float
    rerank_score: float | None = None
    page_width: float | None = None
    page_height: float | None = None
    highlights: list[HighlightResponse] = Field(default_factory=list)


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
    return PdfPageRenderer(
        max_pixels=_positive_int_env("MAX_RENDER_PIXELS", 20_000_000)
    ).render_png(pdf_path, page_number)


def _max_upload_bytes() -> int:
    try:
        megabytes = int(os.getenv("MAX_PDF_UPLOAD_MB", "25"))
    except ValueError as exc:
        raise RuntimeError("MAX_PDF_UPLOAD_MB must be an integer") from exc
    if megabytes <= 0:
        raise RuntimeError("MAX_PDF_UPLOAD_MB must be greater than zero")
    return megabytes * 1024 * 1024


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    root = os.getenv("DOCUMENT_STORAGE_ROOT", "/tmp/furiosa-rag/documents")
    return DocumentStore(
        root,
        max_upload_bytes=_max_upload_bytes(),
        max_documents=_positive_int_env("MAX_DOCUMENTS", 20),
        document_ttl_hours=_positive_float_env("DOCUMENT_TTL_HOURS", 6),
        max_storage_bytes=_positive_int_env("MAX_DOCUMENT_STORAGE_MB", 500) * 1024 * 1024,
        max_pdf_pages=_positive_int_env("MAX_PDF_PAGES", 100),
    )


@lru_cache(maxsize=1)
def get_upload_rate_limiter() -> RateLimiter:
    return RateLimiter(
        _positive_int_env("UPLOAD_RATE_LIMIT_REQUESTS", 3),
        _positive_int_env("UPLOAD_RATE_LIMIT_WINDOW_SECONDS", 600),
    )


@lru_cache(maxsize=1)
def get_ask_rate_limiter() -> RateLimiter:
    return RateLimiter(
        _positive_int_env("ASK_RATE_LIMIT_REQUESTS", 20),
        _positive_int_env("ASK_RATE_LIMIT_WINDOW_SECONDS", 600),
    )


@lru_cache(maxsize=1)
def get_upload_concurrency_limiter() -> ConcurrencyLimiter:
    return ConcurrencyLimiter(_positive_int_env("MAX_CONCURRENT_UPLOADS", 1))


@lru_cache(maxsize=1)
def get_ask_concurrency_limiter() -> ConcurrencyLimiter:
    return ConcurrencyLimiter(_positive_int_env("MAX_CONCURRENT_ASKS", 3))


@lru_cache(maxsize=1)
def get_page_render_concurrency_limiter() -> ConcurrencyLimiter:
    return ConcurrencyLimiter(_positive_int_env("MAX_CONCURRENT_PAGE_RENDERS", 2))


@lru_cache(maxsize=1)
def get_chat_log_repository() -> ChatLogRepository | None:
    try:
        return create_chat_log_repository()
    except Exception:  # noqa: BLE001 - persistence must never break the public API
        logger.warning("Failed to configure chat log persistence")
        return None


def _enforce_rate_limit(request: Request, limiter: RateLimiter) -> None:
    allowed, retry_after = limiter.check(
        client_ip(request, os.getenv("PAPER_RAG_PROXY_SECRET"))
    )
    if not allowed:
        logger.warning("Demo API rate limit exceeded")
        raise HTTPException(
            status_code=429,
            detail=RATE_LIMIT_DETAIL,
            headers={"Retry-After": str(retry_after)},
        )


def enforce_upload_rate_limit(
    request: Request,
    limiter: Annotated[RateLimiter, Depends(get_upload_rate_limiter)],
) -> None:
    _enforce_rate_limit(request, limiter)


def enforce_ask_rate_limit(
    request: Request,
    limiter: Annotated[RateLimiter, Depends(get_ask_rate_limiter)],
) -> None:
    _enforce_rate_limit(request, limiter)


class DocumentPipelineFactory:
    def __init__(
        self,
        embedding: FuriosaEmbedding,
        reranker: FuriosaReranker,
        llm: FuriosaLlm,
        config: RagConfig | None = None,
    ) -> None:
        self.embedding = embedding
        self.reranker = reranker
        self.llm = llm
        self.config = config or RagConfig()

    def create(self, document: RegisteredDocument) -> TextRagPipeline:
        return TextRagPipeline(
            self.embedding,
            self.reranker,
            self.llm,
            config=self.config,
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
            record = self.document_store.get(document_id, touch=True)
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
        source_responses: list[SourceResponse] = []
        seen_sources: set[tuple[str | None, int, str]] = set()
        for source in result.sources:
            key = (document_id, source.chunk.page_number, source.chunk.chunk_id)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            excerpt = source.chunk.text.strip()[:500]
            page_width: float | None = None
            page_height: float | None = None
            highlights: list[HighlightResponse] = []
            try:
                located = find_text_highlights(pdf_path, source.chunk.page_number, excerpt)
                page_width = located.page_width
                page_height = located.page_height
                highlights = [
                    HighlightResponse(
                        x=rectangle.x,
                        y=rectangle.y,
                        width=rectangle.width,
                        height=rectangle.height,
                    )
                    for rectangle in located.rectangles
                ]
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                pass
            source_responses.append(
                SourceResponse(
                    page=source.chunk.page_number,
                    chunk=source.chunk.chunk_id,
                    chunk_id=source.chunk.chunk_id,
                    excerpt=excerpt,
                    retrieval_score=source.retrieval_score,
                    rerank_score=source.rerank_score,
                    page_width=page_width,
                    page_height=page_height,
                    highlights=highlights,
                )
            )

        return AskResponse(
            question=question,
            document_id=document_id,
            answer=clean_internal_citations(result.answer),
            route=decision.route.value,
            routing_reason=decision.reason,
            vision_used=False,
            vision_available=False,
            fallback_used=fallback_used,
            sources=source_responses,
            latency_ms=latency,
        )

    def prepare_document(self, document: RegisteredDocument) -> None:
        if self.pipeline_factory is None:
            raise RuntimeError("uploaded document pipeline is unavailable")
        self.pipeline_factory.prepare(document)

    def filename_for(self, document_id: str | None) -> str | None:
        if document_id is not None and self.document_store is not None:
            return self.document_store.get(document_id).filename
        return self.pdf_path.name if self.pdf_path is not None else None


def _prepare_document_locked(
    service: HostedOnlyRagService, document: RegisteredDocument
) -> RegisteredDocument:
    assert service.document_store is not None
    with service.document_store.processing(document.record.document_id):
        current = RegisteredDocument(
            document.record,
            document.pdf_path,
            document.cache_dir,
            any(document.cache_dir.glob("*.npz")),
        )
        service.prepare_document(current)
        return current


def _ask_locked(service: HostedOnlyRagService, request: AskRequest) -> AskResponse:
    if request.document_id is None or service.document_store is None:
        return service.ask(request.question, request.document_id)
    with service.document_store.processing(request.document_id):
        return service.ask(request.question, request.document_id)


def _persist_chat_log(
    repository: ChatLogRepository,
    service: HostedOnlyRagService,
    request: AskRequest,
    response: AskResponse,
) -> None:
    session_id = request.session_id or uuid.uuid4()
    record = ChatLogRecord.create(
        session_id=session_id,
        document_id=response.document_id,
        filename=service.filename_for(response.document_id),
        question=response.question,
        answer=response.answer,
        route=response.route,
        routing_reason=response.routing_reason,
        vision_used=response.vision_used,
        vision_available=response.vision_available,
        fallback_used=response.fallback_used,
        sources=[
            {
                "page": source.page,
                "chunk_id": source.chunk_id,
                "excerpt": source.excerpt,
                "retrieval_score": source.retrieval_score,
                "rerank_score": source.rerank_score,
            }
            for source in response.sources
        ],
        latency_ms=dict(response.latency_ms),
    )
    repository.persist(record)


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
    rag_config = RagConfig(
        max_extracted_characters=_positive_int_env("MAX_EXTRACTED_CHARACTERS", 1_000_000),
        max_document_chunks=_positive_int_env("MAX_DOCUMENT_CHUNKS", 1_000),
        embedding_batch_size=_positive_int_env("EMBEDDING_BATCH_SIZE", 32),
    )
    pipeline = TextRagPipeline(
        embedding,
        reranker,
        llm,
        config=rag_config,
    )
    return HostedOnlyRagService(
        router,
        pipeline,
        pdf_path or "",
        document_store=get_document_store(),
        pipeline_factory=DocumentPipelineFactory(embedding, reranker, llm, rag_config),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    repository = get_chat_log_repository()
    if repository is not None:
        try:
            await run_in_threadpool(repository.initialize)
        except Exception:  # noqa: BLE001 - persistence is best-effort by design
            logger.warning("Failed to initialize chat log persistence")
    yield


app = FastAPI(title="Furiosa Multimodal RAG", version="0.1.0", lifespan=lifespan)
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
    _: Annotated[None, Depends(enforce_upload_rate_limit)],
    service: Annotated[HostedOnlyRagService, Depends(get_service)],
    concurrency: Annotated[ConcurrencyLimiter, Depends(get_upload_concurrency_limiter)],
) -> DocumentResponse:
    if service.document_store is None:
        raise HTTPException(status_code=503, detail="document storage unavailable")
    if not concurrency.acquire():
        logger.warning("Demo upload concurrency limit reached")
        raise HTTPException(status_code=503, detail=BUSY_DETAIL)
    document: RegisteredDocument | None = None
    prepared = False
    try:
        document = await service.document_store.register(file)
        current = await run_in_threadpool(_prepare_document_locked, service, document)
        prepared = True
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DocumentTooLargeToIndexError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail="invalid PDF") from exc
    except FuriosaApiError as exc:
        raise HTTPException(status_code=502, detail="upstream model service unavailable") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="document processing failed") from exc
    finally:
        if document is not None and not document.cache_hit and not prepared:
            try:
                await run_in_threadpool(
                    service.document_store.discard, document.record.document_id
                )
            except (OSError, RuntimeError):
                logger.warning("Failed to discard an unprepared document")
        concurrency.release()
    return DocumentResponse(
        document_id=document.record.document_id,
        filename=document.record.filename,
        pages=document.record.pages,
        status=document.record.status,
        cache_hit=current.cache_hit,
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
    concurrency: Annotated[
        ConcurrencyLimiter, Depends(get_page_render_concurrency_limiter)
    ],
) -> Response:
    if page_number <= 0:
        raise HTTPException(status_code=422, detail="page_number must be greater than zero")
    try:
        store.get(document_id)
        pdf_path = store.pdf_path(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    if not concurrency.acquire():
        raise HTTPException(
            status_code=503,
            detail=BUSY_DETAIL,
            headers={"Retry-After": "1"},
        )
    try:
        try:
            content = await run_in_threadpool(render_page_png, str(pdf_path), page_number)
        except RenderPixelLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="document page not found") from exc
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail="page preview unavailable") from exc
    finally:
        concurrency.release()
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/document/page/{page_number}", response_class=Response)
async def document_page(
    page_number: int,
    pdf_path: Annotated[Path | None, Depends(get_demo_pdf_path)],
    concurrency: Annotated[
        ConcurrencyLimiter, Depends(get_page_render_concurrency_limiter)
    ],
) -> Response:
    if page_number <= 0:
        raise HTTPException(status_code=422, detail="page_number must be greater than zero")
    if pdf_path is None or not pdf_path.is_file():
        raise HTTPException(status_code=503, detail="demo document unavailable")
    if not concurrency.acquire():
        raise HTTPException(
            status_code=503,
            detail=BUSY_DETAIL,
            headers={"Retry-After": "1"},
        )
    try:
        try:
            content = await run_in_threadpool(render_page_png, str(pdf_path), page_number)
        except RenderPixelLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="demo document unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="document page not found") from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail="page preview unavailable") from exc
    finally:
        concurrency.release()
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
async def ask(
    request: AskRequest,
    _: Annotated[None, Depends(enforce_ask_rate_limit)],
    service: Annotated[HostedOnlyRagService, Depends(get_service)],
    concurrency: Annotated[ConcurrencyLimiter, Depends(get_ask_concurrency_limiter)],
    chat_logs: Annotated[ChatLogRepository | None, Depends(get_chat_log_repository)],
) -> AskResponse:
    if not concurrency.acquire():
        logger.warning("Demo ask concurrency limit reached")
        raise HTTPException(status_code=503, detail=BUSY_DETAIL)
    try:
        response = await run_in_threadpool(_ask_locked, service, request)
        if chat_logs is not None:
            try:
                await run_in_threadpool(_persist_chat_log, chat_logs, service, request, response)
            except Exception:  # noqa: BLE001 - return the completed answer on DB failure
                logger.warning("Failed to persist chat log")
        return response
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
    finally:
        concurrency.release()
