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

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from furiosa_rag.clients import FuriosaApiError, FuriosaClient
from furiosa_rag.config import ModelEndpoint, Settings
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.models import RagAnswer
from furiosa_rag.pipeline import TextRagPipeline
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.router import AdaptiveQueryRouter, LLMQueryRouter, QueryRoute, QueryRouter


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

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
    answer: str
    route: str
    routing_reason: str
    vision_used: bool
    vision_available: bool
    fallback_used: bool
    sources: list[SourceResponse]
    latency_ms: dict[str, float]


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


class HostedOnlyRagService:
    """Apply deployment policy without changing research routers or pipelines."""

    def __init__(
        self,
        router: QueryRouter,
        text_pipeline: TextRagPipeline,
        pdf_path: str | Path,
    ) -> None:
        self.router = router
        self.text_pipeline = text_pipeline
        self.pdf_path = Path(pdf_path) if str(pdf_path).strip() else None

    def ask(self, question: str) -> AskResponse:
        if self.pdf_path is None or not self.pdf_path.is_file():
            raise FileNotFoundError("configured demo PDF is unavailable")

        total_started = time.perf_counter_ns()
        routing_started = time.perf_counter_ns()
        decision = self.router.route(question)
        routing_latency = (time.perf_counter_ns() - routing_started) / 1_000_000

        # hosted_only intentionally uses Text RAG even when visual evidence was requested.
        result: RagAnswer = self.text_pipeline.answer(self.pdf_path, question)
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


@lru_cache(maxsize=1)
def get_service() -> HostedOnlyRagService:
    mode = os.getenv("DEPLOYMENT_MODE", "hosted_only").strip().casefold()
    if mode != "hosted_only":
        raise RuntimeError(f"unsupported deployment mode: {mode}")
    settings = Settings.from_env()
    pdf_path = ensure_demo_pdf(
        os.getenv("DEMO_PDF_PATH", ""),
        os.getenv("DEMO_PDF_URL"),
        timeout=settings.request_timeout,
    )
    client = FuriosaClient(settings.api_key, settings.request_timeout)
    llm = FuriosaLlm(_endpoint(settings, "llm"), client)
    router = AdaptiveQueryRouter(LLMQueryRouter(_endpoint(settings, "llm"), client))
    pipeline = TextRagPipeline(
        FuriosaEmbedding(_endpoint(settings, "embedding"), client),
        FuriosaReranker(_endpoint(settings, "reranker"), client),
        llm,
    )
    return HostedOnlyRagService(router, pipeline, pdf_path or "")


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


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    service: Annotated[HostedOnlyRagService, Depends(get_service)],
) -> AskResponse:
    try:
        return await run_in_threadpool(service.ask, request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="demo document unavailable") from exc
    except FuriosaApiError as exc:
        raise HTTPException(status_code=502, detail="upstream model service unavailable") from exc
    except (TimeoutError, OSError) as exc:
        raise HTTPException(status_code=502, detail="upstream service unavailable") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="request processing failed") from exc
