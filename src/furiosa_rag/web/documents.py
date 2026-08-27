"""Ephemeral, document-isolated PDF storage for the web application."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from fastapi import UploadFile

DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
DEFAULT_MAX_PDF_PAGES = 100
MAX_PDF_PAGE_WIDTH_POINTS = 4_000.0
MAX_PDF_PAGE_HEIGHT_POINTS = 4_000.0


class DocumentValidationError(ValueError):
    pass


class DocumentTooLargeError(DocumentValidationError):
    pass


class DocumentNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    filename: str
    pages: int
    status: str = "ready"
    created_at: float | None = None
    last_accessed_at: float | None = None


@dataclass(frozen=True, slots=True)
class RegisteredDocument:
    record: DocumentRecord
    pdf_path: Path
    cache_dir: Path
    cache_hit: bool


class DocumentStore:
    def __init__(
        self,
        root: str | Path,
        *,
        max_upload_bytes: int,
        max_documents: int = 20,
        document_ttl_hours: float = 6,
        max_storage_bytes: int = 500 * 1024 * 1024,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        self.root = Path(root)
        self.max_upload_bytes = max_upload_bytes
        if (
            max_documents <= 0
            or document_ttl_hours <= 0
            or max_storage_bytes <= 0
            or max_pdf_pages <= 0
        ):
            raise ValueError("document storage limits must be greater than zero")
        self.max_documents = max_documents
        self.document_ttl_seconds = document_ttl_hours * 3600
        self.max_storage_bytes = max_storage_bytes
        self.max_pdf_pages = max_pdf_pages
        self._state_lock = threading.RLock()
        self._document_locks: dict[str, tuple[threading.Lock, int]] = {}
        self._active_documents: dict[str, int] = {}

    def _directory(self, document_id: str) -> Path:
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise DocumentNotFoundError("document not found")
        return self.root / document_id

    def pdf_path(self, document_id: str) -> Path:
        return self._directory(document_id) / "document.pdf"

    def cache_dir(self, document_id: str) -> Path:
        return self._directory(document_id) / "cache"

    def _metadata_path(self, document_id: str) -> Path:
        return self._directory(document_id) / "metadata.json"

    def get(self, document_id: str, *, touch: bool = False) -> DocumentRecord:
        metadata_path = self._metadata_path(document_id)
        pdf_path = self.pdf_path(document_id)
        if not metadata_path.is_file() or not pdf_path.is_file():
            raise DocumentNotFoundError("document not found")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            record = DocumentRecord(
                document_id=str(payload["document_id"]),
                filename=str(payload["filename"]),
                pages=int(payload["pages"]),
                status=str(payload.get("status", "ready")),
                created_at=float(payload["created_at"]) if "created_at" in payload else None,
                last_accessed_at=(
                    float(payload["last_accessed_at"])
                    if "last_accessed_at" in payload
                    else None
                ),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise DocumentNotFoundError("document not found") from exc
        if record.document_id != document_id or record.pages <= 0:
            raise DocumentNotFoundError("document not found")
        if touch:
            now = time.time()
            record = DocumentRecord(
                document_id=record.document_id,
                filename=record.filename,
                pages=record.pages,
                status=record.status,
                created_at=record.created_at,
                last_accessed_at=now,
            )
            self._write_metadata(record)
        return record

    def _write_metadata(self, record: DocumentRecord) -> None:
        directory = self._directory(record.document_id)
        directory.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".metadata.", suffix=".tmp", dir=directory
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            temporary_path.write_text(json.dumps(asdict(record)), encoding="utf-8")
            os.replace(temporary_path, directory / "metadata.json")
        finally:
            temporary_path.unlink(missing_ok=True)

    @contextmanager
    def processing(self, document_id: str) -> Iterator[None]:
        """Serialize one document and keep cleanup away while it is in use."""

        self._directory(document_id)
        with self._state_lock:
            lock, references = self._document_locks.get(document_id, (threading.Lock(), 0))
            self._document_locks[document_id] = (lock, references + 1)
            self._active_documents[document_id] = self._active_documents.get(document_id, 0) + 1
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._state_lock:
                active = self._active_documents[document_id] - 1
                if active:
                    self._active_documents[document_id] = active
                else:
                    self._active_documents.pop(document_id, None)
                current_lock, references = self._document_locks[document_id]
                if references == 1:
                    self._document_locks.pop(document_id, None)
                else:
                    self._document_locks[document_id] = (current_lock, references - 1)

    def _directory_size(self, directory: Path) -> int:
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    def cleanup(
        self,
        *,
        now: float | None = None,
        reserve_documents: int = 0,
        reserve_bytes: int = 0,
    ) -> list[str]:
        """Remove expired documents, then oldest documents until caps are met."""

        current_time = time.time() if now is None else now
        removed: list[str] = []
        with self._state_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            entries: list[tuple[str, Path, float, int]] = []
            for directory in self.root.iterdir():
                if not directory.is_dir() or not DOCUMENT_ID_PATTERN.fullmatch(directory.name):
                    continue
                if directory.name in self._active_documents:
                    continue
                try:
                    record = self.get(directory.name)
                    timestamp = record.last_accessed_at or record.created_at or directory.stat().st_mtime
                    size = self._directory_size(directory)
                except (DocumentNotFoundError, OSError):
                    timestamp = directory.stat().st_mtime
                    size = self._directory_size(directory)
                entries.append((directory.name, directory, timestamp, size))

            survivors: list[tuple[str, Path, float, int]] = []
            for entry in entries:
                if current_time - entry[2] >= self.document_ttl_seconds:
                    shutil.rmtree(entry[1], ignore_errors=True)
                    removed.append(entry[0])
                else:
                    survivors.append(entry)

            survivors.sort(key=lambda item: (item[2], item[0]))
            total_size = sum(item[3] for item in survivors)
            while survivors and (
                len(survivors) + reserve_documents > self.max_documents
                or total_size + reserve_bytes > self.max_storage_bytes
            ):
                document_id, directory, _, size = survivors.pop(0)
                shutil.rmtree(directory, ignore_errors=True)
                removed.append(document_id)
                total_size -= size
        return removed

    @staticmethod
    def _validate_upload_metadata(upload: UploadFile) -> None:
        filename = upload.filename or ""
        if Path(filename).suffix.casefold() != ".pdf":
            raise DocumentValidationError("invalid PDF")
        if (upload.content_type or "").casefold() not in PDF_CONTENT_TYPES:
            raise DocumentValidationError("invalid PDF")

    def _page_count(self, path: Path) -> int:
        try:
            with pymupdf.open(path) as document:
                pages = document.page_count
                if pages <= 0:
                    raise DocumentValidationError("invalid PDF")
                if pages > self.max_pdf_pages:
                    raise DocumentTooLargeError("PDF exceeds the maximum allowed page count.")
                for page in document:
                    if (
                        page.rect.width > MAX_PDF_PAGE_WIDTH_POINTS
                        or page.rect.height > MAX_PDF_PAGE_HEIGHT_POINTS
                    ):
                        raise DocumentTooLargeError("PDF page dimensions are too large.")
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            if isinstance(exc, DocumentTooLargeError):
                raise
            raise DocumentValidationError("invalid PDF") from exc
        return pages

    def discard(self, document_id: str) -> None:
        """Remove an unprepared document without accepting arbitrary paths."""
        directory = self._directory(document_id)
        with self._state_lock:
            if document_id in self._active_documents:
                raise RuntimeError("cannot discard an active document")
            shutil.rmtree(directory, ignore_errors=True)

    async def register(self, upload: UploadFile) -> RegisteredDocument:
        self._validate_upload_metadata(upload)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        digest = hashlib.sha256()
        size = 0
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".upload-", suffix=".tmp", dir=self.root, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                first = True
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise DocumentTooLargeError("PDF upload is too large")
                    if first:
                        first = False
                        if not chunk.startswith(b"%PDF-"):
                            raise DocumentValidationError("invalid PDF")
                    digest.update(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
            if size == 0:
                raise DocumentValidationError("invalid PDF")

            pages = self._page_count(temporary_path)
            document_id = digest.hexdigest()
            directory = self._directory(document_id)
            pdf_path = directory / "document.pdf"
            metadata_path = directory / "metadata.json"
            cache_dir = directory / "cache"
            with self.processing(document_id):
                existing = pdf_path.is_file() and metadata_path.is_file()
                self.cleanup(
                    reserve_documents=0 if existing else 1,
                    reserve_bytes=0 if existing else size,
                )
                directory.mkdir(parents=True, exist_ok=True)
                if not pdf_path.is_file():
                    os.replace(temporary_path, pdf_path)
                    temporary_path = None
                now = time.time()
                record = DocumentRecord(
                    document_id=document_id,
                    filename=Path(upload.filename or "document.pdf").name,
                    pages=pages,
                    created_at=now,
                    last_accessed_at=now,
                )
                if not metadata_path.is_file():
                    self._write_metadata(record)
                else:
                    record = self.get(document_id, touch=True)
                cache_hit = existing and any(cache_dir.glob("*.npz"))
            return RegisteredDocument(record, pdf_path, cache_dir, cache_hit)
        finally:
            await upload.close()
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
