"""Ephemeral, document-isolated PDF storage for the web application."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from fastapi import UploadFile

DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}


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


@dataclass(frozen=True, slots=True)
class RegisteredDocument:
    record: DocumentRecord
    pdf_path: Path
    cache_dir: Path
    cache_hit: bool


class DocumentStore:
    def __init__(self, root: str | Path, *, max_upload_bytes: int) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        self.root = Path(root)
        self.max_upload_bytes = max_upload_bytes

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

    def get(self, document_id: str) -> DocumentRecord:
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
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise DocumentNotFoundError("document not found") from exc
        if record.document_id != document_id or record.pages <= 0:
            raise DocumentNotFoundError("document not found")
        return record

    @staticmethod
    def _validate_upload_metadata(upload: UploadFile) -> None:
        filename = upload.filename or ""
        if Path(filename).suffix.casefold() != ".pdf":
            raise DocumentValidationError("invalid PDF")
        if (upload.content_type or "").casefold() not in PDF_CONTENT_TYPES:
            raise DocumentValidationError("invalid PDF")

    @staticmethod
    def _page_count(path: Path) -> int:
        try:
            with pymupdf.open(path) as document:
                pages = document.page_count
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise DocumentValidationError("invalid PDF") from exc
        if pages <= 0:
            raise DocumentValidationError("invalid PDF")
        return pages

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
            existing = pdf_path.is_file() and metadata_path.is_file()
            directory.mkdir(parents=True, exist_ok=True)
            if not pdf_path.is_file():
                os.replace(temporary_path, pdf_path)
                temporary_path = None
            record = DocumentRecord(
                document_id=document_id,
                filename=Path(upload.filename or "document.pdf").name,
                pages=pages,
            )
            if not metadata_path.is_file():
                metadata_temp = directory / ".metadata.tmp"
                metadata_temp.write_text(json.dumps(asdict(record)), encoding="utf-8")
                os.replace(metadata_temp, metadata_path)
            else:
                record = self.get(document_id)
            cache_hit = existing and any(cache_dir.glob("*.npz"))
            return RegisteredDocument(record, pdf_path, cache_dir, cache_hit)
        finally:
            await upload.close()
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
