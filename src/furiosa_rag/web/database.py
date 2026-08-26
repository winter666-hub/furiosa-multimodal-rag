"""Optional PostgreSQL persistence for successful public-demo conversations."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import JSON, Boolean, Column, DateTime, Index, MetaData, String, Table, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Engine, create_engine

metadata = MetaData()
chat_logs = Table(
    "chat_logs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("session_id", UUID(as_uuid=True), nullable=False),
    Column("document_id", String, nullable=True),
    Column("filename", String, nullable=True),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("route", String, nullable=False),
    Column("routing_reason", Text, nullable=True),
    Column("vision_used", Boolean, nullable=False),
    Column("vision_available", Boolean, nullable=False),
    Column("fallback_used", Boolean, nullable=False),
    Column("sources", JSON().with_variant(JSONB, "postgresql"), nullable=False),
    Column("latency_ms", JSON().with_variant(JSONB, "postgresql"), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_chat_logs_session_id", chat_logs.c.session_id)
Index("ix_chat_logs_created_at", chat_logs.c.created_at)
Index("ix_chat_logs_document_id", chat_logs.c.document_id)


@dataclass(frozen=True, slots=True)
class ChatLogRecord:
    id: uuid.UUID
    session_id: uuid.UUID
    document_id: str | None
    filename: str | None
    question: str
    answer: str
    route: str
    routing_reason: str | None
    vision_used: bool
    vision_available: bool
    fallback_used: bool
    sources: list[dict[str, Any]]
    latency_ms: dict[str, float]
    created_at: datetime

    @classmethod
    def create(cls, **values: Any) -> ChatLogRecord:
        return cls(id=uuid.uuid4(), created_at=datetime.now(timezone.utc), **values)


class ChatLogRepository(Protocol):
    def initialize(self) -> None: ...

    def persist(self, record: ChatLogRecord) -> None: ...


class SqlAlchemyChatLogRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def initialize(self) -> None:
        metadata.create_all(self.engine)

    def persist(self, record: ChatLogRecord) -> None:
        with self.engine.begin() as connection:
            connection.execute(chat_logs.insert().values(**asdict(record)))


def _sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


def create_chat_log_repository(database_url: str | None = None) -> ChatLogRepository | None:
    configured_url = database_url if database_url is not None else os.getenv("DATABASE_URL", "")
    if not configured_url.strip():
        return None
    engine = create_engine(_sqlalchemy_url(configured_url.strip()), pool_pre_ping=True)
    return SqlAlchemyChatLogRepository(engine)
