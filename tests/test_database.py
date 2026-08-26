from __future__ import annotations

import uuid
from unittest.mock import Mock, patch

from furiosa_rag.web.database import (
    ChatLogRecord,
    SqlAlchemyChatLogRepository,
    _sqlalchemy_url,
    chat_logs,
    create_chat_log_repository,
)


def test_database_url_absence_disables_persistence(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert create_chat_log_repository() is None


def test_render_postgres_url_uses_psycopg_driver() -> None:
    assert _sqlalchemy_url("postgres://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )
    assert _sqlalchemy_url("postgresql://user:pass@host/db") == (
        "postgresql+psycopg://user:pass@host/db"
    )


def test_repository_configuration_does_not_connect_eagerly() -> None:
    engine = Mock()
    with patch("furiosa_rag.web.database.create_engine", return_value=engine) as create_engine:
        repository = create_chat_log_repository("postgresql://user:pass@host/db")

    assert isinstance(repository, SqlAlchemyChatLogRepository)
    create_engine.assert_called_once_with(
        "postgresql+psycopg://user:pass@host/db", pool_pre_ping=True
    )


def test_chat_log_schema_contains_only_approved_payload_columns_and_indexes() -> None:
    assert set(chat_logs.c.keys()) == {
        "id",
        "session_id",
        "document_id",
        "filename",
        "question",
        "answer",
        "route",
        "routing_reason",
        "vision_used",
        "vision_available",
        "fallback_used",
        "sources",
        "latency_ms",
        "created_at",
    }
    assert {index.name for index in chat_logs.indexes} == {
        "ix_chat_logs_session_id",
        "ix_chat_logs_created_at",
        "ix_chat_logs_document_id",
    }
    assert chat_logs.c.created_at.server_default is not None


def test_repository_persists_slots_record_without_serializing_extra_data() -> None:
    connection = Mock()
    transaction = Mock()
    transaction.__enter__ = Mock(return_value=connection)
    transaction.__exit__ = Mock(return_value=False)
    engine = Mock()
    engine.begin.return_value = transaction
    repository = SqlAlchemyChatLogRepository(engine)
    record = ChatLogRecord.create(
        session_id=uuid.uuid4(),
        document_id="document-id",
        filename="paper.pdf",
        question="question",
        answer="answer",
        route="TEXT_ONLY",
        routing_reason="text route",
        vision_used=False,
        vision_available=False,
        fallback_used=False,
        sources=[{"page": 1, "chunk": "page-1-chunk-1"}],
        latency_ms={"total": 12.5},
    )

    repository.persist(record)

    connection.execute.assert_called_once()
    parameters = connection.execute.call_args.args[0].compile().params
    assert parameters["session_id"] == record.session_id
    assert parameters["question"] == "question"
    assert "api_key" not in parameters
    assert "client_ip" not in parameters
