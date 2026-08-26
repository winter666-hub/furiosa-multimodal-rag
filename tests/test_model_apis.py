from __future__ import annotations

from unittest.mock import Mock

import pytest

from furiosa_rag.clients import FuriosaApiError
from furiosa_rag.config import ModelEndpoint
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.reranker import FuriosaReranker
from furiosa_rag.vision import FuriosaVision


def test_llm_calls_chat_completions() -> None:
    client = Mock()
    client.post_json.return_value = {"choices": [{"message": {"content": "pong"}}]}
    result = FuriosaLlm(ModelEndpoint("llm", "http://llm/v1", "qwen"), client).generate("ping")
    assert result == "pong"
    assert client.post_json.call_args.args[1] == "chat/completions"


def test_llm_preserves_length_limited_answer_and_logs_safe_diagnostic(caplog) -> None:
    client = Mock()
    truncated_answer = "SQuAD v2.0 test F1 was 83.1, which is 5."
    client.post_json.return_value = {
        "choices": [
            {
                "message": {"content": truncated_answer},
                "finish_reason": "length",
            }
        ]
    }
    backend = FuriosaLlm(ModelEndpoint("llm", "http://llm/v1", "qwen"), client)

    with caplog.at_level("WARNING", logger="furiosa_rag.llm"):
        result = backend.generate("private question", max_tokens=768)

    assert result == truncated_answer
    payload = client.post_json.call_args.args[2]
    assert payload["max_tokens"] == 768
    assert "reached max_tokens" in caplog.text
    assert "max_tokens=768" in caplog.text
    assert "private question" not in caplog.text
    assert truncated_answer not in caplog.text


def test_embedding_preserves_input_order() -> None:
    client = Mock()
    client.post_json.return_value = {
        "data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]
    }
    result = FuriosaEmbedding(
        ModelEndpoint("embedding", "http://embedding/v1", "qwen-embedding"), client
    ).embed(["a", "b"])
    assert result == [[0.1], [0.2]]


def test_reranker_maps_scores_back_to_documents() -> None:
    client = Mock()
    client.post_json.return_value = {"results": [{"index": 1, "relevance_score": 0.9}]}
    result = FuriosaReranker(
        ModelEndpoint("reranker", "http://reranker/v1", "qwen-reranker"), client
    ).rerank("query", ["first", "second"], top_n=1)
    assert result[0].text == "second"
    assert result[0].score == 0.9


@pytest.mark.parametrize(
    "indices",
    ([0, 0], [0], [-1, 1], [0, 2]),
    ids=("duplicate", "missing", "negative", "out-of-range"),
)
def test_embedding_rejects_invalid_indices(indices: list[int]) -> None:
    client = Mock()
    client.post_json.return_value = {
        "data": [{"index": index, "embedding": [0.1]} for index in indices]
    }
    backend = FuriosaEmbedding(ModelEndpoint("embedding", "http://embedding", "model"), client)
    with pytest.raises(FuriosaApiError):
        backend.embed(["a", "b"])


def test_embedding_rejects_inconsistent_dimensions_and_non_finite_values() -> None:
    client = Mock()
    backend = FuriosaEmbedding(ModelEndpoint("embedding", "http://embedding", "model"), client)
    client.post_json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1]},
            {"index": 1, "embedding": [0.2, 0.3]},
        ]
    }
    with pytest.raises(FuriosaApiError, match="consistent"):
        backend.embed(["a", "b"])

    client.post_json.return_value = {"data": [{"index": 0, "embedding": [float("nan")]}]}
    with pytest.raises(FuriosaApiError, match="finite"):
        backend.embed(["a"])


@pytest.mark.parametrize("indices", ([0, 0], [-1], [2]), ids=("duplicate", "negative", "range"))
def test_reranker_rejects_invalid_indices(indices: list[int]) -> None:
    client = Mock()
    client.post_json.return_value = {
        "results": [{"index": index, "relevance_score": 0.5} for index in indices]
    }
    backend = FuriosaReranker(ModelEndpoint("reranker", "http://reranker", "model"), client)
    with pytest.raises(FuriosaApiError):
        backend.rerank("query", ["first", "second"], top_n=2)


def test_vision_uses_configured_endpoint_model_and_image_payload() -> None:
    client = Mock()
    client.post_json.return_value = {"choices": [{"message": {"content": "visual"}}]}
    endpoint = ModelEndpoint("vision", "http://localhost:8000/v1", "qwen-vl")
    result = FuriosaVision(endpoint, client).analyze("한국어 질문", "data:image/png;base64,cG5n")

    assert result == "visual"
    assert client.post_json.call_args.args[:2] == ("http://localhost:8000/v1", "chat/completions")
    payload = client.post_json.call_args.args[2]
    assert payload["model"] == "qwen-vl"
    assert payload["max_tokens"] == 256
    assert payload["messages"][0]["role"] == "system"
    assert "untrusted evidence" in payload["messages"][0]["content"]
    user_content = payload["messages"][1]["content"]
    assert user_content[0]["image_url"]["url"].endswith("cG5n")
    assert "한국어 질문" in user_content[1]["text"]
    assert "Do not summarize the entire page" in user_content[1]["text"]
    assert "at most 5 bullet points" in user_content[1]["text"]
