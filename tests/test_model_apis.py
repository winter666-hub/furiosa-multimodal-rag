from __future__ import annotations

from unittest.mock import Mock

from furiosa_rag.config import ModelEndpoint
from furiosa_rag.embedding import FuriosaEmbedding
from furiosa_rag.llm import FuriosaLlm
from furiosa_rag.reranker import FuriosaReranker


def test_llm_calls_chat_completions() -> None:
    client = Mock()
    client.post_json.return_value = {"choices": [{"message": {"content": "pong"}}]}
    result = FuriosaLlm(ModelEndpoint("llm", "http://llm/v1", "qwen"), client).generate("ping")
    assert result == "pong"
    assert client.post_json.call_args.args[1] == "chat/completions"


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

