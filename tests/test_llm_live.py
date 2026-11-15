"""Opt-in smoke tests against the real LLM provider; at most 2 requests per run.

Two gates keep these out of every default run: the ``llm`` marker (excluded by ``addopts``)
and ``RAG_LIVE_LLM=1``. Run on purpose with:

    RAG_LIVE_LLM=1 uv run --locked pytest -m llm tests/test_llm_live.py -q -rs

The key comes only from the environment or the user's own ``.env``; it is never printed.
The tests make no claim about legal meaning.
"""

import json
import os

import pytest

from app.core.config import Settings
from app.rag.generate import generate, source_label
from app.rag.llm import LLMRequest, build_llm_client
from app.rag.retriever import RetrievedChunk

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(os.environ.get("RAG_LIVE_LLM") != "1", reason="live LLM smoke is opt-in"),
]

TINY_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
REQUEST = LLMRequest(system="Return JSON.", user="ping", schema=TINY_SCHEMA)


def test_live_round_trip_returns_json_matching_the_tiny_schema():
    settings = Settings()
    if settings.llm_api_key is None:
        pytest.skip("LLM_API_KEY is not configured")
    client = build_llm_client(settings)
    result = client.complete(REQUEST)
    payload = json.loads(result.text)
    assert isinstance(payload, dict) and set(payload) == {"ok"}
    assert type(payload["ok"]) is bool
    assert result.requested_model == settings.llm_model  # response_model may be absent


def test_live_generate_on_a_public_chunk_satisfies_the_contract(all_chunks):
    settings = Settings()
    if settings.llm_api_key is None:
        pytest.skip("LLM_API_KEY is not configured")
    client = build_llm_client(settings)
    chunk = next(c for c in all_chunks if c.id == "45-2019-QH14:d1")  # verbatim corpus, unmodified
    contexts = [RetrievedChunk(chunk=chunk, score=1.0)]
    result = generate("Bộ luật Lao động điều chỉnh những gì?", contexts, client)
    assert result.status in {"answered", "insufficient"}
    if result.status == "answered":
        assert {s.source_id for s in result.sources} == {"S1"}
        assert result.sources[0].label == source_label(chunk)  # derived from metadata
        assert result.sources[0].text == chunk.text
