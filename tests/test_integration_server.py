"""Needs Qdrant running (docker compose up -d qdrant). Run with: uv run pytest -m server"""

import uuid
import warnings

import pytest
from qdrant_client.http.exceptions import ApiException

from app.core.config import Settings
from app.rag.retriever import Retriever
from app.rag.store import connect
from ingestion.index import index_chunks

pytestmark = [pytest.mark.integration, pytest.mark.server]

LOCAL_QDRANT_URL = "http://127.0.0.1:6333"


@pytest.fixture
def server_client():
    client = connect(Settings(_env_file=None, qdrant_url=LOCAL_QDRANT_URL))
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def collection(server_client):
    name = f"labor_law_vn__pytest_{uuid.uuid4().hex}"
    yield name
    try:  # cleanup must not hide the failure of the test itself
        server_client.delete_collection(name)
    except ApiException as error:
        warnings.warn(f"could not delete the test collection {name}: {error}", stacklevel=1)


def test_round_trip_through_the_qdrant_server(
    server_client, collection, sample_chunks, fake_embedder
):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    assert index_chunks(sample_chunks, vectors, server_client, collection) == 12
    results = Retriever(server_client, fake_embedder, collection).retrieve(
        sample_chunks[4].context_text, k=1
    )
    assert results[0].chunk == sample_chunks[4]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
