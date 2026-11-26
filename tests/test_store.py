import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from app.core.config import Settings
from app.rag import store
from app.rag.store import (
    QdrantUnavailableError,
    connect,
    hybrid_collection_problem,
    qdrant_errors,
)
from ingestion.index import index_chunks, index_hybrid_chunks


def test_the_connection_probe_keeps_the_specific_hint():
    with pytest.raises(QdrantUnavailableError) as info, qdrant_errors("http://127.0.0.1:6333"):
        raise ResponseHandlingException(OSError("connection reset"))
    assert str(info.value) == (
        "Qdrant is not reachable at http://127.0.0.1:6333. "
        "Start it with: docker compose up -d qdrant"
    )


def test_other_transport_failures_use_the_generic_wording():
    expected = r"Qdrant request failed \(OSError\)"
    with pytest.raises(QdrantUnavailableError, match=expected), qdrant_errors():
        raise ResponseHandlingException(OSError("connection reset"))


def test_server_errors_become_qdrant_unavailable():
    failure = UnexpectedResponse(503, "Service Unavailable", b"", {})
    with pytest.raises(QdrantUnavailableError, match="HTTP 503"), qdrant_errors():
        raise failure


def test_client_errors_are_not_swallowed():
    failure = UnexpectedResponse(400, "Bad Request", b"", {})
    with pytest.raises(UnexpectedResponse), qdrant_errors():
        raise failure


@pytest.mark.parametrize("error", [ValueError("bad input"), RuntimeError("bug")])
def test_other_errors_are_not_swallowed(error):
    with pytest.raises(type(error)), qdrant_errors():
        raise error


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_connect_reports_an_unreachable_server():
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    with pytest.raises(QdrantUnavailableError, match="docker compose up -d qdrant"):
        connect(settings)


BODY_MARKER = "RESPONSE-BODY-MARKER"
HEADER_MARKER = "RESPONSE-HEADER-MARKER"
INDEX_COMMAND = "uv run python -m ingestion.index"


def rejection(status):
    return UnexpectedResponse(
        status, "Rejected", BODY_MARKER.encode(), {"authorization": HEADER_MARKER}
    )


@pytest.mark.parametrize("searching", [True, False])
@pytest.mark.parametrize("status", [None, 400, 401, 403, 404, 418, 422, 429])
def test_a_rejection_message_never_exposes_the_response(status, searching):
    message = store.rejection_message(rejection(status), searching=searching)
    assert "\n" not in message
    assert BODY_MARKER not in message
    assert HEADER_MARKER not in message
    if status is not None:
        assert f"HTTP {status}" in message


@pytest.mark.parametrize("searching", [True, False])
@pytest.mark.parametrize("status", [401, 403])
def test_an_access_rejection_is_not_fixed_by_reindexing(status, searching):
    message = store.rejection_message(rejection(status), searching=searching)
    assert "access" in message
    assert INDEX_COMMAND not in message


@pytest.mark.parametrize("status", [400, 404, 422])
def test_a_rejection_of_a_search_points_to_the_index_command(status):
    assert INDEX_COMMAND in store.rejection_message(rejection(status), searching=True)


def test_a_wrong_service_is_blamed_on_the_url_not_the_collection():
    message = store.rejection_message(rejection(404), searching=False)
    assert "QDRANT_URL" in message
    assert INDEX_COMMAND not in message


def test_a_rate_limit_asks_to_retry_later():
    assert "Retry later" in store.rejection_message(rejection(429), searching=True)


HYBRID_COMMAND = "uv run python -m ingestion.index --hybrid"


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def hybrid(client, sample_chunks, fake_embedder):
    """The client, holding a freshly built hybrid collection named "hybrid"."""
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    index_hybrid_chunks(
        sample_chunks,
        vectors,
        client,
        "hybrid",
        model_name=fake_embedder.model_name,
        chunks_sha256="a" * 64,
    )
    return client


def problem_of(client, collection):
    return hybrid_collection_problem(client.get_collection(collection), collection)


def assert_hint(problem, collection):
    assert collection in problem
    assert problem.endswith(HYBRID_COMMAND)
    assert "\n" not in problem


def test_a_fresh_hybrid_collection_has_no_problem(hybrid):
    assert problem_of(hybrid, "hybrid") is None


def test_a_dense_collection_lacks_the_sparse_vector(client, sample_chunks, fake_embedder):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    index_chunks(sample_chunks, vectors, client, "dense")

    problem = problem_of(client, "dense")

    assert "bm25_ub" in problem
    assert_hint(problem, "dense")


@pytest.mark.parametrize(
    ("sparse_name", "modifier"),
    [
        ("other", Modifier.IDF),
        ("bm25_ub", None),  # the right name, but Qdrant would apply no IDF
    ],
)
def test_a_wrong_sparse_configuration_is_reported(client, sparse_name, modifier):
    client.create_collection(
        "wrong",
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
        sparse_vectors_config={sparse_name: SparseVectorParams(modifier=modifier)},
    )

    problem = problem_of(client, "wrong")

    assert "bm25_ub" in problem
    assert_hint(problem, "wrong")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tokenizer_version", "vi-syl-bigram-v0"),
        ("tokenizer_version", None),
        ("schema_version", 2),
        ("k1", 0.9),
        ("b", 0.5),
        ("points", 0),
        ("points", "12"),
        ("points", None),
    ],
)
def test_stale_or_broken_metadata_is_reported(hybrid, key, value):
    # update_collection merges, and None stands for a value the collection does not have.
    hybrid.update_collection("hybrid", metadata={key: value})

    problem = problem_of(hybrid, "hybrid")

    assert key in problem
    assert_hint(problem, "hybrid")


def test_a_collection_without_metadata_is_reported(client):
    client.create_collection(
        "bare",
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
        sparse_vectors_config={"bm25_ub": SparseVectorParams(modifier=Modifier.IDF)},
    )

    problem = problem_of(client, "bare")

    assert "tokenizer_version" in problem
    assert_hint(problem, "bare")
