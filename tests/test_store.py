import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings
from app.rag import store
from app.rag.store import QdrantUnavailableError, connect, qdrant_errors


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
