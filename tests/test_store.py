import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings
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
