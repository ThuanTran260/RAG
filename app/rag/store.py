"""Qdrant connection and the single error boundary for every call to the server."""

from collections.abc import Iterator
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import Settings


class QdrantUnavailableError(RuntimeError):
    """Qdrant cannot be reached or answered with a server error."""


@contextmanager
def qdrant_errors(url: str | None = None) -> Iterator[None]:
    """The single place that turns Qdrant transport and 5xx failures into QdrantUnavailableError."""
    try:
        yield
    except ResponseHandlingException as error:
        # Also raised when a response cannot be parsed, so only the connection probe
        # (which passes the url) claims "not reachable"; other calls use the generic wording.
        if url:
            message = (
                f"Qdrant is not reachable at {url}. Start it with: docker compose up -d qdrant"
            )
        else:
            message = (
                f"Qdrant request failed ({type(error.source).__name__}). "
                "Check it with: docker compose ps qdrant"
            )
        raise QdrantUnavailableError(message) from error
    except UnexpectedResponse as error:
        if error.status_code is not None and error.status_code >= 500:
            raise QdrantUnavailableError(
                f"Qdrant answered HTTP {error.status_code}. "
                "Check it with: docker compose logs qdrant"
            ) from error
        raise


def connect(settings: Settings) -> QdrantClient:
    """Return a client for ``settings.qdrant_url`` after proving that the server answers."""
    # qdrant-client runs its version check in a background thread, so an unreachable server
    # makes it warn at a random later moment; the probe below already covers reachability.
    client = QdrantClient(url=settings.qdrant_url, timeout=10, check_compatibility=False)
    try:
        with qdrant_errors(settings.qdrant_url):
            client.get_collections()
    except BaseException:
        client.close()
        raise
    return client
