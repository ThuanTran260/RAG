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


INDEX_COMMAND = "uv run python -m ingestion.index"


def rejection_message(error: UnexpectedResponse, *, searching: bool) -> str:
    """One safe line for a Qdrant 4xx: the status and a hint, never the response content.

    qdrant_errors re-raises 4xx so that callers decide; the command line tools use this to
    report them. ``searching`` adds the index command where reindexing could be the cure.
    """
    status = error.status_code
    head = (
        "Qdrant rejected the request"
        if status is None
        else f"Qdrant rejected the request (HTTP {status})"
    )
    if status in (400, 422):
        if searching:
            hint = (
                "Check that the collection matches the embedding model; "
                f"if not, reindex with: {INDEX_COMMAND}"
            )
        else:
            hint = "Check QDRANT_COLLECTION and EMBEDDING_MODEL, then rerun this command."
    elif status in (401, 403):
        hint = "Check the access configuration of the Qdrant server."
    elif status == 404:
        if searching:
            hint = (
                "Check that QDRANT_URL points to a Qdrant server "
                f"and that the collection exists; create it with: {INDEX_COMMAND}"
            )
        else:
            hint = "Check that QDRANT_URL points to a Qdrant server."
    elif status == 429:
        hint = "Retry later."
    else:
        hint = "Check QDRANT_URL and the Qdrant configuration."
    return f"{head}. {hint}"


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
