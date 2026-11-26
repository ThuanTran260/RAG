"""Qdrant connection and the single error boundary for every call to the server."""

from collections.abc import Iterator
from contextlib import contextmanager

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import CollectionInfo, Modifier

from app.core.config import HYBRID_SCHEMA_VERSION, Settings
from app.rag.sparse import K1, TOKENIZER_VERSION, B


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
HYBRID_INDEX_COMMAND = f"{INDEX_COMMAND} --hybrid"
SPARSE_VECTOR_NAME = "bm25_ub"  # BM25 over unigrams and bigrams


def hybrid_collection_problem(info: CollectionInfo, collection: str) -> str | None:
    """None when this code can query ``info`` as a hybrid collection, else a one-line hint.

    ``info`` is what ``get_collection(collection)`` returned. The hint ends with the command that
    rebuilds the collection. The index command writes the metadata last, so an interrupted build
    is reported too.
    """
    problem = _hybrid_problem(info)
    if problem is None:
        return None
    return f"Collection {collection} {problem}. Index lại bằng: {HYBRID_INDEX_COMMAND}"


def _hybrid_problem(info: CollectionInfo) -> str | None:
    sparse = (info.config.params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
    if sparse is None:
        return f"không có sparse vector {SPARSE_VECTOR_NAME}"
    if sparse.modifier != Modifier.IDF:
        return f"có sparse vector {SPARSE_VECTOR_NAME} nhưng thiếu modifier IDF"
    metadata = info.config.metadata or {}
    # The tokenizer comes first: it is the rule most likely to change between two commits.
    expected = {
        "tokenizer_version": TOKENIZER_VERSION,
        "schema_version": HYBRID_SCHEMA_VERSION,
        "k1": K1,
        "b": B,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            return f"có metadata {key} = {metadata.get(key)!r}, code hiện tại cần {value!r}"
    # Queries read the whole ranking, so the stored point count sets their depth.
    points = metadata.get("points")
    if type(points) is not int or points < 1:
        return f"có metadata points = {points!r}, cần một số nguyên dương"
    return None


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
