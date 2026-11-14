import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import REPO_ROOT, Settings, model_key
from app.rag.store import QdrantUnavailableError
from ingestion import index
from ingestion.index import index_chunks, load_chunks, point_id

BAD_INPUTS = [
    "no_chunks",
    "flat_vectors",
    "too_few_vectors",
    "zero_columns",
    "integer_vectors",
    "nan_vector",
    "infinite_vector",
    "duplicate_chunk_ids",
]


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


def vectors_for(chunks, embedder):
    return embedder.embed_documents([chunk.context_text for chunk in chunks])


def bad_input(case, chunks, good):
    """Return (chunks, vectors, expected error message) for one invalid input."""
    poisoned = good.copy()
    if case == "no_chunks":
        return [], good[:0], "no chunks to index"
    if case == "flat_vectors":
        return chunks, good.reshape(-1), "2-D"
    if case == "too_few_vectors":
        return chunks, good[:3], "12 chunks but 3 vectors"
    if case == "zero_columns":
        return chunks, good[:, :0], "at least one column"
    if case == "integer_vectors":
        return chunks, good.astype(np.int64), "floating-point"
    if case == "nan_vector":
        poisoned[3, 2] = np.nan
        return chunks, poisoned, "NaN or infinite"
    if case == "infinite_vector":
        poisoned[4, 1] = np.inf
        return chunks, poisoned, "NaN or infinite"
    if case == "duplicate_chunk_ids":
        return [chunks[0], *chunks[:-1]], good, "duplicate chunk ids"
    raise AssertionError(case)


def test_point_id_is_a_stable_uuid():
    first = point_id("45-2019-QH14:d35:k1")
    assert first == point_id("45-2019-QH14:d35:k1")
    assert first != point_id("45-2019-QH14:d35:k2")
    assert str(uuid.UUID(first)) == first


def test_load_chunks_reads_the_committed_file():
    chunks = load_chunks(REPO_ROOT / "data" / "processed" / "chunks.jsonl")
    assert len(chunks) == 676
    assert chunks[0].id == "45-2019-QH14:d1"


def test_index_stores_every_chunk_with_its_payload(client, sample_chunks, fake_embedder):
    count = index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")
    assert count == 12
    chunk = sample_chunks[5]
    stored = client.retrieve("test", ids=[point_id(chunk.id)], with_payload=True)[0]
    assert stored.payload == chunk.model_dump(mode="json")


def test_indexing_again_replaces_the_collection(client, sample_chunks, fake_embedder):
    index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")
    fewer = sample_chunks[:5]
    assert index_chunks(fewer, vectors_for(fewer, fake_embedder), client, "test") == 5
    assert index_chunks(fewer, vectors_for(fewer, fake_embedder), client, "test") == 5


def test_collection_dimension_follows_the_vectors(client, sample_chunks, make_fake_embedder):
    index_chunks(
        sample_chunks, vectors_for(sample_chunks, make_fake_embedder(dimension=8)), client, "test"
    )
    wider = vectors_for(sample_chunks, make_fake_embedder(dimension=16))
    assert index_chunks(sample_chunks, wider, client, "test") == 12
    assert client.get_collection("test").config.params.vectors.size == 16


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_invalid_input_raises_and_keeps_the_existing_collection(
    client, sample_chunks, fake_embedder, case
):
    good = vectors_for(sample_chunks, fake_embedder)
    assert index_chunks(sample_chunks, good, client, "test") == 12
    chunks, vectors, message = bad_input(case, sample_chunks, good)

    with pytest.raises(ValueError, match=message):
        index_chunks(chunks, vectors, client, "test")

    assert client.count("test", exact=True).count == 12
    assert client.get_collection("test").config.params.vectors.size == 8
    stored = client.retrieve("test", ids=[point_id(sample_chunks[0].id)], with_payload=True)
    assert stored[0].payload == sample_chunks[0].model_dump(mode="json")


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_invalid_input_does_not_create_a_collection(client, sample_chunks, fake_embedder, case):
    good = vectors_for(sample_chunks, fake_embedder)
    chunks, vectors, message = bad_input(case, sample_chunks, good)
    with pytest.raises(ValueError, match=message):
        index_chunks(chunks, vectors, client, "test")
    assert not client.collection_exists("test")


@pytest.mark.parametrize("method", ["collection_exists", "create_collection", "upsert", "count"])
def test_transport_failures_while_writing_become_qdrant_unavailable(
    client, sample_chunks, fake_embedder, monkeypatch, method
):
    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, method, fail)
    with pytest.raises(QdrantUnavailableError, match="Qdrant request failed"):
        index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")


def test_programming_errors_while_writing_are_not_swallowed(
    client, sample_chunks, fake_embedder, monkeypatch
):
    def bug(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(client, "upsert", bug)
    with pytest.raises(RuntimeError, match="bug"):
        index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "test")


def test_main_indexes_all_chunks(monkeypatch, client, make_fake_embedder, tmp_path, capsys):
    settings = Settings(_env_file=None)
    embedder = make_fake_embedder(model_name=settings.embedding_model)
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "connect", lambda _settings: client)
    monkeypatch.setattr(index, "build_embedder", lambda _settings: embedder)
    # main() closes its client, and a closed in-memory client cannot be read back.
    monkeypatch.setattr(client, "close", lambda: None)

    assert index.main(["--cache-dir", str(tmp_path)]) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"collection: {settings.collection_name}",
        "points: 676",
        "dimension: 8",
    ]
    assert client.count(settings.collection_name, exact=True).count == 676
    assert (tmp_path / f"{model_key(settings.embedding_model)}.npz").is_file()


def test_main_reports_a_missing_chunks_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(index, "get_settings", lambda: Settings(_env_file=None))
    assert index.main(["--chunks", str(tmp_path / "missing.jsonl")]) == 1
    assert "uv run python -m ingestion.build" in capsys.readouterr().err


def test_main_reports_invalid_settings(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    monkeypatch.setattr(index, "connect", lambda _settings: pytest.fail("connected"))
    assert index.main(["--cache-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore:Failed to obtain server version")
def test_main_reports_an_unreachable_server(monkeypatch, tmp_path, capsys):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "build_embedder", lambda _settings: pytest.fail("model was loaded"))
    assert index.main(["--cache-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Qdrant is not reachable at http://127.0.0.1:9" in captured.err
    assert captured.out == ""


def test_main_reports_a_server_failure_during_the_rebuild(
    monkeypatch, client, make_fake_embedder, tmp_path, capsys
):
    settings = Settings(_env_file=None)

    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, "upsert", fail)
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "connect", lambda _settings: client)
    monkeypatch.setattr(
        index, "build_embedder", lambda _settings: make_fake_embedder(settings.embedding_model)
    )

    assert index.main(["--cache-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert "rerun: uv run python -m ingestion.index" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def reject_with(status):
    def reject(*args, **kwargs):
        raise UnexpectedResponse(
            status, "Rejected", b"RESPONSE-BODY-MARKER", {"authorization": "HEADER-MARKER"}
        )

    return reject


@pytest.fixture
def wired(monkeypatch, client, make_fake_embedder):
    """Run index.main() against an in-memory Qdrant and a fake embedder."""
    settings = Settings(_env_file=None)
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    monkeypatch.setattr(index, "connect", lambda _settings: client)
    monkeypatch.setattr(
        index, "build_embedder", lambda _settings: make_fake_embedder(settings.embedding_model)
    )
    return client


@pytest.mark.parametrize("phase", ["connect", "collection_exists", "upsert"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 418, 422, 429])
def test_main_reports_an_http_rejection(wired, monkeypatch, tmp_path, capsys, phase, status):
    if phase == "connect":
        monkeypatch.setattr(index, "connect", reject_with(status))
        monkeypatch.setattr(index, "build_embedder", lambda _s: pytest.fail("model was loaded"))
    else:
        monkeypatch.setattr(wired, phase, reject_with(status))

    assert index.main(["--cache-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"HTTP {status}" in captured.err
    assert "MARKER" not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("fail", ["none", "transport", "rejection", "bug"])
def test_main_closes_the_client_however_it_ends(wired, monkeypatch, tmp_path, fail):
    closed = []
    real_close = wired.close
    monkeypatch.setattr(wired, "close", lambda: (closed.append(True), real_close()))
    errors = {
        "transport": ResponseHandlingException(OSError("connection reset")),
        "rejection": UnexpectedResponse(400, "Bad Request", b"", {}),
        "bug": RuntimeError("bug"),
    }
    if fail != "none":

        def raise_error(*args, **kwargs):
            raise errors[fail]

        monkeypatch.setattr(wired, "upsert", raise_error)

    if fail == "bug":
        with pytest.raises(RuntimeError, match="bug"):
            index.main(["--cache-dir", str(tmp_path)])
    else:
        assert index.main(["--cache-dir", str(tmp_path)]) == (0 if fail == "none" else 1)
    assert closed == [True]
