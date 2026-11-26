import hashlib
import json
import os
import subprocess
import sys
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, Modifier

from app.core.config import REPO_ROOT, Settings, model_key
from app.rag.embedder import EmbeddingModelUnavailableError
from app.rag.sparse import average_length, document_terms, document_vector, term_id
from app.rag.store import QdrantUnavailableError, hybrid_collection_problem
from ingestion import index
from ingestion.index import index_chunks, index_hybrid_chunks, load_chunks, point_id

CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"

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


def test_main_reports_a_missing_model_cache_and_closes_the_client(
    wired, monkeypatch, tmp_path, capsys
):
    closed = []
    monkeypatch.setattr(wired, "close", lambda: closed.append(True))

    def unavailable(_settings):
        raise EmbeddingModelUnavailableError(
            "embedding model x is not in the local cache. "
            "Download it once with: uv run python -m ingestion.download_model"
        )

    monkeypatch.setattr(index, "build_embedder", unavailable)
    assert index.main(["--cache-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "uv run python -m ingestion.download_model" in captured.err
    assert "Traceback" not in captured.err
    assert closed == [True]


def build_hybrid(chunks, embedder, client, collection="hybrid", vectors=None):
    if vectors is None:
        vectors = vectors_for(chunks, embedder)
    return index_hybrid_chunks(
        chunks,
        vectors,
        client,
        collection,
        model_name=embedder.model_name,
        chunks_sha256="a" * 64,
    )


def stored_points(client, collection):
    points, _ = client.scroll(collection, limit=1000, with_payload=True, with_vectors=True)
    return {point.id: point for point in points}


def test_hybrid_schema_is_an_unnamed_dense_vector_and_an_idf_sparse_vector(
    client, sample_chunks, fake_embedder
):
    build_hybrid(sample_chunks, fake_embedder, client)
    params = client.get_collection("hybrid").config.params
    assert (params.vectors.size, params.vectors.distance) == (8, Distance.COSINE)
    assert list(params.sparse_vectors) == ["bm25_ub"]
    assert params.sparse_vectors["bm25_ub"].modifier == Modifier.IDF


def test_hybrid_metadata_describes_the_build(client, sample_chunks, fake_embedder):
    returned = build_hybrid(sample_chunks, fake_embedder, client)
    expected = {
        "schema_version": 1,
        "model_key": model_key("fake/model"),
        "chunks_sha256": "a" * 64,
        "points": 12,
        "tokenizer_version": "vi-syl-bigram-v1",
        "k1": 1.2,
        "b": 0.75,
        "avgdl": average_length(chunk.context_text for chunk in sample_chunks),
    }
    assert returned == expected
    assert client.get_collection("hybrid").config.metadata == expected


def test_hybrid_points_keep_the_dense_vector_and_payload_of_the_dense_collection(
    client, sample_chunks, fake_embedder
):
    vectors = vectors_for(sample_chunks, fake_embedder)
    index_chunks(sample_chunks, vectors, client, "dense")
    build_hybrid(sample_chunks, fake_embedder, client, vectors=vectors)

    dense = stored_points(client, "dense")
    hybrid = stored_points(client, "hybrid")

    assert set(hybrid) == set(dense) == {point_id(chunk.id) for chunk in sample_chunks}
    for key, point in dense.items():
        assert hybrid[key].vector[""] == point.vector
        assert hybrid[key].payload == point.payload


def test_hybrid_sparse_weights_are_bm25_over_the_context_text(client, sample_chunks, fake_embedder):
    build_hybrid(sample_chunks, fake_embedder, client)
    stored = stored_points(client, "hybrid")
    avgdl = average_length(chunk.context_text for chunk in sample_chunks)

    for chunk in sample_chunks:
        sparse = stored[point_id(chunk.id)].vector["bm25_ub"]
        expected = document_vector(chunk.context_text, avgdl)
        assert tuple(sparse.indices) == expected.indices
        assert sparse.values == pytest.approx(expected.values)

    # One weight by hand, so a wrong text field or avgdl cannot hide behind the same helper.
    chunk = sample_chunks[5]
    assert chunk.context_text != chunk.text
    terms = document_terms(chunk.context_text)
    term = max(set(terms), key=terms.count)
    tf = terms.count(term)
    weight = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(terms) / avgdl))
    sparse = stored[point_id(chunk.id)].vector["bm25_ub"]
    by_id = dict(zip(sparse.indices, sparse.values, strict=True))
    assert by_id[term_id(term)] == pytest.approx(weight)


def test_hybrid_indexing_leaves_the_dense_collection_untouched(
    client, sample_chunks, fake_embedder
):
    index_chunks(sample_chunks, vectors_for(sample_chunks, fake_embedder), client, "dense")
    config = client.get_collection("dense").config.model_dump()
    points = stored_points(client, "dense")

    build_hybrid(sample_chunks, fake_embedder, client)

    assert client.get_collection("dense").config.model_dump() == config
    assert stored_points(client, "dense") == points


def test_hybrid_indexing_again_replaces_the_collection_and_its_metadata(
    client, sample_chunks, fake_embedder
):
    build_hybrid(sample_chunks, fake_embedder, client)
    fewer = sample_chunks[:5]

    metadata = build_hybrid(fewer, fake_embedder, client)

    assert client.count("hybrid", exact=True).count == metadata["points"] == 5
    assert client.get_collection("hybrid").config.metadata == metadata
    assert metadata["avgdl"] == average_length(chunk.context_text for chunk in fewer)


@pytest.mark.parametrize("case", BAD_INPUTS)
def test_invalid_input_keeps_the_existing_hybrid_collection(
    client, sample_chunks, fake_embedder, case
):
    metadata = build_hybrid(sample_chunks, fake_embedder, client)
    good = vectors_for(sample_chunks, fake_embedder)
    chunks, vectors, message = bad_input(case, sample_chunks, good)

    with pytest.raises(ValueError, match=message):
        build_hybrid(chunks, fake_embedder, client, vectors=vectors)

    assert client.count("hybrid", exact=True).count == 12
    assert client.get_collection("hybrid").config.metadata == metadata


@pytest.mark.parametrize(
    "method",
    ["collection_exists", "create_collection", "upsert", "count", "update_collection"],
)
def test_transport_failures_while_writing_the_hybrid_collection(
    client, sample_chunks, fake_embedder, monkeypatch, method
):
    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, method, fail)
    with pytest.raises(QdrantUnavailableError, match="Qdrant request failed"):
        build_hybrid(sample_chunks, fake_embedder, client)


@pytest.mark.parametrize("method", ["upsert", "count", "update_collection"])
def test_an_interrupted_hybrid_build_fails_the_metadata_check(
    client, sample_chunks, fake_embedder, monkeypatch, method
):
    def fail(*args, **kwargs):
        raise ResponseHandlingException(OSError("connection reset"))

    monkeypatch.setattr(client, method, fail)
    with pytest.raises(QdrantUnavailableError):
        build_hybrid(sample_chunks, fake_embedder, client)

    problem = hybrid_collection_problem(client.get_collection("hybrid"), "hybrid")
    assert "uv run python -m ingestion.index --hybrid" in problem


def test_main_builds_only_the_hybrid_collection(wired, monkeypatch, tmp_path, capsys):
    settings = Settings(_env_file=None)
    monkeypatch.setattr(wired, "close", lambda: None)

    assert index.main(["--hybrid", "--cache-dir", str(tmp_path)]) == 0

    collection = settings.hybrid_collection_name
    metadata = wired.get_collection(collection).config.metadata
    assert metadata["chunks_sha256"] == hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    assert metadata["model_key"] == model_key(settings.embedding_model)
    assert metadata["points"] == 676
    assert wired.count(collection, exact=True).count == 676
    assert not wired.collection_exists(settings.collection_name)
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        f"collection: {collection}",
        "points: 676",
        "dimension: 8",
        "sparse vector: bm25_ub (modifier idf)",
    ]
    assert json.loads(lines[4].removeprefix("metadata: ")) == metadata
    assert len(lines) == 5


@pytest.mark.parametrize(
    "error",
    [
        ResponseHandlingException(OSError("connection reset")),
        UnexpectedResponse(400, "Bad Request", b"RESPONSE-BODY-MARKER", {}),
    ],
)
def test_main_reports_a_failed_hybrid_build(wired, monkeypatch, tmp_path, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(wired, "upsert", fail)

    assert index.main(["--hybrid", "--cache-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "may be incomplete" in captured.err
    assert "MARKER" not in captured.err
    assert "Traceback" not in captured.err
    if isinstance(error, ResponseHandlingException):
        assert "rerun: uv run python -m ingestion.index --hybrid" in captured.err


# The child process gets no fixtures, so it wires the in-memory Qdrant and a fake model itself,
# and it must not read the developer's .env.
CONSOLE_SCRIPT = """
import sys
import numpy as np
from qdrant_client import QdrantClient
from app.core.config import Settings
from ingestion import index


class Embedder:
    model_name = "fake/model"
    dimension = 4

    def embed_documents(self, texts):
        return np.ones((len(texts), self.dimension), dtype=np.float32)


Settings.model_config["env_file"] = None
index.connect = lambda _settings: QdrantClient(":memory:")
index.build_embedder = lambda _settings: Embedder()
raise SystemExit(index.main(sys.argv[1:]))
"""


def run_on_a_cp1252_console(arguments):
    env = os.environ | {"PYTHONIOENCODING": "cp1252", "QDRANT_COLLECTION": "luật_lao_động"}
    return subprocess.run(
        [sys.executable, "-c", CONSOLE_SCRIPT, *arguments],
        check=False,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
    )


def test_the_hybrid_command_survives_a_legacy_console_encoding(tmp_path):
    result = run_on_a_cp1252_console(["--hybrid", "--cache-dir", str(tmp_path)])
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "collection: luật_lao_động__" in result.stdout.decode("utf-8")

    missing = tmp_path / "thiếu-điều.jsonl"
    result = run_on_a_cp1252_console(["--hybrid", "--chunks", str(missing)])
    assert result.returncode == 1
    assert "thiếu-điều.jsonl" in result.stderr.decode("utf-8")
