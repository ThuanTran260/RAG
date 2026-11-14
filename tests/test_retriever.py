import unicodedata
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.config import Settings
from app.rag import retriever as retriever_module
from app.rag import search
from app.rag.retriever import (
    CollectionNotIndexedError,
    InvalidQueryError,
    RetrievedChunk,
    Retriever,
    default_retriever,
    retrieve,
)
from app.rag.store import QdrantUnavailableError
from ingestion.index import index_chunks


class NoEmbedder:
    """An embedder that must never be reached."""

    def embed_query(self, text):
        pytest.fail("the query was embedded")


def fail_transport(*args, **kwargs):
    raise ResponseHandlingException(OSError("connection reset"))


@pytest.fixture(autouse=True)
def fresh_default_retriever():
    default_retriever.cache_clear()
    yield
    default_retriever.cache_clear()


@pytest.fixture
def client():
    qdrant = QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture
def retriever(client, sample_chunks, fake_embedder):
    vectors = fake_embedder.embed_documents([chunk.context_text for chunk in sample_chunks])
    index_chunks(sample_chunks, vectors, client, "test")
    return Retriever(client, fake_embedder, "test")


@pytest.fixture
def wired(monkeypatch, client, make_fake_embedder):
    """Point default_retriever() at an in-memory Qdrant and a fake embedder, counting use."""
    settings = Settings(_env_file=None)
    state = SimpleNamespace(
        settings=settings,
        client=client,
        embedder=make_fake_embedder(model_name=settings.embedding_model),
        connects=0,
        model_loads=0,
    )

    def fake_connect(_settings):
        state.connects += 1
        return state.client

    def fake_build_embedder(_settings):
        state.model_loads += 1
        return state.embedder

    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever_module, "connect", fake_connect)
    monkeypatch.setattr(retriever_module, "build_embedder", fake_build_embedder)
    return state


@pytest.fixture
def indexed(wired, sample_chunks):
    texts = [chunk.context_text for chunk in sample_chunks]
    vectors = wired.embedder.embed_documents(texts)
    index_chunks(sample_chunks, vectors, wired.client, wired.settings.collection_name)
    return wired


# --- Retriever class -------------------------------------------------------------


def test_the_identical_text_is_the_top_hit(retriever, sample_chunks):
    target = sample_chunks[7]
    results = retriever.retrieve(target.context_text, k=3)
    assert len(results) == 3
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk == target
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_results_are_sorted_best_first(retriever, sample_chunks):
    scores = [result.score for result in retriever.retrieve(sample_chunks[0].context_text, k=12)]
    assert scores == sorted(scores, reverse=True)


def test_default_k_is_five(retriever, sample_chunks):
    assert len(retriever.retrieve(sample_chunks[0].context_text)) == 5


def test_k_larger_than_the_collection_returns_every_chunk(retriever, sample_chunks):
    results = retriever.retrieve(sample_chunks[0].context_text, k=100)
    assert {result.chunk.id for result in results} == {chunk.id for chunk in sample_chunks}


def test_query_is_stripped_and_normalised_to_nfc(retriever, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert decomposed != target.context_text
    results = retriever.retrieve(f"  {decomposed}\n", k=1)
    assert results[0].chunk.id == target.id
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_raises(retriever, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retriever.retrieve(query)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(retriever, k):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retriever.retrieve("thử việc", k=k)


def test_missing_collection_raises_before_the_query_is_embedded(client):
    retriever = Retriever(client, NoEmbedder(), "labor_law_vn__missing")
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retriever.retrieve("thử việc")


@pytest.mark.parametrize("method", ["collection_exists", "query_points"])
def test_a_server_failure_becomes_qdrant_unavailable(retriever, monkeypatch, method):
    monkeypatch.setattr(retriever.client, method, fail_transport)
    with pytest.raises(QdrantUnavailableError):
        retriever.retrieve("thử việc")


# --- public retrieve() and default_retriever() -----------------------------------


def test_retrieve_returns_results_and_reuses_the_cached_retriever(indexed, sample_chunks):
    target = sample_chunks[7]
    first = retrieve(target.context_text, k=2)
    second = retrieve(target.context_text, k=2)
    assert first[0].chunk == target
    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert indexed.connects == 1
    assert indexed.model_loads == 1


def test_retrieve_normalises_a_decomposed_query(indexed, sample_chunks):
    target = sample_chunks[2]
    decomposed = unicodedata.normalize("NFD", target.context_text)
    assert retrieve(decomposed, k=1)[0].chunk.id == target.id


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_rejects_an_empty_query_before_connecting(wired, query):
    with pytest.raises(InvalidQueryError, match="query is empty"):
        retrieve(query)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_rejects_a_small_k_before_connecting(wired):
    with pytest.raises(InvalidQueryError, match="k must be at least 1"):
        retrieve("thử việc", k=0)
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_retrieve_reports_a_missing_collection_without_loading_the_model(wired):
    with pytest.raises(CollectionNotIndexedError, match="uv run python -m ingestion.index"):
        retrieve("thử việc")
    assert wired.model_loads == 0


def test_retrieve_reports_an_unreachable_server_without_loading_the_model(monkeypatch):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    with pytest.raises(QdrantUnavailableError, match="docker compose up -d qdrant"):
        retrieve("thử việc")


# --- search command ---------------------------------------------------------------


def test_search_cli_prints_ranked_results(indexed, sample_chunks, capsys):
    target = sample_chunks[7]

    assert search.main([target.context_text, "-k", "2"]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0] == f"1. 1.0000  {target.id}  Điều {target.article}. {target.article_title}"
    assert lines[1] == "   " + " ".join(target.text.split())[:160]
    assert lines[2].startswith("2. ")


def test_search_cli_reports_errors_without_a_traceback(monkeypatch, capsys):
    def not_indexed(_query, _k):
        raise CollectionNotIndexedError("collection x does not exist. Create it with: ...")

    monkeypatch.setattr(search, "retrieve", not_indexed)
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "collection x does not exist. Create it with: ..."


@pytest.mark.parametrize(
    ("args", "message"), [(["   "], "query is empty"), (["x", "-k", "0"], "k must be at least 1")]
)
def test_search_cli_rejects_a_bad_query_before_connecting(wired, capsys, args, message):
    assert search.main(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert captured.out == ""
    assert (wired.connects, wired.model_loads) == (0, 0)


def test_search_cli_reports_a_missing_collection(wired, capsys):
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "uv run python -m ingestion.index" in captured.err
    assert captured.out == ""
    assert wired.model_loads == 0


def test_search_cli_reports_an_unreachable_server(monkeypatch, capsys):
    settings = Settings(_env_file=None, qdrant_url="http://127.0.0.1:9")
    monkeypatch.setattr(retriever_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        retriever_module, "build_embedder", lambda _settings: pytest.fail("model was loaded")
    )
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "docker compose up -d qdrant" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_search_cli_reports_a_server_that_fails_after_the_retriever_was_cached(
    indexed, sample_chunks, monkeypatch, capsys
):
    text = sample_chunks[0].context_text
    assert search.main([text]) == 0
    capsys.readouterr()

    monkeypatch.setattr(indexed.client, "query_points", fail_transport)
    assert search.main([text]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1


def test_search_cli_reports_invalid_settings(monkeypatch, capsys):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    assert search.main(["thử việc"]) == 1
    captured = capsys.readouterr()
    assert "embedding_batch_size" in captured.err
    assert captured.out == ""
